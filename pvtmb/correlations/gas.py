"""Natural gas properties: pseudo-critical properties, z-factor, Bg, viscosity.

All functions take field units: pressure in psia, temperature in degF unless
the argument name says otherwise (``t_rankine``, ``tpr``...).

References
----------
- Sutton, R.P. (1985). Compressibility factors for high-molecular-weight
  reservoir gases. SPE 14265.
- Wichert, E. & Aziz, K. (1972). Calculate Z's for sour gases. Hydrocarbon
  Processing 51(5).
- Hall, K.R. & Yarborough, L. (1973). A new equation of state for
  Z-factor calculations. Oil & Gas Journal, June 18.
- Dranchuk, P.M. & Abou-Kassem, J.H. (1975). Calculation of Z factors for
  natural gases using equations of state. JCPT 14(3).
- Lee, A.L., Gonzalez, M.H. & Eakin, B.E. (1966). The viscosity of natural
  gases. JPT 18(8).
"""

from __future__ import annotations

import math

import numpy as np

from ..numerics import newton_bracketed
from ..units import FT3_PER_BBL, PSC_PSIA, RANKINE_OFFSET, TSC_DEGF

AIR_MOLAR_MASS = 28.9625  # lb/lbmol
R_FIELD = 10.7316  # psia ft3 / (lbmol degR)


def _vectorize(func):
    """Let a scalar (tpr, ppr) solver accept broadcastable arrays."""
    vec = np.vectorize(func, otypes=[float])

    def wrapper(tpr, ppr):
        out = vec(tpr, ppr)
        return float(out) if out.ndim == 0 else out

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.scalar = func
    return wrapper


# ---------------------------------------------------------------------------
# Pseudo-critical properties
# ---------------------------------------------------------------------------

def pseudo_critical_sutton(gas_gravity):
    """Sutton (1985) pseudo-critical temperature (degR) and pressure (psia)."""
    g = np.asarray(gas_gravity, dtype=float)
    tpc = 169.2 + 349.5 * g - 74.0 * g**2
    ppc = 756.8 - 131.0 * g - 3.6 * g**2
    return (float(tpc), float(ppc)) if g.ndim == 0 else (tpc, ppc)


def wichert_aziz(tpc, ppc, y_h2s: float = 0.0, y_co2: float = 0.0):
    """Wichert-Aziz correction of pseudo-criticals for sour gas.

    ``y_h2s`` and ``y_co2`` are mole fractions. Returns corrected (Tpc, ppc).
    """
    a = y_h2s + y_co2
    eps = 120.0 * (a**0.9 - a**1.6) + 15.0 * (y_h2s**0.5 - y_h2s**4)
    tpc_c = tpc - eps
    ppc_c = ppc * tpc_c / (tpc + y_h2s * (1.0 - y_h2s) * eps)
    return tpc_c, ppc_c


def pseudo_reduced(p, t_degf, gas_gravity, y_h2s: float = 0.0, y_co2: float = 0.0):
    """Pseudo-reduced temperature and pressure (Sutton + Wichert-Aziz)."""
    tpc, ppc = pseudo_critical_sutton(gas_gravity)
    if y_h2s or y_co2:
        tpc, ppc = wichert_aziz(tpc, ppc, y_h2s, y_co2)
    tpr = (np.asarray(t_degf, dtype=float) + RANKINE_OFFSET) / tpc
    ppr = np.asarray(p, dtype=float) / ppc
    return tpr, ppr


# ---------------------------------------------------------------------------
# z-factor
# ---------------------------------------------------------------------------

@_vectorize
def z_hall_yarborough(tpr: float, ppr: float) -> float:
    """z-factor from the Hall-Yarborough (1973) equation of state.

    Solves for the reduced density ``y`` in

        -A ppr + (y + y^2 + y^3 - y^4) / (1 - y)^3 - B y^2 + C y^D = 0

    with ``t = 1 / Tpr``, then ``z = A ppr / y``. Fit to the Standing-Katz
    chart for 1.0 < Tpr <= 3.0 and ppr <= 24.
    """
    if ppr <= 0.0:
        return 1.0
    t = 1.0 / tpr
    a = 0.06125 * t * math.exp(-1.2 * (1.0 - t) ** 2)
    b = t * (14.76 - 9.76 * t + 4.58 * t * t)
    c = t * (90.7 - 242.2 * t + 42.4 * t * t)
    d = 2.18 + 2.82 * t

    def f(y):
        return (-a * ppr + (y + y * y + y**3 - y**4) / (1.0 - y) ** 3
                - b * y * y + c * y**d)

    def dfdy(y):
        return ((1.0 + 4.0 * y + 4.0 * y * y - 4.0 * y**3 + y**4) / (1.0 - y) ** 4
                - 2.0 * b * y + c * d * y ** (d - 1.0))

    # f(0) = -A ppr < 0 and f -> +inf as y -> 1, so (0, 1) always brackets
    y = newton_bracketed(f, dfdy, a * ppr, 1e-15, 1.0 - 1e-12)
    return a * ppr / y


_DAK = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165,
        0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)


@_vectorize
def z_dranchuk_abou_kassem(tpr: float, ppr: float) -> float:
    """z-factor from the Dranchuk-Abou-Kassem (1975) 11-constant EOS.

    Solved in terms of reduced density ``rho_r = 0.27 ppr / (z Tpr)``. Fit to
    the Standing-Katz chart for 1.0 <= Tpr <= 3.0 and 0.2 <= ppr < 30.
    """
    if ppr <= 0.0:
        return 1.0
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _DAK
    c1 = a1 + a2 / tpr + a3 / tpr**3 + a4 / tpr**4 + a5 / tpr**5
    c2 = a6 + a7 / tpr + a8 / tpr**2
    c3 = a9 * (a7 / tpr + a8 / tpr**2)
    c4 = a10 / tpr**3
    k = 0.27 * ppr / tpr

    def z_of(rho):
        e = math.exp(-a11 * rho * rho)
        return (1.0 + c1 * rho + c2 * rho**2 - c3 * rho**5
                + c4 * rho**2 * (1.0 + a11 * rho**2) * e)

    def f(rho):
        return z_of(rho) - k / rho

    def dfdrho(rho):
        e = math.exp(-a11 * rho * rho)
        dz = (c1 + 2.0 * c2 * rho - 5.0 * c3 * rho**4
              + c4 * e * (2.0 * rho + 2.0 * a11 * rho**3 - 2.0 * a11**2 * rho**5))
        return dz + k / rho**2

    hi = 3.0
    while f(hi) < 0.0:  # extremely high ppr; widen the bracket
        hi *= 2.0
    rho = newton_bracketed(f, dfdrho, k, 1e-15, hi)
    return k / rho


Z_METHODS = {
    "dak": z_dranchuk_abou_kassem,
    "dranchuk_abou_kassem": z_dranchuk_abou_kassem,
    "hall_yarborough": z_hall_yarborough,
    "hy": z_hall_yarborough,
}


def z_factor(p, t_degf, gas_gravity, *, method: str = "dak",
             y_h2s: float = 0.0, y_co2: float = 0.0):
    """Gas compressibility factor at pressure ``p`` (psia), temperature (degF)."""
    try:
        solver = Z_METHODS[method.lower()]
    except KeyError:
        raise ValueError(f"unknown z-factor method {method!r}; choose from {sorted(Z_METHODS)}")
    tpr, ppr = pseudo_reduced(p, t_degf, gas_gravity, y_h2s, y_co2)
    return solver(tpr, ppr)


# ---------------------------------------------------------------------------
# Derived gas properties
# ---------------------------------------------------------------------------

# Bg = (psc / Tsc) * z T / p  [rcf/scf]
_BG_RCF = PSC_PSIA / (TSC_DEGF + RANKINE_OFFSET)


def gas_fvf(p, t_degf, z):
    """Gas formation volume factor in rb/scf (0.005035 z T / p)."""
    p = np.asarray(p, dtype=float)
    out = _BG_RCF / FT3_PER_BBL * np.asarray(z) * (np.asarray(t_degf) + RANKINE_OFFSET) / p
    return float(out) if out.ndim == 0 else out


def gas_density(p, t_degf, z, gas_gravity):
    """Gas density in lbm/ft3."""
    mw = AIR_MOLAR_MASS * gas_gravity
    out = np.asarray(p) * mw / (np.asarray(z) * R_FIELD * (np.asarray(t_degf) + RANKINE_OFFSET))
    return float(out) if np.ndim(out) == 0 else out


def gas_viscosity_lee(p, t_degf, z, gas_gravity):
    """Gas viscosity in cp from Lee-Gonzalez-Eakin (1966)."""
    mw = AIR_MOLAR_MASS * gas_gravity
    t_r = np.asarray(t_degf, dtype=float) + RANKINE_OFFSET
    rho_gcc = gas_density(p, t_degf, z, gas_gravity) / 62.427961  # g/cm3
    k = (9.4 + 0.02 * mw) * t_r**1.5 / (209.0 + 19.0 * mw + t_r)
    x = 3.5 + 986.0 / t_r + 0.01 * mw
    y = 2.4 - 0.2 * x
    out = 1e-4 * k * np.exp(x * np.asarray(rho_gcc) ** y)
    return float(out) if np.ndim(out) == 0 else out

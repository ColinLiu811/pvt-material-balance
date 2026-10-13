"""Black-oil property correlations.

Field units throughout: pressure psia, temperature degF, Rs scf/STB,
Bo rb/STB, viscosity cp, compressibility 1/psi.

The ``rs_*`` / ``bo_*`` correlations describe saturated oil (at or below the
bubble point, where Rs depends on pressure). The ``*_undersaturated``
functions extend a bubble-point value to higher pressures at ``Rs = Rsb``.

References
----------
- Standing, M.B. (1947). A pressure-volume-temperature correlation for
  mixtures of California oils and gases. Drilling & Production Practice, API.
- Vasquez, M. & Beggs, H.D. (1980). Correlations for fluid physical property
  prediction. JPT 32(6).
- Beggs, H.D. & Robinson, J.R. (1975). Estimating the viscosity of crude oil
  systems. JPT 27(9).
"""

from __future__ import annotations

import numpy as np

from ..units import api_to_sg


def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


# ---------------------------------------------------------------------------
# Standing (1947)
# ---------------------------------------------------------------------------

def _standing_x(api, t_degf):
    return 0.0125 * np.asarray(api) - 0.00091 * np.asarray(t_degf)


def rs_standing(p, api, gas_gravity, t_degf):
    """Solution GOR (scf/STB) at saturation pressure ``p``.

    Rs = gg [ (p / 18.2 + 1.4) 10^x ]^1.2048,  x = 0.0125 API - 0.00091 T
    """
    p = np.asarray(p, dtype=float)
    base = (p / 18.2 + 1.4) * 10.0 ** _standing_x(api, t_degf)
    return _out(gas_gravity * np.clip(base, 0.0, None) ** 1.2048)


def pb_standing(rs, api, gas_gravity, t_degf):
    """Bubble-point pressure (psia) for solution GOR ``rs``.

    pb = 18.2 [ (Rs / gg)^0.83 10^-x - 1.4 ]  -- the inverse of
    :func:`rs_standing`.
    """
    rs = np.asarray(rs, dtype=float)
    return _out(18.2 * ((rs / gas_gravity) ** (1.0 / 1.2048)
                        * 10.0 ** (-_standing_x(api, t_degf)) - 1.4))


def bo_standing(rs, api, gas_gravity, t_degf):
    """Saturated oil FVF (rb/STB).

    Bo = 0.9759 + 0.000120 [ Rs (gg / go)^0.5 + 1.25 T ]^1.2
    """
    go = api_to_sg(api)
    f = np.asarray(rs) * np.sqrt(gas_gravity / go) + 1.25 * np.asarray(t_degf)
    return _out(0.9759 + 0.000120 * f**1.2)


# ---------------------------------------------------------------------------
# Vasquez-Beggs (1980)
# ---------------------------------------------------------------------------

# (C1, C2, C3) for Rs and (C1, C2, C3) for Bo, split at 30 API
_VB_RS = {"heavy": (0.0362, 1.0937, 25.7240), "light": (0.0178, 1.1870, 23.931)}
_VB_BO = {"heavy": (4.677e-4, 1.751e-5, -1.811e-8), "light": (4.670e-4, 1.100e-5, 1.337e-9)}


def _vb_coeffs(table, api):
    api = np.asarray(api, dtype=float)
    heavy = np.asarray(table["heavy"])[:, None]
    light = np.asarray(table["light"])[:, None]
    c = np.where(api.reshape(1, -1) <= 30.0, heavy, light)
    return c.reshape((3,) + api.shape)


def separator_gas_gravity(gas_gravity, api, sep_pressure=114.7, sep_temperature=60.0):
    """Gas gravity corrected to the 100 psig reference separator.

    gs = gg [1 + 5.912e-5 API Tsep log10(psep / 114.7)]
    With the default separator conditions the correction vanishes.
    """
    return _out(gas_gravity * (1.0 + 5.912e-5 * np.asarray(api) * sep_temperature
                               * np.log10(sep_pressure / 114.7)))


def rs_vasquez_beggs(p, api, gas_gravity_sep, t_degf):
    """Solution GOR (scf/STB) at saturation pressure ``p``.

    Rs = C1 gs p^C2 exp[C3 API / (T + 460)]
    """
    c1, c2, c3 = _vb_coeffs(_VB_RS, api)
    p = np.asarray(p, dtype=float)
    return _out(c1 * gas_gravity_sep * np.clip(p, 0.0, None) ** c2
                * np.exp(c3 * np.asarray(api) / (np.asarray(t_degf) + 460.0)))


def pb_vasquez_beggs(rs, api, gas_gravity_sep, t_degf):
    """Bubble-point pressure (psia), the algebraic inverse of Rs.

    Equivalent to the published form pb = [(C1' Rs / gs) 10^a]^C2' with
    C1' = 1/C1, C2' = 1/C2 and a = -C3' API / T(degR).
    """
    c1, c2, c3 = _vb_coeffs(_VB_RS, api)
    rs = np.asarray(rs, dtype=float)
    return _out((rs / (c1 * gas_gravity_sep
                       * np.exp(c3 * np.asarray(api) / (np.asarray(t_degf) + 460.0)))) ** (1.0 / c2))


def bo_vasquez_beggs(rs, api, gas_gravity_sep, t_degf):
    """Saturated oil FVF (rb/STB).

    Bo = 1 + C1 Rs + (T - 60)(API / gs)(C2 + C3 Rs)
    """
    c1, c2, c3 = _vb_coeffs(_VB_BO, api)
    rs = np.asarray(rs, dtype=float)
    return _out(1.0 + c1 * rs + (np.asarray(t_degf) - 60.0)
                * (np.asarray(api) / gas_gravity_sep) * (c2 + c3 * rs))


def _vb_compressibility_numerator(rsb, api, gas_gravity_sep, t_degf):
    return 1e-5 * (-1433.0 + 5.0 * np.asarray(rsb) + 17.2 * np.asarray(t_degf)
                   - 1180.0 * gas_gravity_sep + 12.61 * np.asarray(api))


def co_vasquez_beggs(p, rsb, api, gas_gravity_sep, t_degf):
    """Undersaturated isothermal oil compressibility (1/psi), p >= pb.

    co = (-1433 + 5 Rsb + 17.2 T - 1180 gs + 12.61 API) / (1e5 p)
    """
    a = _vb_compressibility_numerator(rsb, api, gas_gravity_sep, t_degf)
    return _out(a / np.asarray(p, dtype=float))


def bo_undersaturated(p, pb, bob, rsb, api, gas_gravity_sep, t_degf):
    """Oil FVF above the bubble point (rb/STB).

    Integrating dBo/Bo = -co dp with the Vasquez-Beggs ``co = A / p`` gives
    Bo = Bob (pb / p)^A. Used with either Standing or Vasquez-Beggs Bob.
    """
    a = _vb_compressibility_numerator(rsb, api, gas_gravity_sep, t_degf)
    return _out(np.asarray(bob) * (np.asarray(pb) / np.asarray(p, dtype=float)) ** a)


# ---------------------------------------------------------------------------
# Viscosity: Beggs-Robinson (1975) below pb, Vasquez-Beggs above pb
# ---------------------------------------------------------------------------

def mu_dead_beggs_robinson(api, t_degf):
    """Gas-free (dead) oil viscosity in cp.

    mu_od = 10^x - 1,  x = T^-1.163 exp(6.9824 - 0.04658 API)
    """
    x = np.asarray(t_degf, dtype=float) ** -1.163 * np.exp(6.9824 - 0.04658 * np.asarray(api))
    return _out(10.0**x - 1.0)


def mu_live_beggs_robinson(mu_dead, rs):
    """Gas-saturated (live) oil viscosity in cp.

    mu_o = A mu_od^B,  A = 10.715 (Rs + 100)^-0.515,  B = 5.44 (Rs + 150)^-0.338
    """
    rs = np.asarray(rs, dtype=float)
    a = 10.715 * (rs + 100.0) ** -0.515
    b = 5.44 * (rs + 150.0) ** -0.338
    return _out(a * np.asarray(mu_dead) ** b)


def mu_undersaturated_vasquez_beggs(p, pb, mu_ob):
    """Undersaturated oil viscosity in cp, p >= pb.

    mu_o = mu_ob (p / pb)^m,  m = 2.6 p^1.187 exp(-11.513 - 8.98e-5 p)
    """
    p = np.asarray(p, dtype=float)
    m = 2.6 * p**1.187 * np.exp(-11.513 - 8.98e-5 * p)
    return _out(np.asarray(mu_ob) * (p / np.asarray(pb)) ** m)

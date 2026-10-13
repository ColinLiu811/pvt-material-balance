"""Black-oil property correlations.

Field units throughout: pressure psia, temperature degF, Rs scf/STB,
Bo rb/STB, viscosity cp, compressibility 1/psi.

The ``*_saturated`` functions are valid at or below the bubble point, where
``Rs`` is a function of pressure; the ``*_undersaturated`` functions extend a
bubble-point value to higher pressures at constant ``Rs = Rsb``.

References
----------
- Standing, M.B. (1947). A pressure-volume-temperature correlation for
  mixtures of California oils and gases. Drilling & Production Practice, API.
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

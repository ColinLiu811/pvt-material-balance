"""Formation water properties (field units).

Reference: McCain, W.D. (1990). The Properties of Petroleum Fluids, 2nd ed.
"""

from __future__ import annotations

import numpy as np

# Typical gas-free brine compressibility, 1/psi; good enough for material
# balance where the water term is small.
DEFAULT_CW = 3.0e-6


def bw_mccain(p, t_degf):
    """Water FVF (rb/STB) for gas-saturated pure water.

    Bw = (1 + dVwp)(1 + dVwt), McCain (1990) eqs. for 100-260 degF and
    pressures up to 5000 psia.
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t_degf, dtype=float)
    dvwt = -1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * t**2
    dvwp = (-1.95301e-9 * p * t - 1.72834e-13 * p**2 * t
            - 3.58922e-7 * p - 2.25341e-10 * p**2)
    out = (1.0 + dvwp) * (1.0 + dvwt)
    return float(out) if out.ndim == 0 else out

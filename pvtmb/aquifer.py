"""Aquifer influx models in the form We = C * W(p, t).

Each model is written as an unknown aquifer constant ``C`` times a known
"influx function" ``W`` of the pressure history, which keeps the material
balance linear in its unknowns:

    F / Et = N + C * W / Et

=============  ==========================  ====================================
model          W                           C
=============  ==========================  ====================================
pot            pi - p                      (cw + cf) Wi f        [rb/psi]
schilthuis     integral of (pi - p) dt      steady-state constant [rb/(psi day)]
=============  ==========================  ====================================

References
----------
- Schilthuis, R.J. (1936). Active oil and reservoir energy. Trans. AIME 118.
- Ahmed, T. Reservoir Engineering Handbook, ch. 10 (water influx).
"""

from __future__ import annotations

import numpy as np

from .numerics import cumulative_trapezoid

MODELS = ("pot", "schilthuis")


def pot_influx_function(p) -> np.ndarray:
    """Pressure drop pi - p for a small (pot) aquifer, psi."""
    p = np.asarray(p, dtype=float)
    return p[0] - p


def schilthuis_influx_function(p, t) -> np.ndarray:
    """Cumulative pressure-drop integral for steady-state influx, psi-days.

    ``t`` is elapsed time in days. Uses the trapezoid rule, i.e. linear
    pressure between survey points.
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing for the Schilthuis model")
    return cumulative_trapezoid(p[0] - p, t)


def influx_function(model: str, p, t=None) -> np.ndarray:
    """Dispatch to the named aquifer model."""
    model = model.lower()
    if model == "pot":
        return pot_influx_function(p)
    if model == "schilthuis":
        if t is None:
            raise ValueError("the Schilthuis model needs a time column 't' (days)")
        return schilthuis_influx_function(p, t)
    raise ValueError(f"unknown aquifer model {model!r}; choose from {MODELS}")


def pot_aquifer_volume(constant: float, cw: float, cf: float, f: float = 1.0) -> float:
    """Initial aquifer water volume Wi (bbl) implied by a pot-aquifer constant.

    ``f`` is the fraction of the reservoir perimeter in contact with the
    aquifer (encroachment angle / 360).
    """
    return constant / ((cw + cf) * f)

"""Forward material-balance model to generate histories with known answers.

Given N, m, an aquifer influx history and a pressure decline, solve the
material balance for cumulative oil production. Fitting the generated
history must give back the inputs, which makes this a solver test that does
not depend on any textbook transcription.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .material_balance import mb_terms


def synthetic_history(fluid, *, N: float, pressures, m: float = 0.0, swi: float = 0.2,
                      cw: float = 3e-6, cf: float = 4e-6, we=None, rp=None,
                      wp=None, t=None) -> pd.DataFrame:
    """Production history consistent with ``F = N Et + We``.

    ``pressures`` starts at the initial pressure. ``rp`` (cumulative
    producing GOR, scf/STB) defaults to a ramp from Rsi that mimics free gas
    production below the bubble point. ``we`` is the cumulative water influx
    in rb (default none) and ``wp`` cumulative water production in STB.
    """
    p = np.asarray(pressures, dtype=float)
    n = p.size
    bo, rs, bg, bw = fluid.bo(p), fluid.rs(p), fluid.bg(p), fluid.bw(p)
    we = np.zeros(n) if we is None else np.asarray(we, dtype=float)
    wp = np.zeros(n) if wp is None else np.asarray(wp, dtype=float)
    if rp is None:
        # free gas production grows as pressure falls below pb
        rp = rs[0] + 3.0 * np.maximum(rs[0] - rs, 0.0) + 2.0 * (rs[0] - fluid.rs(p).min()) * (p[0] - p) / p[0]
    rp = np.asarray(rp, dtype=float) * np.ones(n)

    # expansion terms do not depend on production, so compute them first
    base = pd.DataFrame({"p": p, "np": np.zeros(n), "gp": np.zeros(n), "wp": wp,
                         "bo": bo, "rs": rs, "bg": bg, "bw": bw,
                         "winj": 0.0, "ginj": 0.0})
    terms = mb_terms(base, swi=swi, cw=cw, cf=cf)
    et = terms["Eo"] + m * terms["Eg"] + (1.0 + m) * terms["Ef"]
    F = N * et.to_numpy() + we

    np_ = (F - wp * bw) / (bo + (rp - rs) * bg)
    np_[0] = 0.0
    hist = base.drop(columns=["winj", "ginj"]).assign(np=np_, gp=np_ * rp)
    if t is not None:
        hist.insert(0, "t", np.asarray(t, dtype=float))
    return hist

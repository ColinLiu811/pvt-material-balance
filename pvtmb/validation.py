"""Validation of the correlations and material balance against textbook data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .correlations import gas, oil
from .datasets import AHMED_CRUDES, VIRGINIA_HILLS, VIRGINIA_HILLS_PARAMS
from .diagnostics import identify_drive
from .material_balance import MaterialBalance


def oil_correlations_vs_lab() -> pd.DataFrame:
    """Standing and Vasquez-Beggs predictions for the six Ahmed crudes."""
    d = AHMED_CRUDES
    out = pd.DataFrame(index=d.index)
    out["rsb_lab"] = d.rsb
    out["rs_standing"] = oil.rs_standing(d.pb, d.api, d.gas_gravity, d.temp)
    out["rs_vb"] = oil.rs_vasquez_beggs(d.pb, d.api, d.gas_gravity, d.temp)
    out["bob_lab"] = d.bob
    out["bo_standing"] = oil.bo_standing(d.rsb, d.api, d.gas_gravity, d.temp)
    out["bo_vb"] = oil.bo_vasquez_beggs(d.rsb, d.api, d.gas_gravity, d.temp)
    out["pb_lab"] = d.pb
    out["pb_standing"] = oil.pb_standing(d.rsb, d.api, d.gas_gravity, d.temp)
    out["pb_vb"] = oil.pb_vasquez_beggs(d.rsb, d.api, d.gas_gravity, d.temp)
    return out


def error_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Average absolute relative error (%) of each correlation column."""
    rows = []
    for prop in ("rs", "bo", "pb"):
        lab = table[[c for c in table if c.startswith(prop) and c.endswith("_lab")][0]]
        for method in ("standing", "vb"):
            pred = table[f"{prop}_{method}"]
            are = (pred - lab).abs() / lab * 100.0
            rows.append({"property": prop, "correlation": method,
                         "AARE_%": are.mean(), "max_%": are.max()})
    return pd.DataFrame(rows)


def z_factor_cross_check(tpr=(1.2, 1.5, 2.0, 3.0), ppr=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)) -> pd.DataFrame:
    """Hall-Yarborough vs Dranchuk-Abou-Kassem on a (Tpr, ppr) grid."""
    rows = []
    for t in tpr:
        for p in ppr:
            hy = gas.z_hall_yarborough(t, p)
            dak = gas.z_dranchuk_abou_kassem(t, p)
            rows.append({"Tpr": t, "ppr": p, "z_HY": hy, "z_DAK": dak,
                         "diff_%": (hy - dak) / dak * 100.0})
    return pd.DataFrame(rows)


def virginia_hills() -> tuple[MaterialBalance, object]:
    """Material balance and drive diagnosis for the Virginia Hills example."""
    mb = MaterialBalance(VIRGINIA_HILLS, **VIRGINIA_HILLS_PARAMS)
    return mb, identify_drive(mb)


def report() -> str:
    """Plain-text validation report."""
    fmt = {"float_format": lambda v: f"{v:,.4g}"}
    oil_table = oil_correlations_vs_lab()
    z = z_factor_cross_check()
    mb, diag = virginia_hills()
    t = mb.terms.iloc[1:]
    parts = [
        "== Oil correlations vs lab data (Ahmed, six crudes) ==",
        oil_table.to_string(**fmt),
        "",
        error_summary(oil_table).to_string(index=False, **fmt),
        "",
        "== z-factor: Hall-Yarborough vs Dranchuk-Abou-Kassem ==",
        z.to_string(index=False, **fmt),
        f"mean |diff| = {np.abs(z['diff_%']).mean():.2f}%   max |diff| = {np.abs(z['diff_%']).max():.2f}%",
        "",
        "== Havlena-Odeh: Virginia Hills Beaverhill Lake (Ahmed) ==",
        pd.DataFrame({"p": t.p, "F_Mbbl": t.F / 1e3, "Eo": t.Eo, "Eo+Efw": t.Eo + t.Ef}).to_string(index=False, **fmt),
        "",
        diag.report(),
        "Book answer: N ~ 257 MMSTB (line drawn by eye); volumetric estimate 270.6 MMSTB.",
    ]
    return "\n".join(parts)

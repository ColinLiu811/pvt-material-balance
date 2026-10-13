"""Drive-mechanism identification.

Two complementary tools:

1. **Campbell plot** - F/Et against F. If the assumed model (no aquifer,
   given m) is right, F/Et equals N at every point, so the plot is flat.
   Any trend means the model is missing (or overstating) a source of
   energy. A steadily rising plot is the signature of an active aquifer; a
   small (pot) aquifer gives a hump that falls once the reservoir is below
   the bubble point. A wrong m also tilts the plot, so a trend is a prompt
   to test models, not a verdict on its own.

   Above the bubble point a pot aquifer cannot be separated from extra
   OOIP: both W = pi - p and Et are proportional to the pressure drop.

2. **Drive indices** (Pirson) - after fitting, each term of the material
   balance divided by the hydrocarbon withdrawal Np [Bo + (Rp - Rs) Bg]
   gives the fraction of production driven by that mechanism:

       DDI = N Eo / A                    depletion (oil + solution gas)
       SDI = N m Eg / A                  gas-cap expansion
       WDI = (We - Wp Bw) / A            net water influx
       EDI = N (1 + m) Ef / A            rock and connate-water expansion

   They sum to one when the model fits exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .material_balance import MaterialBalance, MBResult
from .numerics import fit_line

# spread of F/Et (relative to its median) above which the plot is not flat
TREND_THRESHOLD = 0.10
# an aquifer model must cut the volumetric model's unexplained variance of F
# by at least this factor to be preferred (guards against overfitting)
MIN_IMPROVEMENT = 2.0
# ignore points whose expansion is tiny compared with the last one: there the
# ratio F/Et is dominated by rounding in the PVT data
MIN_ET_FRACTION = 0.05


@dataclass
class CampbellTrend:
    x: np.ndarray  # F
    y: np.ndarray  # F / Et
    used: np.ndarray  # mask of points used for the trend
    relative_change: float  # (max - min) / median of F/Et over the used points
    trend: str  # "flat", "rising", "falling" or "hump"


def campbell_trend(mb: MaterialBalance, m: float = 0.0) -> CampbellTrend:
    """Shape of F/Et vs F for a volumetric model with gas-cap ratio ``m``."""
    t = mb.terms.iloc[1:]
    et = (t["Eo"] + m * t["Eg"] + (1.0 + m) * t["Ef"]).to_numpy()
    F = t["F"].to_numpy()
    y = F / et
    used = et >= MIN_ET_FRACTION * np.abs(et).max()
    if used.sum() < 3:
        used[:] = True
    yu = y[used]
    rel = float((yu.max() - yu.min()) / np.median(yu))
    if rel <= TREND_THRESHOLD:
        trend = "flat"
    else:
        # judge direction from a smoothed start / peak / end comparison
        k = max(1, yu.size // 4)
        start, end, peak = yu[:k].mean(), yu[-k:].mean(), yu.max()
        tol = 0.5 * TREND_THRESHOLD * np.median(yu)
        if peak - max(start, end) > tol:
            trend = "hump"
        else:
            trend = "rising" if end > start else "falling"
    return CampbellTrend(x=F, y=y, used=used, relative_change=rel, trend=trend)


def drive_indices(mb: MaterialBalance, result: MBResult) -> pd.DataFrame:
    """Pirson drive indices at each history point after a fit."""
    h, t = mb.history, mb.terms
    N, m = result.N, result.m
    net_water = (h["wp"] - h["winj"]) * h["bw"]
    bg = h["bg"].fillna(0.0)
    A = t["F"] - net_water + h["ginj"] * bg
    with np.errstate(invalid="ignore", divide="ignore"):
        df = pd.DataFrame({
            "p": h["p"],
            "DDI": N * t["Eo"] / A,
            "SDI": N * m * t["Eg"] / A,
            "WDI": (result.We - net_water) / A,
            "EDI": N * (1.0 + m) * t["Ef"] / A,
        })
    df["total"] = df[["DDI", "SDI", "WDI", "EDI"]].sum(axis=1)
    return df.iloc[1:].reset_index(drop=True)


_INDEX_NAMES = {
    "SDI": "gas-cap drive",
    "WDI": "water drive",
    "EDI": "rock and connate-water expansion",
}


@dataclass
class DriveDiagnosis:
    """Drive-mechanism verdict plus the evidence behind it."""

    mechanism: str
    result: MBResult
    campbell: CampbellTrend
    indices: pd.DataFrame
    alternatives: dict[str, MBResult] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def final_indices(self) -> dict[str, float]:
        row = self.indices.iloc[-1]
        return {k: float(row[k]) for k in ("DDI", "SDI", "WDI", "EDI")}

    def report(self) -> str:
        idx = self.final_indices
        lines = [
            f"Drive mechanism : {self.mechanism}",
            f"Campbell plot   : {self.campbell.trend} "
            f"(F/Et varies {self.campbell.relative_change:.1%} across the history)",
            "Drive indices   : " + ", ".join(f"{k} {v:.2f}" for k, v in idx.items())
            + "  (latest point)",
            "",
            self.result.summary(),
        ]
        if self.alternatives:
            lines.append("")
            lines.append("Models compared (R^2 of F):")
            for name, res in self.alternatives.items():
                lines.append(f"  {name:<12} N = {res.N / 1e6:9,.2f} MMSTB   R^2 = {_r2_F(res.terms, res):.5f}")
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def _r2_F(terms: pd.DataFrame, result: MBResult) -> float:
    """R^2 of the fitted model in F space, comparable across model forms."""
    F = terms["F"].to_numpy()[1:]
    pred = result.N * result.Et[1:] + result.We[1:]
    sse = float(((F - pred) ** 2).sum())
    sst = float(((F - F.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else 1.0


def _name_mechanism(idx: dict[str, float], below_pb: bool) -> str:
    depletion = "solution-gas drive" if below_pb else "undersaturated oil expansion"
    names = {"DDI": depletion, **_INDEX_NAMES}
    ranked = sorted(idx.items(), key=lambda kv: kv[1], reverse=True)
    top, value = ranked[0]
    if value >= 0.5 or ranked[1][1] < 0.1:
        return names[top]
    second = ranked[1][0]
    return f"combination drive ({names[top]} + {names[second]})"


def identify_drive(mb: MaterialBalance, m: float = 0.0, aquifer: str | None = "auto") -> DriveDiagnosis:
    """Identify the dominant drive mechanism of a reservoir.

    Steps: check the Campbell trend for the volumetric model; if it rises
    and ``aquifer="auto"``, fit each available aquifer model and keep the
    one that best reproduces F; then compute drive indices from the chosen
    fit. Pass ``aquifer=None`` to force a volumetric model, or a model name
    to force that aquifer.
    """
    trend = campbell_trend(mb, m)
    notes: list[str] = []
    alternatives: dict[str, MBResult] = {}

    if aquifer == "auto":
        volumetric = mb.fit(m=m)
        alternatives["volumetric"] = volumetric
        result = volumetric
        if trend.trend != "flat":
            models = ["pot"] + (["schilthuis"] if "t" in mb.history else [])
            for name in models:
                res = mb.fit(m=m, aquifer=name)
                if res.N > 0 and res.aquifer_constant > 0:
                    alternatives[name] = res
            if m == 0 and mb.initially_saturated:
                res = mb.fit(m="estimate")
                if res.N > 0 and res.m > 0:
                    alternatives["gas cap"] = res
            best = max(alternatives.values(), key=lambda r: _r2_F(mb.terms, r))
            miss_vol = 1.0 - _r2_F(mb.terms, volumetric)
            miss_best = 1.0 - _r2_F(mb.terms, best)
            if best is not volumetric and miss_best * MIN_IMPROVEMENT < miss_vol:
                result = best
            else:
                notes.append(f"Campbell plot is not flat ({trend.trend}) but no alternative model explains it clearly "
                             "better; check PVT, pressures, compressibility and the assumed m")
            if "gas cap" in alternatives and "pot" in alternatives:
                notes.append("gas-cap and pot-aquifer expansion both scale with pressure drop and "
                             "are hard to tell apart from production data alone; use geology / "
                             "logs to confirm which is present")
            if trend.trend == "falling" and result is volumetric:
                notes.append("a falling Campbell plot can mean m is overestimated")
            if "t" not in mb.history:
                notes.append("no time column: the Schilthuis aquifer model was not tried")
    else:
        result = mb.fit(m=m, aquifer=aquifer)

    indices = drive_indices(mb, result)
    mechanism = _name_mechanism(
        {k: float(indices[k].iloc[-1]) for k in ("DDI", "SDI", "WDI", "EDI")},
        below_pb=not mb.above_bubble_point,
    )
    return DriveDiagnosis(mechanism=mechanism, result=result, campbell=trend,
                          indices=indices, alternatives=alternatives, notes=notes)

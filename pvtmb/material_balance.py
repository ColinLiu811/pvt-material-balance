"""Havlena-Odeh material balance for oil reservoirs.

The general material balance equation, written as underground withdrawal
equals expansion plus influx (Havlena & Odeh, 1963):

    F = N (Eo + m Eg + Efw) + We

with

    F   = Np [Bo + (Rp - Rs) Bg] + Wp Bw - Winj Bw - Ginj Bg      (rb)
    Eo  = (Bo - Boi) + (Rsi - Rs) Bg                              (rb/STB)
    Eg  = Boi (Bg / Bgi - 1)                                      (rb/STB)
    Efw = (1 + m) Boi (cw Swi + cf) / (1 - Swi) * (pi - p)        (rb/STB)

N is the original oil in place (STB), m the initial gas-cap to oil-zone
hydrocarbon pore volume ratio, and We the cumulative water influx (rb).

Arranging the equation so that the unknowns appear as the slope and
intercept of a straight line turns OOIP estimation into linear regression
and makes departures from the assumed drive mechanism visible as
curvature.

Reference: Havlena, D. & Odeh, A.S. (1963). The material balance as an
equation of a straight line. JPT 15(8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import aquifer as aq
from .numerics import LinearFit, fit_line, lstsq

# history columns (field units): p psia, np STB, gp scf, wp STB, bo rb/STB,
# rs scf/STB, bg rb/scf, bw rb/STB, winj STB, ginj scf, t days
REQUIRED_COLUMNS = ("p", "np")
PVT_COLUMNS = ("bo", "rs", "bg")
DEFAULTS = {"wp": 0.0, "winj": 0.0, "ginj": 0.0}


def prepare_history(history: pd.DataFrame, fluid=None) -> pd.DataFrame:
    """Validate a production/pressure history and fill in missing PVT.

    The first row must be the initial condition (``np == 0``). Column names
    are case-insensitive. Missing ``bo``/``rs``/``bg``/``bw`` columns are
    computed from ``fluid`` (a :class:`~pvtmb.fluid.BlackOil`) when given;
    a missing ``bw`` defaults to 1.0. Without a fluid model, ``rs`` and ``bg``
    may be omitted for an undersaturated history: Rs is then constant, free
    gas never forms and Bg drops out of every term. A missing ``gp`` is
    allowed only while the reservoir stays above the bubble point
    (then ``Rp = Rsi``).
    """
    df = history.rename(columns=str.lower).copy()
    for col in REQUIRED_COLUMNS:
        if col not in df:
            raise ValueError(f"history is missing required column {col!r}")
    if len(df) < 3:
        raise ValueError("need the initial state plus at least two production points")
    if df["np"].iloc[0] != 0:
        raise ValueError("first row must be the initial condition (np = 0)")
    if (np.diff(df["np"]) < 0).any():
        raise ValueError("cumulative oil production must be non-decreasing")

    if fluid is None and "rs" not in df and "bg" not in df:
        df["rs"] = 0.0  # undersaturated: only differences in Rs matter
        df["bg"] = np.nan  # undefined, flagged so a gas-cap fit is refused
    for col in PVT_COLUMNS:
        if col not in df:
            if fluid is None:
                raise ValueError(f"history has no {col!r} column; pass a fluid model to compute PVT")
            df[col] = getattr(fluid, col)(df["p"].to_numpy())
    if "bw" not in df:
        df["bw"] = fluid.bw(df["p"].to_numpy()) if fluid is not None else 1.0
    for col, default in DEFAULTS.items():
        if col not in df:
            df[col] = default

    if "gp" not in df:
        rsi = df["rs"].iloc[0]
        if not np.allclose(df["rs"], rsi):
            raise ValueError("history has no 'gp' column but pressure falls below the bubble "
                             "point; cumulative gas is needed to compute Rp")
        df["gp"] = df["np"] * rsi
    return df.reset_index(drop=True)


def mb_terms(history: pd.DataFrame, *, swi: float, cw: float, cf: float) -> pd.DataFrame:
    """Havlena-Odeh withdrawal and expansion terms for each history row.

    Returns a frame with ``F``, ``Eo``, ``Eg``, ``Ef`` and ``dp`` where
    ``Ef = Boi (cw Swi + cf) / (1 - Swi) dp`` is the rock/water expansion
    *per unit (1 + m)*; the full Efw is ``(1 + m) Ef``.
    """
    df = history
    boi, rsi, bgi = df["bo"].iloc[0], df["rs"].iloc[0], df["bg"].iloc[0]
    pi = df["p"].iloc[0]

    with np.errstate(invalid="ignore", divide="ignore"):
        rp = np.where(df["np"] > 0, df["gp"] / df["np"], rsi)
    F = (df["np"] * (df["bo"] + (rp - df["rs"]) * df["bg"])
         + (df["wp"] - df["winj"]) * df["bw"] - df["ginj"] * df["bg"])
    dp = pi - df["p"]
    if np.isnan(bgi):
        # no gas data: undersaturated, so Eg and the Rs terms drop out
        eg = np.zeros(len(df))
        eo = df["bo"] - boi
        F = df["np"] * df["bo"] + (df["wp"] - df["winj"]) * df["bw"]
    else:
        eg = boi * (df["bg"] / bgi - 1.0)
        eo = (df["bo"] - boi) + (rsi - df["rs"]) * df["bg"]
    out = pd.DataFrame({
        "p": df["p"],
        "dp": dp,
        "rp": rp,
        "F": F,
        "Eo": eo,
        "Eg": eg,
        "Ef": boi * (cw * swi + cf) / (1.0 - swi) * dp,
    })
    if "t" in df:
        out.insert(0, "t", df["t"])
    out.insert(1, "np", df["np"])
    return out


@dataclass
class MBResult:
    """Outcome of a Havlena-Odeh fit."""

    N: float
    m: float
    method: str
    fit: LinearFit
    terms: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    xlabel: str
    ylabel: str
    N_stderr: float = float("nan")
    m_stderr: float = float("nan")
    aquifer: str | None = None
    aquifer_constant: float = 0.0
    aquifer_constant_stderr: float = float("nan")
    We: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.We is None:
            self.We = np.zeros(len(self.terms))

    @property
    def Et(self) -> np.ndarray:
        """Total expansion per STB, Eo + m Eg + (1 + m) Ef."""
        t = self.terms
        return (t["Eo"] + self.m * t["Eg"] + (1.0 + self.m) * t["Ef"]).to_numpy()

    def summary(self) -> str:
        lines = [
            f"Method      : {self.method}",
            f"OOIP N      : {self.N / 1e6:,.2f} MMSTB  (+/- {self.N_stderr / 1e6:,.2f})",
            f"Gas cap m   : {self.m:.4f}" + (f"  (+/- {self.m_stderr:.4f})" if np.isfinite(self.m_stderr) else ""),
            f"R^2         : {self.fit.r2:.5f}  ({self.fit.n} points)",
        ]
        if self.aquifer:
            lines.insert(3, f"Aquifer     : {self.aquifer}, C = {self.aquifer_constant:,.4g} "
                            f"(+/- {self.aquifer_constant_stderr:,.3g}), "
                            f"We = {self.We[-1] / 1e6:,.2f} MMrb at last point")
        lines += [f"Note        : {n}" for n in self.notes]
        return "\n".join(lines)


class MaterialBalance:
    """Havlena-Odeh material balance on a reservoir history.

    Parameters
    ----------
    history:
        DataFrame with at least ``p`` (psia) and ``np`` (STB) and usually
        ``gp`` (scf), ``wp`` (STB) and PVT columns ``bo``, ``rs``, ``bg``,
        ``bw``. The first row is the initial state.
    swi, cw, cf:
        Connate water saturation and water / formation compressibility (1/psi).
    fluid:
        Optional :class:`~pvtmb.fluid.BlackOil` used to fill missing PVT.
    """

    def __init__(self, history: pd.DataFrame, *, swi: float, cw: float = 3e-6,
                 cf: float = 4e-6, fluid=None):
        if not 0.0 <= swi < 1.0:
            raise ValueError("swi must be in [0, 1)")
        self.history = prepare_history(history, fluid)
        self.swi, self.cw, self.cf = swi, cw, cf
        self.terms = mb_terms(self.history, swi=swi, cw=cw, cf=cf)

    @property
    def above_bubble_point(self) -> bool:
        """True when Rs never drops, i.e. no free gas has evolved."""
        rs = self.history["rs"].to_numpy()
        return bool(np.allclose(rs, rs[0]))

    @property
    def initially_saturated(self) -> bool:
        """True when Rs falls from the first production step (pi <= pb), the
        only situation in which an initial gas cap can exist."""
        rs = self.history["rs"].to_numpy()
        return bool(self.has_gas_data and rs[1] < rs[0])

    @property
    def has_gas_data(self) -> bool:
        return not self.history["bg"].isna().any()

    def _rows(self):
        # the initial row is 0 = N * 0 and carries no information
        return self.terms.iloc[1:]

    def influx_function(self, model: str) -> np.ndarray:
        """Aquifer influx function W for every history row (see :mod:`pvtmb.aquifer`)."""
        t = self.history["t"].to_numpy() if "t" in self.history else None
        return aq.influx_function(model, self.history["p"].to_numpy(), t)

    def fit(self, m: float | str = 0.0, aquifer: str | None = None) -> MBResult:
        """Estimate OOIP (and gas-cap size / aquifer strength).

        ``m`` is the known gas-cap ratio (0 for no gas cap) or ``"estimate"``.
        ``aquifer`` is ``None`` (volumetric) or an influx model name from
        :mod:`pvtmb.aquifer`.

        ==============  ==========  ===========================================
        m               aquifer     straight line
        ==============  ==========  ===========================================
        known           none        F = N Et, through the origin, slope N
        "estimate"      none        F/E1 = N + Nm E2/E1, intercept N, slope Nm
        known           model       F/Et = N + C W/Et, intercept N, slope C
        "estimate"      model       F = N E1 + Nm E2 + C W (multiple regression)
        ==============  ==========  ===========================================

        with E1 = Eo + Ef and E2 = Eg + Ef, so that Et = E1 + m E2.
        """
        estimate_m = isinstance(m, str)
        if estimate_m and m != "estimate":
            raise ValueError("m must be a number or 'estimate'")
        if (estimate_m or m) and not self.has_gas_data:
            raise ValueError("a gas-cap fit needs Bg in the history (or a fluid model)")
        if aquifer is not None:
            return self._fit_water_drive(m, aquifer)

        t = self._rows()
        F = t["F"].to_numpy()
        E1 = (t["Eo"] + t["Ef"]).to_numpy()
        E2 = (t["Eg"] + t["Ef"]).to_numpy()

        if estimate_m:
            x, y = E2 / E1, F / E1
            line = fit_line(x, y)
            N, mN = line.intercept, line.slope
            m_hat = mN / N
            # first-order error propagation for m = slope / intercept
            m_err = abs(m_hat) * np.hypot(line.stderr[1] / mN, line.stderr[0] / N) if mN else np.nan
            return MBResult(N=N, m=m_hat, method="F/Eo vs Eg/Eo (gas cap, m estimated)",
                            fit=line, terms=self.terms, x=x, y=y,
                            xlabel="(Eg + Ef) / (Eo + Ef)", ylabel="F / (Eo + Ef)  [STB]",
                            N_stderr=line.stderr[0], m_stderr=m_err)

        Et = E1 + m * E2
        line = fit_line(Et, F, through_origin=True)
        label = "F vs Et" + (" (gas cap, m known)" if m else
                             " (undersaturated)" if self.above_bubble_point else " (solution gas)")
        return MBResult(N=line.slope, m=float(m), method=label, fit=line, terms=self.terms,
                        x=Et, y=F, xlabel="Et = Eo + m Eg + Efw  [rb/STB]", ylabel="F  [rb]",
                        N_stderr=line.stderr[0])

    def _fit_water_drive(self, m, model: str) -> MBResult:
        W_all = self.influx_function(model)
        t = self._rows()
        W = W_all[1:]
        F = t["F"].to_numpy()
        E1 = (t["Eo"] + t["Ef"]).to_numpy()
        E2 = (t["Eg"] + t["Ef"]).to_numpy()

        if isinstance(m, str):
            reg = lstsq(np.column_stack([E1, E2, W]), F, centered_r2=False)
            N, mN, C = reg.coef
            m_hat = mN / N
            m_err = abs(m_hat) * np.hypot(reg.stderr[1] / mN, reg.stderr[0] / N) if mN else np.nan
            return MBResult(N=N, m=m_hat, method=f"F = N E1 + Nm E2 + C W ({model} aquifer, m estimated)",
                            fit=reg, terms=self.terms, x=F - C * W, y=F,
                            xlabel="F - We  [rb]", ylabel="F  [rb]",
                            N_stderr=reg.stderr[0], m_stderr=m_err, aquifer=model,
                            aquifer_constant=C, aquifer_constant_stderr=reg.stderr[2],
                            We=C * W_all)

        Et = E1 + m * E2
        x, y = W / Et, F / Et
        line = fit_line(x, y)
        C = line.slope
        return MBResult(N=line.intercept, m=float(m), method=f"F/Et vs W/Et ({model} aquifer)",
                        fit=line, terms=self.terms, x=x, y=y,
                        xlabel=f"W / Et  ({model} influx function)", ylabel="F / Et  [STB]",
                        N_stderr=line.stderr[0], aquifer=model, aquifer_constant=C,
                        aquifer_constant_stderr=line.stderr[1], We=C * W_all)

"""Matplotlib figures for PVT tables and material balance diagnostics."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .diagnostics import DriveDiagnosis  # noqa: E402
from .material_balance import MBResult  # noqa: E402

_INDEX_COLORS = {"DDI": "#4C72B0", "SDI": "#DD8452", "WDI": "#55A868", "EDI": "#8172B3"}
_INDEX_LABELS = {"DDI": "Depletion (DDI)", "SDI": "Gas cap (SDI)",
                 "WDI": "Water (WDI)", "EDI": "Rock + water expansion (EDI)"}


def plot_pvt(table: pd.DataFrame, pb: float | None = None, title: str = "", path=None):
    """Four-panel PVT summary: Rs, Bo/Bt, viscosities and z/Bg."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    p = table["p"]
    ax = axes[0, 0]
    ax.plot(p, table["rs"], color="#4C72B0")
    ax.set_ylabel("Rs [scf/STB]")
    ax = axes[0, 1]
    ax.plot(p, table["bo"], color="#4C72B0", label="Bo")
    sat = table["p"] <= (pb if pb else np.inf)
    ax.plot(p[sat], table["bt"][sat], color="#DD8452", ls="--", label="Bt")
    ax.set_ylabel("FVF [rb/STB]")
    # Bt explodes at low pressure; keep Bo readable
    ax.set_ylim(0.95 * table["bo"].min(), min(table["bt"].max(), 2.5 * table["bo"].max()) * 1.02)
    ax.legend()
    ax = axes[1, 0]
    ax.plot(p, table["mu_o"], color="#4C72B0", label="oil")
    ax.set_ylabel("Oil viscosity [cp]")
    ax2 = ax.twinx()
    ax2.plot(p, table["mu_g"], color="#55A868", label="gas")
    ax2.set_ylabel("Gas viscosity [cp]", color="#55A868")
    ax = axes[1, 1]
    ax.plot(p, table["z"], color="#8172B3")
    ax.set_ylabel("z-factor", color="#8172B3")
    ax2 = ax.twinx()
    ax2.plot(p, table["bg"] * 1e3, color="#C44E52")
    ax2.set_yscale("log")
    ax2.set_ylabel("Bg [rb/Mscf, log]", color="#C44E52")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        if pb:
            ax.axvline(pb, color="grey", lw=0.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("Pressure [psia]")
    fig.suptitle(title or "Black-oil PVT")
    fig.tight_layout()
    return _finish(fig, path)


def plot_havlena_odeh(result: MBResult, path=None, title: str = ""):
    """The Havlena-Odeh straight line used to estimate N."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x, y = np.asarray(result.x), np.asarray(result.y)
    ax.scatter(x, y, color="#4C72B0", zorder=3, label="history")
    xs = np.linspace(0 if result.fit.coef.size == 1 else x.min(), x.max(), 50)
    if result.fit.coef.size == 1:
        ax.plot(xs, result.fit.slope * xs, color="#C44E52", label=f"slope N = {result.N / 1e6:,.1f} MMSTB")
    elif result.fit.coef.size == 2:
        ax.plot(xs, result.fit.intercept + result.fit.slope * xs, color="#C44E52",
                label=f"intercept N = {result.N / 1e6:,.1f} MMSTB")
    else:
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color="#C44E52", label="1:1")
    ax.set_xlabel(result.xlabel)
    ax.set_ylabel(result.ylabel)
    ax.set_title(title or result.method)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path)


def plot_campbell(diagnosis: DriveDiagnosis, path=None):
    """Campbell plot: F/Et against F for the volumetric model."""
    c = diagnosis.campbell
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(c.x[c.used] / 1e6, c.y[c.used] / 1e6, color="#4C72B0", label="used")
    if (~c.used).any():
        ax.scatter(c.x[~c.used] / 1e6, c.y[~c.used] / 1e6, facecolor="none",
                   edgecolor="grey", label="excluded (tiny Et)")
    # the Campbell plot tests the volumetric model, so compare with its N
    vol = diagnosis.alternatives.get("volumetric")
    if vol is None and diagnosis.result.aquifer is None:
        vol = diagnosis.result
    if vol is not None:
        ax.axhline(vol.N / 1e6, color="#C44E52", ls="--",
                   label=f"volumetric fit N = {vol.N / 1e6:,.1f} MMSTB")
    ax.set_xlabel("F [MMrb]")
    ax.set_ylabel("F / Et [MMSTB]")
    ax.set_title(f"Campbell plot - {c.trend}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path)


def plot_drive_indices(diagnosis: DriveDiagnosis, path=None):
    """Stacked drive indices against pressure (production runs left to right)."""
    idx = diagnosis.indices
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cols = [c for c in ("DDI", "SDI", "WDI", "EDI") if (idx[c].abs() > 1e-6).any()]
    ax.stackplot(np.arange(len(idx)), *[idx[c].clip(lower=0) for c in cols],
                 labels=[_INDEX_LABELS[c] for c in cols],
                 colors=[_INDEX_COLORS[c] for c in cols], alpha=0.85)
    ax.set_xticks(np.arange(len(idx)))
    ax.set_xticklabels([f"{p:.0f}" for p in idx["p"]], rotation=45)
    ax.set_xlabel("Pressure [psia]")
    ax.set_ylabel("Fraction of withdrawal")
    ax.set_ylim(0, max(1.05, float(idx[cols].clip(lower=0).sum(axis=1).max()) * 1.02))
    ax.set_title(f"Drive indices - {diagnosis.mechanism}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(cols), fontsize=8,
              frameon=False)
    fig.tight_layout()
    return _finish(fig, path)


def _finish(fig, path):
    if path is not None:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig

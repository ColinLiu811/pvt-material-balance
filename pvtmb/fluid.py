"""Black-oil fluid model assembled from the individual correlations.

>>> oil = BlackOil(api=35.0, gas_gravity=0.75, temperature=200.0, rsb=600.0)
>>> round(oil.pb)
2570
>>> table = oil.table([4000, 3000, 2000, 1000])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import units
from .correlations import gas, oil, water

CORRELATIONS = ("standing", "vasquez_beggs")

# Column -> (field unit, SI unit) for PVT tables
TABLE_UNITS = {
    "p": ("psia", "MPa"),
    "rs": ("scf/stb", "sm3/sm3"),
    "bo": ("rb/stb", "rm3/sm3"),
    "bt": ("rb/stb", "rm3/sm3"),
    "co": ("1/psi", "1/MPa"),
    "mu_o": ("cp", "mPa.s"),
    "z": (None, None),
    "bg": ("rb/scf", "rm3/sm3_gas"),
    "mu_g": ("cp", "mPa.s"),
    "bw": ("rb/stb", "rm3/sm3"),
}


@dataclass
class BlackOil:
    """Black-oil PVT model from stock-tank oil and gas properties.

    Give either the solution GOR at the bubble point (``rsb``) or the bubble
    point itself (``pb``); the other is derived from the chosen correlation.

    Parameters are in field units (degF, psia, scf/STB); use
    :meth:`from_units` to build one from SI or other inputs.

    ``correlation`` selects Standing or Vasquez-Beggs for Rs, pb and Bo.
    Viscosity always uses Beggs-Robinson below pb and Vasquez-Beggs above,
    undersaturated Bo uses the Vasquez-Beggs compressibility.
    """

    api: float
    gas_gravity: float
    temperature: float
    rsb: float | None = None
    pb: float | None = None
    correlation: str = "standing"
    z_method: str = "dak"
    sep_pressure: float = 114.7
    sep_temperature: float = 60.0
    y_h2s: float = 0.0
    y_co2: float = 0.0
    gas_gravity_sep: float = field(init=False)
    bob: float = field(init=False)

    def __post_init__(self):
        self.correlation = self.correlation.lower()
        if self.correlation not in CORRELATIONS:
            raise ValueError(f"correlation must be one of {CORRELATIONS}, got {self.correlation!r}")
        if (self.rsb is None) == (self.pb is None):
            raise ValueError("specify exactly one of rsb or pb")
        if not 5.0 <= self.api <= 70.0:
            raise ValueError(f"API gravity {self.api} outside 5-70")
        if not 0.5 <= self.gas_gravity <= 1.8:
            raise ValueError(f"gas gravity {self.gas_gravity} outside 0.5-1.8")

        self.gas_gravity_sep = oil.separator_gas_gravity(
            self.gas_gravity, self.api, self.sep_pressure, self.sep_temperature)
        if self.pb is None:
            self.pb = float(self._pb_from_rs(self.rsb))
        else:
            self.rsb = float(self._rs_saturated(self.pb))
        self.bob = float(self._bo_saturated(self.rsb))

    @classmethod
    def from_units(cls, *, api: float, gas_gravity: float, temperature: float,
                   rsb: float | None = None, pb: float | None = None,
                   temperature_unit: str = "degF", pressure_unit: str = "psia",
                   gor_unit: str = "scf/stb", **kwargs) -> "BlackOil":
        """Build a model from inputs in arbitrary units.

        >>> fluid = BlackOil.from_units(api=35, gas_gravity=0.75, temperature=93.3,
        ...                             temperature_unit="degC", rsb=106.9,
        ...                             gor_unit="sm3/sm3")
        >>> round(fluid.temperature), round(fluid.rsb)
        (200, 600)
        """
        t = units.convert(temperature, temperature_unit, "degF")
        if rsb is not None:
            rsb = units.convert(rsb, gor_unit, "scf/stb")
        if pb is not None:
            pb = units.convert(pb, pressure_unit, "psia")
        if "sep_pressure" in kwargs:
            kwargs["sep_pressure"] = units.convert(kwargs["sep_pressure"], pressure_unit, "psia")
        if "sep_temperature" in kwargs:
            kwargs["sep_temperature"] = units.convert(kwargs["sep_temperature"], temperature_unit, "degF")
        return cls(api=api, gas_gravity=gas_gravity, temperature=t, rsb=rsb, pb=pb, **kwargs)

    # -- correlation dispatch -------------------------------------------------

    def _rs_saturated(self, p):
        if self.correlation == "standing":
            return oil.rs_standing(p, self.api, self.gas_gravity, self.temperature)
        return oil.rs_vasquez_beggs(p, self.api, self.gas_gravity_sep, self.temperature)

    def _pb_from_rs(self, rs):
        if self.correlation == "standing":
            return oil.pb_standing(rs, self.api, self.gas_gravity, self.temperature)
        return oil.pb_vasquez_beggs(rs, self.api, self.gas_gravity_sep, self.temperature)

    def _bo_saturated(self, rs):
        if self.correlation == "standing":
            return oil.bo_standing(rs, self.api, self.gas_gravity, self.temperature)
        return oil.bo_vasquez_beggs(rs, self.api, self.gas_gravity_sep, self.temperature)

    # -- oil properties -------------------------------------------------------

    @staticmethod
    def _p(p):
        p = np.asarray(p, dtype=float)
        if np.any(p <= 0):
            raise ValueError("pressures must be positive (psia)")
        return p

    @staticmethod
    def _ret(x):
        x = np.asarray(x, dtype=float)
        return float(x) if x.ndim == 0 else x

    def rs(self, p):
        """Solution GOR, scf/STB (constant at Rsb above the bubble point)."""
        p = self._p(p)
        return self._ret(np.where(p < self.pb, self._rs_saturated(np.minimum(p, self.pb)), self.rsb))

    def bo(self, p):
        """Oil formation volume factor, rb/STB."""
        p = self._p(p)
        sat = self._bo_saturated(self.rs(np.minimum(p, self.pb)))
        under = oil.bo_undersaturated(np.maximum(p, self.pb), self.pb, self.bob, self.rsb,
                                      self.api, self.gas_gravity_sep, self.temperature)
        return self._ret(np.where(p < self.pb, sat, under))

    def bt(self, p):
        """Total (two-phase) FVF, rb/STB: Bo + (Rsb - Rs) Bg."""
        return self._ret(self.bo(p) + (self.rsb - self.rs(p)) * self.bg(p))

    def co(self, p):
        """Oil compressibility, 1/psi.

        Above pb: Vasquez-Beggs. Below pb the apparent compressibility
        includes liberated gas, co = -(1/Bo)(dBo/dp - Bg dRs/dp), evaluated
        with central differences on the saturated branch.
        """
        p = self._p(p)
        under = oil.co_vasquez_beggs(np.maximum(p, self.pb), self.rsb, self.api,
                                     self.gas_gravity_sep, self.temperature)
        ps = np.minimum(p, self.pb)
        h = 1e-4 * ps
        hi, lo = np.minimum(ps + h, self.pb), ps - h
        dbo = (self._bo_saturated(self._rs_saturated(hi))
               - self._bo_saturated(self._rs_saturated(lo))) / (hi - lo)
        drs = (self._rs_saturated(hi) - self._rs_saturated(lo)) / (hi - lo)
        sat = -(dbo - self.bg(ps) * drs) / self._bo_saturated(self._rs_saturated(ps))
        return self._ret(np.where(p < self.pb, sat, under))

    def mu_dead(self) -> float:
        """Dead-oil viscosity at reservoir temperature, cp."""
        return oil.mu_dead_beggs_robinson(self.api, self.temperature)

    def mu_o(self, p):
        """Oil viscosity, cp."""
        p = self._p(p)
        mu_od = self.mu_dead()
        sat = oil.mu_live_beggs_robinson(mu_od, self.rs(np.minimum(p, self.pb)))
        mu_ob = oil.mu_live_beggs_robinson(mu_od, self.rsb)
        under = oil.mu_undersaturated_vasquez_beggs(np.maximum(p, self.pb), self.pb, mu_ob)
        return self._ret(np.where(p < self.pb, sat, under))

    # -- gas and water --------------------------------------------------------

    def z(self, p):
        """Gas z-factor at reservoir temperature."""
        return gas.z_factor(self._p(p), self.temperature, self.gas_gravity,
                            method=self.z_method, y_h2s=self.y_h2s, y_co2=self.y_co2)

    def bg(self, p):
        """Gas FVF, rb/scf."""
        p = self._p(p)
        return gas.gas_fvf(p, self.temperature, self.z(p))

    def mu_g(self, p):
        """Gas viscosity, cp (Lee-Gonzalez-Eakin)."""
        p = self._p(p)
        return gas.gas_viscosity_lee(p, self.temperature, self.z(p), self.gas_gravity)

    def bw(self, p):
        """Water FVF, rb/STB (McCain)."""
        return water.bw_mccain(self._p(p), self.temperature)

    # -- tables ---------------------------------------------------------------

    def table(self, pressures, *, pressure_unit: str = "psia", units_out: str = "field") -> pd.DataFrame:
        """PVT table at the given pressures, sorted high to low.

        ``units_out="si"`` converts each column with :data:`TABLE_UNITS`.
        The bubble point is inserted so the kink in Bo is visible.
        """
        p = np.atleast_1d(units.convert(pressures, pressure_unit, "psia")).astype(float)
        if p.min() < self.pb < p.max():
            p = np.append(p, self.pb)
        p = np.unique(p)[::-1]
        df = pd.DataFrame({
            "p": p,
            "rs": self.rs(p),
            "bo": self.bo(p),
            "bt": self.bt(p),
            "co": self.co(p),
            "mu_o": self.mu_o(p),
            "z": self.z(p),
            "bg": self.bg(p),
            "mu_g": self.mu_g(p),
            "bw": self.bw(p),
        })
        if units_out.lower() == "si":
            for col, (field_unit, si_unit) in TABLE_UNITS.items():
                if field_unit:
                    df[col] = units.convert(df[col].to_numpy(), field_unit, si_unit)
        elif units_out.lower() != "field":
            raise ValueError("units_out must be 'field' or 'si'")
        return df

    def describe(self) -> str:
        """One-line summary of the fluid."""
        return (f"{self.api:g} API oil, gas SG {self.gas_gravity:g}, T={self.temperature:g} degF, "
                f"pb={self.pb:.0f} psia, Rsb={self.rsb:.0f} scf/STB, Bob={self.bob:.4f} rb/STB "
                f"[{self.correlation}, z={self.z_method}]")

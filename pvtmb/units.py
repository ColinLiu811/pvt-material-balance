"""Unit conversions between oilfield and SI units.

Every correlation in this package works internally in oilfield units:

=================  ===========
quantity           unit
=================  ===========
pressure           psia
temperature        degF
oil / water FVF    rb/STB
gas FVF            rb/scf
solution GOR       scf/STB
viscosity          cp
compressibility    1/psi
liquid volume      STB (bbl)
gas volume         scf
=================  ===========

Use :func:`convert` at the boundaries (input parsing, output tables) so that
the physics code never has to care about units.

Standard conditions are taken as 14.696 psia / 60 degF for field units. The
small difference to the SI standard (101.325 kPa / 15 degC) is ignored when
converting GOR and gas FVF, which is the usual industry practice.
"""

from __future__ import annotations

import numpy as np

PSC_PSIA = 14.695948775  # standard pressure, psia
TSC_DEGF = 60.0  # standard temperature, degF
RANKINE_OFFSET = 459.67

BBL_PER_M3 = 6.289810770
FT3_PER_M3 = 35.31466672
FT3_PER_BBL = FT3_PER_M3 / BBL_PER_M3  # 5.614583
PSI_PER_KPA = 0.1450377377

# Multiplicative units: factor converts *from* the unit *to* the base unit of
# its kind (the first entry of each table).
_FACTORS: dict[str, dict[str, float]] = {
    "pressure": {
        "psia": 1.0,
        "kpa": PSI_PER_KPA,
        "mpa": PSI_PER_KPA * 1e3,
        "pa": PSI_PER_KPA * 1e-3,
        "bar": PSI_PER_KPA * 1e2,
        "atm": PSC_PSIA,
    },
    "liquid_volume": {
        "stb": 1.0,
        "bbl": 1.0,
        "rb": 1.0,
        "mstb": 1e3,
        "mmstb": 1e6,
        "m3": BBL_PER_M3,
        "sm3": BBL_PER_M3,
        "ft3": 1.0 / FT3_PER_BBL,
    },
    "gas_volume": {
        "scf": 1.0,
        "mscf": 1e3,
        "mmscf": 1e6,
        "bscf": 1e9,
        "sm3_gas": FT3_PER_M3,
    },
    "gor": {
        "scf/stb": 1.0,
        "mscf/stb": 1e3,
        "sm3/sm3": FT3_PER_BBL,
    },
    "oil_fvf": {
        "rb/stb": 1.0,
        "rm3/sm3": 1.0,
    },
    "gas_fvf": {
        "rb/scf": 1.0,
        "rb/mscf": 1e-3,
        "rcf/scf": 1.0 / FT3_PER_BBL,
        "rm3/sm3_gas": 1.0 / FT3_PER_BBL,
    },
    "viscosity": {
        "cp": 1.0,
        "mpa.s": 1.0,
        "pa.s": 1e3,
    },
    "compressibility": {
        "1/psi": 1.0,
        "1/kpa": 1.0 / PSI_PER_KPA,
        "1/mpa": 1.0 / (PSI_PER_KPA * 1e3),
        "1/bar": 1.0 / (PSI_PER_KPA * 1e2),
    },
}

_TEMPERATURE_UNITS = {"degf", "degc", "k", "degr"}

_ALIASES = {
    "psi": "psia",
    "f": "degf",
    "°f": "degf",
    "c": "degc",
    "°c": "degc",
    "r": "degr",
    "°r": "degr",
    "kelvin": "k",
    "mpas": "mpa.s",
    "pas": "pa.s",
    "scf/bbl": "scf/stb",
    "bbl/stb": "rb/stb",
    "bbl/scf": "rb/scf",
    "ft3/scf": "rcf/scf",
}

_KIND_OF: dict[str, str] = {
    unit: kind for kind, table in _FACTORS.items() for unit in table
}
for _u in _TEMPERATURE_UNITS:
    _KIND_OF[_u] = "temperature"
_KIND_OF["psig"] = "pressure"


class UnitError(ValueError):
    """Raised for unknown units or conversions between incompatible kinds."""


def normalize(unit: str) -> str:
    """Canonical lower-case spelling of a unit string."""
    key = unit.strip().lower().replace(" ", "").replace("·", ".")
    key = _ALIASES.get(key, key)
    if key not in _KIND_OF:
        raise UnitError(f"unknown unit {unit!r}")
    return key


def kind_of(unit: str) -> str:
    """Physical quantity a unit measures, e.g. ``"pressure"``."""
    return _KIND_OF[normalize(unit)]


def _scalar_or_array(x: np.ndarray):
    return float(x) if x.ndim == 0 else x


def _to_degf(x: np.ndarray, unit: str) -> np.ndarray:
    if unit == "degf":
        return x
    if unit == "degc":
        return x * 1.8 + 32.0
    if unit == "k":
        return x * 1.8 - RANKINE_OFFSET
    return x - RANKINE_OFFSET  # degR


def _from_degf(x: np.ndarray, unit: str) -> np.ndarray:
    if unit == "degf":
        return x
    if unit == "degc":
        return (x - 32.0) / 1.8
    if unit == "k":
        return (x + RANKINE_OFFSET) / 1.8
    return x + RANKINE_OFFSET  # degR


def _to_psia(x: np.ndarray, unit: str) -> np.ndarray:
    if unit == "psig":
        return x + PSC_PSIA
    return x * _FACTORS["pressure"][unit]


def _from_psia(x: np.ndarray, unit: str) -> np.ndarray:
    if unit == "psig":
        return x - PSC_PSIA
    return x / _FACTORS["pressure"][unit]


def convert(value, from_unit: str, to_unit: str):
    """Convert ``value`` (scalar or array-like) between compatible units.

    >>> round(convert(20.0, "MPa", "psia"), 2)
    2900.75
    >>> round(convert(200.0, "degF", "degR"), 2)
    659.67
    """
    src, dst = normalize(from_unit), normalize(to_unit)
    kind = _KIND_OF[src]
    if _KIND_OF[dst] != kind:
        raise UnitError(f"cannot convert {kind} ({from_unit}) to {_KIND_OF[dst]} ({to_unit})")

    x = np.asarray(value, dtype=float)
    if kind == "temperature":
        out = _from_degf(_to_degf(x, src), dst)
    elif kind == "pressure":
        out = _from_psia(_to_psia(x, src), dst)
    else:
        table = _FACTORS[kind]
        out = x * (table[src] / table[dst])
    return _scalar_or_array(out)


def degf_to_degr(t_degf):
    """Absolute temperature in degrees Rankine."""
    return _scalar_or_array(np.asarray(t_degf, dtype=float) + RANKINE_OFFSET)


def api_to_sg(api):
    """Stock-tank oil specific gravity (water = 1) from API gravity."""
    return _scalar_or_array(141.5 / (np.asarray(api, dtype=float) + 131.5))


def sg_to_api(sg):
    """API gravity from stock-tank oil specific gravity."""
    return _scalar_or_array(141.5 / np.asarray(sg, dtype=float) - 131.5)

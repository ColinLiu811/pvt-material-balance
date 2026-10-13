import numpy as np
import pytest

from pvtmb import units
from pvtmb.units import UnitError, convert


def test_pressure_conversions():
    assert convert(1.0, "atm", "psia") == pytest.approx(14.696, abs=1e-3)
    assert convert(100.0, "kPa", "psia") == pytest.approx(14.5038, rel=1e-5)
    assert convert(1.0, "bar", "kpa") == pytest.approx(100.0)
    assert convert(0.0, "psig", "psia") == pytest.approx(units.PSC_PSIA)
    assert convert(3000.0, "psia", "psig") == pytest.approx(3000.0 - 14.696, abs=1e-3)


def test_temperature_conversions():
    assert convert(32.0, "degF", "degC") == pytest.approx(0.0)
    assert convert(100.0, "degC", "degF") == pytest.approx(212.0)
    assert convert(0.0, "K", "degR") == pytest.approx(0.0, abs=1e-9)
    assert convert(60.0, "°F", "R") == pytest.approx(519.67)
    assert units.degf_to_degr(200.0) == pytest.approx(659.67)


def test_volume_and_ratio_conversions():
    assert convert(1.0, "m3", "bbl") == pytest.approx(6.28981, rel=1e-6)
    assert convert(1.0, "bbl", "ft3") == pytest.approx(5.614583, rel=1e-6)
    # 1 sm3/sm3 of gas per oil = 5.6146 scf/STB
    assert convert(100.0, "sm3/sm3", "scf/stb") == pytest.approx(561.4583, rel=1e-6)
    assert convert(1.0, "rcf/scf", "rb/scf") == pytest.approx(1 / 5.614583, rel=1e-6)
    assert convert(2.5, "MMSTB", "STB") == pytest.approx(2.5e6)


def test_viscosity_and_compressibility():
    assert convert(1.0, "Pa.s", "cp") == pytest.approx(1000.0)
    assert convert(1.0, "mPa.s", "cp") == pytest.approx(1.0)
    # 10e-6 1/psi is about 1.45e-6 1/kPa
    assert convert(10e-6, "1/psi", "1/kPa") == pytest.approx(1.4504e-6, rel=1e-4)


def test_round_trip_arrays():
    p = np.linspace(500, 6000, 7)
    back = convert(convert(p, "psia", "MPa"), "MPa", "psia")
    np.testing.assert_allclose(back, p)
    t = np.array([60.0, 150.0, 300.0])
    np.testing.assert_allclose(convert(convert(t, "degF", "K"), "K", "degF"), t)


def test_scalar_in_scalar_out():
    assert isinstance(convert(1.0, "bar", "psia"), float)
    assert isinstance(convert([1.0, 2.0], "bar", "psia"), np.ndarray)


def test_api_gravity():
    assert units.api_to_sg(10.0) == pytest.approx(1.0)
    assert units.sg_to_api(units.api_to_sg(35.0)) == pytest.approx(35.0)


def test_incompatible_and_unknown_units():
    with pytest.raises(UnitError):
        convert(1.0, "psia", "degF")
    with pytest.raises(UnitError):
        convert(1.0, "furlongs", "bbl")
    assert units.kind_of("MPa") == "pressure"

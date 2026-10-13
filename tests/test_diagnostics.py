import numpy as np
import pytest

from pvtmb.aquifer import influx_function
from pvtmb.datasets import VIRGINIA_HILLS, VIRGINIA_HILLS_PARAMS
from pvtmb.diagnostics import campbell_trend, drive_indices, identify_drive
from pvtmb.fluid import BlackOil
from pvtmb.material_balance import MaterialBalance
from pvtmb.synthetic import synthetic_history


@pytest.fixture
def fluid():
    return BlackOil(api=32.0, gas_gravity=0.78, temperature=190.0, pb=3000.0)


def test_solution_gas_drive(fluid):
    hist = synthetic_history(fluid, N=80e6, pressures=np.linspace(3000, 1200, 15), swi=0.2)
    mb = MaterialBalance(hist, swi=0.2)
    d = identify_drive(mb)
    assert d.campbell.trend == "flat"
    assert d.mechanism == "solution-gas drive"
    assert d.result.N == pytest.approx(80e6, rel=1e-9)
    assert d.final_indices["DDI"] > 0.8


def test_gas_cap_drive(fluid):
    hist = synthetic_history(fluid, N=40e6, m=1.5, pressures=np.linspace(3000, 2000, 12), swi=0.2)
    d = identify_drive(MaterialBalance(hist, swi=0.2), m=1.5)
    assert d.campbell.trend == "flat"
    assert d.mechanism == "gas-cap drive"


def test_unrecognised_gas_cap_tilts_campbell_plot(fluid):
    hist = synthetic_history(fluid, N=40e6, m=1.0, pressures=np.linspace(3000, 2000, 12), swi=0.2)
    mb = MaterialBalance(hist, swi=0.2)
    assert campbell_trend(mb, m=0.0).trend != "flat"
    assert campbell_trend(mb, m=1.0).trend == "flat"
    d = identify_drive(mb, m=0.0)
    # the gas-cap model fits exactly and must win over the pot aquifer
    assert "gas cap" in d.alternatives and "pot" in d.alternatives
    assert d.result.m == pytest.approx(1.0, rel=1e-6)
    assert d.mechanism.startswith("gas-cap drive") or "gas-cap drive" in d.mechanism
    assert any("hard to tell apart" in n for n in d.notes)


@pytest.mark.parametrize("model,C,trend", [("pot", 3e4, "falling"), ("schilthuis", 60.0, "hump")])
def test_water_drive_detected_and_model_selected(model, C, trend):
    # depletes through the bubble point so a pot aquifer is identifiable
    fluid = BlackOil(api=30.0, gas_gravity=0.8, temperature=180.0, pb=3200.0)
    p = np.linspace(4000.0, 2000.0, 16)
    t = np.linspace(0.0, 3000.0, 16)
    hist = synthetic_history(fluid, N=90e6, pressures=p, t=t, swi=0.22,
                             we=C * influx_function(model, p, t))
    d = identify_drive(MaterialBalance(hist, swi=0.22))
    assert d.campbell.trend == trend
    assert d.result.aquifer == model
    assert d.result.N == pytest.approx(90e6, rel=1e-6)
    assert d.mechanism == "water drive" or d.mechanism.startswith("combination drive (water drive")
    assert {"volumetric", "pot", "schilthuis"} <= set(d.alternatives)
    assert "water drive" in d.report()


def test_indices_sum_to_one_for_exact_fit(fluid):
    p = np.linspace(3000, 1800, 12)
    t = np.linspace(0, 2000, 12)
    hist = synthetic_history(fluid, N=50e6, m=0.3, pressures=p, t=t, swi=0.2,
                             we=20.0 * influx_function("schilthuis", p, t))
    mb = MaterialBalance(hist, swi=0.2)
    res = mb.fit(m=0.3, aquifer="schilthuis")
    idx = drive_indices(mb, res)
    np.testing.assert_allclose(idx["total"], 1.0, rtol=1e-8)
    assert (idx[["DDI", "SDI", "WDI", "EDI"]] >= 0).all().all()


def test_virginia_hills_diagnosis():
    """Undersaturated and treated as volumetric in the book. Oil expansion
    dominates, with rock/water expansion a large secondary contributor; the
    Campbell plot rises but an aquifer does not explain it."""
    mb = MaterialBalance(VIRGINIA_HILLS, **VIRGINIA_HILLS_PARAMS)
    d = identify_drive(mb)
    assert d.mechanism == "undersaturated oil expansion"
    assert d.result.aquifer is None
    idx = d.final_indices
    assert 0.3 < idx["EDI"] < 0.5
    assert idx["SDI"] == 0.0


def test_pot_aquifer_above_pb_is_not_identifiable():
    """W = pi - p and Et are both ~ proportional to the pressure drop above pb,
    so a pot aquifer just looks like extra oil in place."""
    fluid = BlackOil(api=30.0, gas_gravity=0.8, temperature=180.0, pb=1800.0)
    p = np.linspace(4000.0, 2600.0, 16)
    hist = synthetic_history(fluid, N=90e6, pressures=p, swi=0.22,
                             we=6e4 * influx_function("pot", p))
    mb = MaterialBalance(hist, swi=0.22)
    assert campbell_trend(mb).relative_change < 0.12  # barely any curvature
    assert mb.fit().N > 1.5 * 90e6  # apparent OOIP absorbs the aquifer


def test_forced_volumetric(fluid):
    hist = synthetic_history(fluid, N=80e6, pressures=np.linspace(3000, 1200, 10), swi=0.2)
    d = identify_drive(MaterialBalance(hist, swi=0.2), aquifer=None)
    assert d.alternatives == {}
    assert d.result.aquifer is None

import numpy as np
import pytest

from pvtmb.correlations import oil, water
from pvtmb.fluid import BlackOil
from pvtmb.units import convert


@pytest.fixture
def fluid():
    return BlackOil(api=35.0, gas_gravity=0.75, temperature=200.0, rsb=600.0)


def test_rsb_and_pb_are_interchangeable(fluid):
    other = BlackOil(api=35.0, gas_gravity=0.75, temperature=200.0, pb=fluid.pb)
    assert other.rsb == pytest.approx(600.0)
    assert other.bob == pytest.approx(fluid.bob)


def test_requires_exactly_one_of_rsb_pb():
    with pytest.raises(ValueError):
        BlackOil(api=35.0, gas_gravity=0.75, temperature=200.0)
    with pytest.raises(ValueError):
        BlackOil(api=35.0, gas_gravity=0.75, temperature=200.0, rsb=600.0, pb=2500.0)
    with pytest.raises(ValueError):
        BlackOil(api=35.0, gas_gravity=0.75, temperature=200.0, rsb=600.0, correlation="glaso")


def test_rs_flat_above_pb(fluid):
    p = np.array([fluid.pb, 3500.0, 5000.0])
    np.testing.assert_allclose(fluid.rs(p), 600.0)
    assert fluid.rs(1000.0) < 600.0


def test_bo_peaks_at_bubble_point(fluid):
    p = np.linspace(300.0, 6000.0, 200)
    bo = fluid.bo(p)
    assert p[np.argmax(bo)] == pytest.approx(fluid.pb, abs=30.0)
    assert fluid.bo(fluid.pb) == pytest.approx(fluid.bob)
    # continuous across pb
    assert fluid.bo(fluid.pb - 1e-6) == pytest.approx(fluid.bo(fluid.pb + 1e-6), rel=1e-6)


def test_bt_equals_bo_above_pb_and_grows_below(fluid):
    assert fluid.bt(4000.0) == pytest.approx(fluid.bo(4000.0))
    p = np.linspace(500.0, fluid.pb, 20)
    assert np.all(np.diff(fluid.bt(p)) < 0)  # Bt falls as pressure rises toward pb


def test_viscosity_minimum_at_bubble_point(fluid):
    p = np.linspace(300.0, 6000.0, 300)
    mu = fluid.mu_o(p)
    assert p[np.argmin(mu)] == pytest.approx(fluid.pb, abs=30.0)
    assert fluid.mu_o(fluid.pb) == pytest.approx(
        oil.mu_live_beggs_robinson(fluid.mu_dead(), fluid.rsb))


def test_compressibility_jumps_below_pb(fluid):
    # liberated gas makes the apparent compressibility an order of magnitude larger
    assert fluid.co(fluid.pb - 50.0) > 5.0 * fluid.co(fluid.pb + 50.0)
    assert fluid.co(4000.0) == pytest.approx(
        oil.co_vasquez_beggs(4000.0, 600.0, 35.0, fluid.gas_gravity_sep, 200.0))


def test_vasquez_beggs_option(fluid):
    vb = BlackOil(api=35.0, gas_gravity=0.75, temperature=200.0, rsb=600.0,
                  correlation="vasquez_beggs", z_method="hall_yarborough")
    assert vb.pb != pytest.approx(fluid.pb, rel=1e-3)
    assert vb.bob == pytest.approx(oil.bo_vasquez_beggs(600.0, 35.0, 0.75, 200.0))
    assert vb.z(3000.0) == pytest.approx(fluid.z(3000.0), rel=0.01)


def test_table_inserts_bubble_point_and_sorts(fluid):
    df = fluid.table([1000, 2000, 3000, 4000])
    assert len(df) == 5
    assert df.p.is_monotonic_decreasing
    assert (np.isclose(df.p, fluid.pb)).any()
    assert list(df.columns) == ["p", "rs", "bo", "bt", "co", "mu_o", "z", "bg", "mu_g", "bw"]


def test_table_in_si_units(fluid):
    field = fluid.table([1000.0, 4000.0])
    si = fluid.table(convert([1000.0, 4000.0], "psia", "MPa"), pressure_unit="MPa", units_out="si")
    np.testing.assert_allclose(si.p, convert(field.p.to_numpy(), "psia", "MPa"))
    np.testing.assert_allclose(si.rs, field.rs / 5.614583, rtol=1e-6)
    np.testing.assert_allclose(si.bo, field.bo)
    np.testing.assert_allclose(si.mu_o, field.mu_o)
    np.testing.assert_allclose(si.bg, field.bg * 5.614583, rtol=1e-6)


def test_from_units_matches_field_inputs():
    si = BlackOil.from_units(api=30.0, gas_gravity=0.8, temperature=80.0, temperature_unit="degC",
                             pb=20.0, pressure_unit="MPa")
    fld = BlackOil(api=30.0, gas_gravity=0.8, temperature=176.0, pb=convert(20.0, "MPa", "psia"))
    assert si.rsb == pytest.approx(fld.rsb)


def test_rejects_nonpositive_pressure(fluid):
    with pytest.raises(ValueError):
        fluid.bo([1000.0, 0.0])


def test_water_fvf():
    # McCain: Bw a little above 1 at reservoir conditions, decreasing with pressure
    bw = water.bw_mccain(np.array([1000.0, 5000.0]), 200.0)
    assert 1.01 < bw[1] < bw[0] < 1.05

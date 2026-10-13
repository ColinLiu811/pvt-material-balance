import numpy as np
import pytest

from pvtmb.correlations import oil
from pvtmb.datasets import AHMED_CRUDES


def _are(pred, meas):
    """Absolute relative error in percent."""
    return np.abs(np.asarray(pred) - meas) / meas * 100.0


# ---------------------------------------------------------------------------
# Standing
# ---------------------------------------------------------------------------

def test_standing_pb_inverts_rs():
    p = np.linspace(200.0, 5000.0, 20)
    rs = oil.rs_standing(p, 35.0, 0.75, 180.0)
    np.testing.assert_allclose(oil.pb_standing(rs, 35.0, 0.75, 180.0), p, rtol=1e-10)


def test_standing_rs_increases_with_pressure_and_api():
    p = np.linspace(500.0, 4000.0, 8)
    rs = oil.rs_standing(p, 30.0, 0.8, 200.0)
    assert np.all(np.diff(rs) > 0)
    assert oil.rs_standing(2000.0, 40.0, 0.8, 200.0) > oil.rs_standing(2000.0, 30.0, 0.8, 200.0)


def test_standing_against_ahmed_crudes():
    d = AHMED_CRUDES
    bo = oil.bo_standing(d.rsb, d.api, d.gas_gravity, d.temp)
    rs = oil.rs_standing(d.pb, d.api, d.gas_gravity, d.temp)
    pb = oil.pb_standing(d.rsb, d.api, d.gas_gravity, d.temp)

    # Bo is Standing's strongest correlation: every sample within ~5.5%
    assert _are(bo, d.bob).mean() < 2.5
    assert _are(bo, d.bob).max() < 6.0
    # Rs / pb carry more scatter; oil 6 (27 API, 4239 psia) is the outlier
    assert _are(rs, d.rsb).mean() < 12.0
    assert _are(pb, d.pb).mean() < 12.0
    # spot-check the first sample against a hand calculation
    assert bo[0] == pytest.approx(1.506, abs=1e-3)
    assert rs[0] == pytest.approx(832.0, abs=1.0)


# ---------------------------------------------------------------------------
# Vasquez-Beggs
# ---------------------------------------------------------------------------

def test_vb_pb_matches_published_form():
    """Our algebraic inverse equals the published pb correlation."""
    rs, gs, t = 600.0, 0.8, 190.0
    for api, (c1, c2, c3) in [(25.0, (27.624, 0.914328, 11.172)),
                              (38.0, (56.18, 0.84246, 10.393))]:
        published = (c1 * rs / gs * 10.0 ** (-c3 * api / (t + 460.0))) ** c2
        assert oil.pb_vasquez_beggs(rs, api, gs, t) == pytest.approx(published, rel=2e-4)


def test_vb_coefficients_switch_at_30_api():
    rs = oil.rs_vasquez_beggs(2000.0, np.array([29.9, 30.0, 30.1]), 0.8, 180.0)
    c1, c2, c3 = 0.0362, 1.0937, 25.7240
    assert rs[1] == pytest.approx(c1 * 0.8 * 2000**c2 * np.exp(c3 * 30 / 640))
    assert rs[2] != pytest.approx(rs[1], rel=1e-3)


def test_separator_correction():
    assert oil.separator_gas_gravity(0.75, 35.0) == pytest.approx(0.75)
    # higher separator pressure -> heavier corrected gas gravity
    assert oil.separator_gas_gravity(0.75, 35.0, 300.0, 80.0) > 0.75


def test_vb_against_ahmed_crudes():
    d = AHMED_CRUDES
    rs = oil.rs_vasquez_beggs(d.pb, d.api, d.gas_gravity, d.temp)
    bo = oil.bo_vasquez_beggs(d.rsb, d.api, d.gas_gravity, d.temp)
    assert _are(rs, d.rsb).mean() < 6.0
    assert _are(bo, d.bob).mean() < 3.0
    assert _are(bo, d.bob).max() < 6.0


def test_undersaturated_bo_shrinks_with_pressure():
    pb, bob, rsb = 2500.0, 1.30, 600.0
    p = np.linspace(pb, 6000.0, 30)
    bo = oil.bo_undersaturated(p, pb, bob, rsb, 35.0, 0.75, 200.0)
    assert bo[0] == pytest.approx(bob)
    assert np.all(np.diff(bo) < 0)


def test_undersaturated_bo_consistent_with_compressibility():
    """-(1/Bo) dBo/dp should equal the VB co at every pressure."""
    args = (2500.0, 1.30, 600.0, 35.0, 0.75, 200.0)
    p, h = 4000.0, 0.5
    bo = lambda x: oil.bo_undersaturated(x, *args)  # noqa: E731
    co_numeric = -(bo(p + h) - bo(p - h)) / (2 * h) / bo(p)
    co = oil.co_vasquez_beggs(p, 600.0, 35.0, 0.75, 200.0)
    assert co_numeric == pytest.approx(co, rel=1e-6)
    assert 5e-6 < co < 3e-5  # typical black-oil range


# ---------------------------------------------------------------------------
# Viscosity
# ---------------------------------------------------------------------------

def test_dead_oil_viscosity_hand_calculation():
    # x = 200^-1.163 * exp(6.9824 - 0.04658*35) = 0.44518 -> 10^x - 1
    x = 200.0**-1.163 * np.exp(6.9824 - 0.04658 * 35.0)
    assert oil.mu_dead_beggs_robinson(35.0, 200.0) == pytest.approx(10**x - 1)
    assert oil.mu_dead_beggs_robinson(35.0, 200.0) == pytest.approx(1.787, abs=2e-3)


def test_dead_oil_viscosity_trends():
    t = np.linspace(100, 300, 9)
    assert np.all(np.diff(oil.mu_dead_beggs_robinson(30.0, t)) < 0)
    api = np.linspace(15, 50, 8)
    assert np.all(np.diff(oil.mu_dead_beggs_robinson(api, 180.0)) < 0)


def test_live_oil_reduces_to_dead_oil_at_zero_rs():
    # A(0) = 10.715 * 100^-0.515 ~ 1 and B(0) = 5.44 * 150^-0.338 ~ 1
    mu_od = np.array([0.8, 2.0, 15.0])
    np.testing.assert_allclose(oil.mu_live_beggs_robinson(mu_od, 0.0), mu_od, rtol=2e-3)


def test_dissolved_gas_thins_the_oil():
    mu_od = oil.mu_dead_beggs_robinson(30.0, 180.0)
    rs = np.array([0.0, 200.0, 500.0, 1000.0])
    mu = oil.mu_live_beggs_robinson(mu_od, rs)
    assert np.all(np.diff(mu) < 0)
    assert mu[-1] < 0.5 * mu_od


def test_undersaturated_viscosity_increases_above_pb():
    p = np.linspace(2500.0, 6000.0, 10)
    mu = oil.mu_undersaturated_vasquez_beggs(p, 2500.0, 0.6)
    assert mu[0] == pytest.approx(0.6)
    assert np.all(np.diff(mu) > 0)
    # pressure effect is modest: tens of percent, not multiples
    assert mu[-1] < 1.5 * mu[0]

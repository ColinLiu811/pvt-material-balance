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

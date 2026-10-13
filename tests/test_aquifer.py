import numpy as np
import pytest

from pvtmb import aquifer


def test_pot_influx_is_pressure_drop():
    np.testing.assert_allclose(aquifer.pot_influx_function([3000, 2950, 2800]), [0, 50, 200])


def test_schilthuis_integral():
    # constant drawdown of 100 psi after t=0 step, linear ramp in between
    p = np.array([3000.0, 2900.0, 2900.0])
    t = np.array([0.0, 10.0, 30.0])
    # 0.5*100*10 + 100*20
    np.testing.assert_allclose(aquifer.schilthuis_influx_function(p, t), [0, 500, 2500])


def test_schilthuis_rejects_bad_time():
    with pytest.raises(ValueError):
        aquifer.schilthuis_influx_function([3000, 2900], [0, 0])


def test_dispatch_and_pot_volume():
    with pytest.raises(ValueError):
        aquifer.influx_function("carter_tracy", [3000, 2900])
    # C = (cw + cf) Wi f
    assert aquifer.pot_aquifer_volume(70.0, 3e-6, 4e-6, f=0.5) == pytest.approx(2e7)

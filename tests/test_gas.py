import math

import numpy as np
import pytest

from pvtmb.correlations import gas


def test_sutton_pseudo_criticals():
    tpc, ppc = gas.pseudo_critical_sutton(0.7)
    assert tpc == pytest.approx(169.2 + 349.5 * 0.7 - 74.0 * 0.49)
    assert ppc == pytest.approx(756.8 - 131.0 * 0.7 - 3.6 * 0.49)


def test_wichert_aziz_no_acid_gas_is_identity():
    assert gas.wichert_aziz(380.0, 665.0) == pytest.approx((380.0, 665.0))
    tpc, ppc = gas.wichert_aziz(380.0, 665.0, y_h2s=0.1, y_co2=0.05)
    assert tpc < 380.0 and ppc < 665.0


@pytest.mark.parametrize("solver", [gas.z_hall_yarborough, gas.z_dranchuk_abou_kassem])
def test_ideal_gas_limit(solver):
    assert solver(1.5, 0.0) == 1.0
    assert solver(1.5, 1e-4) == pytest.approx(1.0, abs=1e-4)


def test_hall_yarborough_satisfies_its_eos():
    tpr, ppr = 1.4, 3.7
    z = gas.z_hall_yarborough(tpr, ppr)
    t = 1 / tpr
    a = 0.06125 * t * math.exp(-1.2 * (1 - t) ** 2)
    b = t * (14.76 - 9.76 * t + 4.58 * t**2)
    c = t * (90.7 - 242.2 * t + 42.4 * t**2)
    d = 2.18 + 2.82 * t
    y = a * ppr / z
    resid = -a * ppr + (y + y**2 + y**3 - y**4) / (1 - y) ** 3 - b * y**2 + c * y**d
    assert resid == pytest.approx(0.0, abs=1e-10)


def test_dak_satisfies_its_eos():
    tpr, ppr = 1.4, 3.7
    z = gas.z_dranchuk_abou_kassem(tpr, ppr)
    a = gas._DAK
    rho = 0.27 * ppr / (z * tpr)
    rhs = (1 + (a[0] + a[1] / tpr + a[2] / tpr**3 + a[3] / tpr**4 + a[4] / tpr**5) * rho
           + (a[5] + a[6] / tpr + a[7] / tpr**2) * rho**2
           - a[8] * (a[6] / tpr + a[7] / tpr**2) * rho**5
           + a[9] * (1 + a[10] * rho**2) * rho**2 / tpr**3 * math.exp(-a[10] * rho**2))
    assert z == pytest.approx(rhs, abs=1e-10)


def test_methods_agree_over_standing_katz_range():
    """HY and DAK are independent fits of the Standing-Katz chart.

    Away from the critical region (Tpr < 1.1) both reproduce the chart to
    well under 1% on average, so they should agree with each other closely.
    """
    tpr, ppr = np.meshgrid(np.linspace(1.15, 3.0, 12), np.linspace(0.2, 15.0, 30))
    z_hy = gas.z_hall_yarborough(tpr, ppr)
    z_dak = gas.z_dranchuk_abou_kassem(tpr, ppr)
    rel = np.abs(z_hy - z_dak) / z_dak
    assert rel.mean() < 0.005
    # worst case (~3%) sits at the low-Tpr edge near ppr 2, where the chart
    # bends sharply and both fits lose accuracy
    assert rel.max() < 0.035
    assert rel[:, tpr[0] >= 1.3].max() < 0.015


def test_z_shape_along_isotherm():
    """Classic Standing-Katz shape: dips below 1, minimum near ppr 2-3 for
    Tpr = 1.5, then rises above 1 at high pressure."""
    ppr = np.linspace(0.2, 12, 60)
    z = gas.z_dranchuk_abou_kassem(1.5, ppr)
    i_min = int(np.argmin(z))
    assert 2.0 < ppr[i_min] < 4.0
    assert 0.74 < z[i_min] < 0.80
    assert z[-1] > 1.0


def test_z_factor_vectorized_and_method_names():
    p = np.array([500.0, 2000.0, 5000.0])
    z1 = gas.z_factor(p, 200.0, 0.7, method="dak")
    z2 = gas.z_factor(p, 200.0, 0.7, method="Hall_Yarborough")
    assert z1.shape == (3,)
    np.testing.assert_allclose(z1, z2, rtol=0.01)
    with pytest.raises(ValueError):
        gas.z_factor(1000.0, 200.0, 0.7, method="papay")


def test_gas_fvf_matches_textbook_constant():
    # Bg = 0.005035 z T / p  rb/scf  (e.g. Ahmed, Reservoir Engineering Handbook)
    bg = gas.gas_fvf(2000.0, 200.0, 0.87)
    assert bg == pytest.approx(0.005035 * 0.87 * 659.67 / 2000.0, rel=1e-3)
    # ideal gas at standard conditions occupies 1 scf
    assert gas.gas_fvf(14.695948775, 60.0, 1.0) * 5.614583 == pytest.approx(1.0)


def test_gas_density_and_viscosity_reasonable():
    z = gas.z_factor(2000.0, 200.0, 0.72)
    rho = gas.gas_density(2000.0, 200.0, z, 0.72)
    # real-gas law check: rho = p M / (z R T)
    assert rho == pytest.approx(2000 * 28.9625 * 0.72 / (z * 10.7316 * 659.67))
    mu = gas.gas_viscosity_lee(2000.0, 200.0, z, 0.72)
    assert 0.012 < mu < 0.025
    # viscosity increases with pressure at constant temperature
    z_hi = gas.z_factor(5000.0, 200.0, 0.72)
    assert gas.gas_viscosity_lee(5000.0, 200.0, z_hi, 0.72) > mu

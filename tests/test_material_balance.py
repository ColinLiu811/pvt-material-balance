import numpy as np
import pandas as pd
import pytest

from pvtmb.datasets import VIRGINIA_HILLS, VIRGINIA_HILLS_PARAMS
from pvtmb.fluid import BlackOil
from pvtmb.material_balance import MaterialBalance, prepare_history
from pvtmb.numerics import fit_line
from pvtmb.synthetic import synthetic_history


# ---------------------------------------------------------------------------
# Textbook: Virginia Hills Beaverhill Lake (Ahmed, Reservoir Eng. Handbook)
# ---------------------------------------------------------------------------

# Solution table from the book: F (Mbbl), Eo, Eo + Ew,f at each pressure
BOOK_F = [26.84, 45.54, 102.95, 133.47, 282.74, 478.23, 712.66, 1105.65,
          1674.72, 2229.84, 2805.73, 3399.71]
BOOK_EO = [0.0002, 0.0002, 0.0003, 0.0003, 0.0007, 0.0014, 0.0020, 0.0026,
           0.0028, 0.0048, 0.0058, 0.0068]
BOOK_ET = [0.00025, 0.00029, 0.00048, 0.00051, 0.00115, 0.00220, 0.00318,
           0.00430, 0.00517, 0.00805, 0.00990, 0.01170]


@pytest.fixture
def virginia():
    return MaterialBalance(VIRGINIA_HILLS, **VIRGINIA_HILLS_PARAMS)


def test_virginia_hills_terms_match_book(virginia):
    t = virginia.terms.iloc[1:]
    np.testing.assert_allclose(t.F / 1e3, BOOK_F, atol=0.01)
    np.testing.assert_allclose(t.Eo, BOOK_EO, atol=5e-5)
    # the book rounds Ew,f to 1e-5 per psi; ours is 1.003e-5 per psi
    np.testing.assert_allclose(t.Eo + t.Ef, BOOK_ET, rtol=0.01)


def test_virginia_hills_ooip(virginia):
    res = virginia.fit(m=0)
    assert res.method.startswith("F vs Et (undersaturated)")
    # The book draws the line by eye and reads N ~ 257 MMSTB. A least-squares
    # line through the origin of the book's own tabulated points gives 284
    # MMSTB; we must agree with that, and land within ~10% of both the
    # hand-drawn answer and the 270.6 MMSTB volumetric estimate.
    book_ls = fit_line(BOOK_ET, np.array(BOOK_F) * 1e3, through_origin=True).slope
    assert res.N == pytest.approx(book_ls, rel=0.01)
    assert res.N == pytest.approx(257e6, rel=0.11)
    assert res.N == pytest.approx(270.6e6, rel=0.06)
    assert res.fit.r2 > 0.99


def test_gas_cap_fit_needs_bg(virginia):
    with pytest.raises(ValueError, match="Bg"):
        virginia.fit(m="estimate")


# ---------------------------------------------------------------------------
# Synthetic histories with known answers
# ---------------------------------------------------------------------------

@pytest.fixture
def saturated_fluid():
    return BlackOil(api=32.0, gas_gravity=0.78, temperature=190.0, pb=3000.0)


def test_recovers_N_undersaturated_with_fluid_model():
    fluid = BlackOil(api=32.0, gas_gravity=0.78, temperature=190.0, pb=2000.0)
    p = np.linspace(4500.0, 2600.0, 12)
    hist = synthetic_history(fluid, N=150e6, pressures=p, swi=0.25, cw=3e-6, cf=5e-6)
    mb = MaterialBalance(hist[["p", "np", "gp"]], swi=0.25, cw=3e-6, cf=5e-6, fluid=fluid)
    assert mb.above_bubble_point
    res = mb.fit(m=0)
    assert res.N == pytest.approx(150e6, rel=1e-9)
    assert res.fit.r2 == pytest.approx(1.0)


def test_recovers_N_solution_gas_drive(saturated_fluid):
    p = np.linspace(3000.0, 1200.0, 15)
    hist = synthetic_history(saturated_fluid, N=80e6, pressures=p)
    mb = MaterialBalance(hist, swi=0.2)
    res = mb.fit(m=0)
    assert "solution gas" in res.method
    assert res.N == pytest.approx(80e6, rel=1e-9)


def test_recovers_N_with_known_gas_cap(saturated_fluid):
    p = np.linspace(3000.0, 1800.0, 12)
    hist = synthetic_history(saturated_fluid, N=60e6, m=0.4, pressures=p)
    res = MaterialBalance(hist, swi=0.2).fit(m=0.4)
    assert res.N == pytest.approx(60e6, rel=1e-9)


def test_estimates_gas_cap_size(saturated_fluid):
    p = np.linspace(3000.0, 1800.0, 12)
    hist = synthetic_history(saturated_fluid, N=60e6, m=0.4, pressures=p)
    res = MaterialBalance(hist, swi=0.2).fit(m="estimate")
    assert res.N == pytest.approx(60e6, rel=1e-6)
    assert res.m == pytest.approx(0.4, rel=1e-6)
    assert res.fit.r2 == pytest.approx(1.0)


def test_noisy_gas_cap_estimate_and_uncertainty(saturated_fluid):
    rng = np.random.default_rng(42)
    p = np.linspace(3000.0, 1500.0, 20)
    hist = synthetic_history(saturated_fluid, N=60e6, m=0.4, pressures=p)
    hist.loc[1:, "np"] *= 1.0 + rng.normal(0.0, 0.01, len(hist) - 1)
    hist["gp"] = hist["np"] * (hist["gp"] / hist["np"]).fillna(0.0)
    res = MaterialBalance(hist, swi=0.2).fit(m="estimate")
    assert res.N == pytest.approx(60e6, rel=0.1)
    assert abs(res.N - 60e6) < 3 * res.N_stderr + 1e6
    assert np.isfinite(res.m_stderr)


def test_ignoring_gas_cap_overstates_N(saturated_fluid):
    """Fitting a gas-cap reservoir as if m = 0 gives a too-large N: the
    classic Havlena-Odeh warning sign."""
    p = np.linspace(3000.0, 1800.0, 12)
    hist = synthetic_history(saturated_fluid, N=60e6, m=0.4, pressures=p)
    res = MaterialBalance(hist, swi=0.2).fit(m=0)
    assert res.N > 70e6


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def test_history_validation():
    base = pd.DataFrame({"p": [3000.0, 2900.0, 2800.0], "np": [0.0, 1e5, 2e5], "bo": 1.3})
    prepare_history(base)  # undersaturated, no gas data needed
    with pytest.raises(ValueError, match="initial"):
        prepare_history(base.assign(np=[1.0, 1e5, 2e5]))
    with pytest.raises(ValueError, match="non-decreasing"):
        prepare_history(base.assign(np=[0.0, 2e5, 1e5]))
    with pytest.raises(ValueError, match="required"):
        prepare_history(base.drop(columns="np"))
    with pytest.raises(ValueError, match="'bo'"):
        prepare_history(base.drop(columns="bo"))


def test_missing_gp_below_pb_is_an_error(saturated_fluid):
    hist = synthetic_history(saturated_fluid, N=60e6, pressures=np.linspace(3000, 2000, 5))
    with pytest.raises(ValueError, match="gp"):
        prepare_history(hist.drop(columns="gp"))


def test_column_names_are_case_insensitive():
    df = VIRGINIA_HILLS.rename(columns=str.upper)
    res = MaterialBalance(df, **VIRGINIA_HILLS_PARAMS).fit()
    assert res.N == pytest.approx(MaterialBalance(VIRGINIA_HILLS, **VIRGINIA_HILLS_PARAMS).fit().N)

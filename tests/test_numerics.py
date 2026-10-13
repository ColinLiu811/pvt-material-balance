import math

import numpy as np
import pytest

from pvtmb.numerics import (
    ConvergenceError,
    cumulative_trapezoid,
    fit_line,
    lstsq,
    newton_bracketed,
)


def test_newton_finds_sqrt2():
    root = newton_bracketed(lambda x: x * x - 2, lambda x: 2 * x, 1.0, 0.0, 2.0)
    assert root == pytest.approx(math.sqrt(2), abs=1e-12)


def test_newton_falls_back_to_bisection_on_bad_derivative():
    # f'(x) is wrong on purpose (sign flipped) - the bracket keeps it honest
    f = lambda x: x**3 - x - 2  # noqa: E731
    root = newton_bracketed(f, lambda x: -(3 * x * x - 1), 1.0, 1.0, 2.0, maxiter=200)
    assert f(root) == pytest.approx(0.0, abs=1e-10)


def test_newton_flat_derivative_and_bad_start():
    # derivative zero at the starting point, start outside bracket
    f = lambda x: math.cos(x)  # noqa: E731
    root = newton_bracketed(f, lambda x: -math.sin(x), 10.0, 1.0, 2.0)
    assert root == pytest.approx(math.pi / 2, abs=1e-12)


def test_newton_requires_bracket():
    with pytest.raises(ValueError):
        newton_bracketed(lambda x: x * x + 1, lambda x: 2 * x, 0.0, -1.0, 1.0)


def test_newton_reports_non_convergence():
    with pytest.raises(ConvergenceError):
        newton_bracketed(lambda x: x - 0.3, lambda x: 1e-30, 0.5, 0.0, 1.0, maxiter=3, xtol=0)


def test_fit_line_exact():
    x = np.arange(10.0)
    fit = fit_line(x, 3.0 + 2.0 * x)
    assert fit.intercept == pytest.approx(3.0)
    assert fit.slope == pytest.approx(2.0)
    assert fit.r2 == pytest.approx(1.0)


def test_fit_through_origin_matches_closed_form():
    rng = np.random.default_rng(0)
    x = np.linspace(1, 10, 25)
    y = 4.0 * x + rng.normal(0, 0.5, x.size)
    fit = fit_line(x, y, through_origin=True)
    assert fit.slope == pytest.approx((x @ y) / (x @ x))
    # standard error: sqrt(s^2 / sum x^2)
    s2 = (fit.residuals @ fit.residuals) / (x.size - 1)
    assert fit.stderr[0] == pytest.approx(math.sqrt(s2 / (x @ x)))


def test_multilinear_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (40, 3))
    beta = np.array([2.0, -1.0, 0.5])
    fit = lstsq(X, X @ beta)
    np.testing.assert_allclose(fit.coef, beta, atol=1e-12)


def test_cumulative_trapezoid():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(cumulative_trapezoid(2 * x, x), x**2, atol=1e-12)

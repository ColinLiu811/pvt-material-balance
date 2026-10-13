"""Small numerical toolkit: root finding and linear least squares.

Kept dependency-free (numpy only) so the iteration behaviour is explicit and
testable rather than hidden inside a library call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class ConvergenceError(RuntimeError):
    """Raised when an iterative solver fails to converge."""


def newton_bracketed(
    f: Callable[[float], float],
    dfdx: Callable[[float], float],
    x0: float,
    lo: float,
    hi: float,
    *,
    xtol: float = 1e-12,
    ftol: float = 1e-13,
    maxiter: int = 100,
) -> float:
    """Newton-Raphson safeguarded by bisection.

    ``f(lo)`` and ``f(hi)`` must have opposite signs. Each iteration takes the
    Newton step when it lands strictly inside the current bracket and falls
    back to bisection otherwise, so convergence is guaranteed while keeping
    Newton's quadratic rate near the root.
    """
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if np.sign(flo) == np.sign(fhi):
        raise ValueError(f"root is not bracketed: f({lo})={flo:.4g}, f({hi})={fhi:.4g}")

    x = x0 if lo < x0 < hi else 0.5 * (lo + hi)
    for _ in range(maxiter):
        fx = f(x)
        if abs(fx) <= ftol:
            return x
        # shrink the bracket around the root
        if np.sign(fx) == np.sign(flo):
            lo, flo = x, fx
        else:
            hi = x

        d = dfdx(x)
        x_new = x - fx / d if d != 0.0 and np.isfinite(d) else np.nan
        if not (lo < x_new < hi):
            x_new = 0.5 * (lo + hi)
        if abs(x_new - x) <= xtol * (1.0 + abs(x)):
            return x_new
        x = x_new
    raise ConvergenceError(f"no convergence after {maxiter} iterations (x={x:.6g})")


@dataclass(frozen=True)
class LinearFit:
    """Result of a linear least-squares fit ``y = X @ coef``."""

    coef: np.ndarray
    stderr: np.ndarray
    r2: float
    residuals: np.ndarray
    n: int

    @property
    def intercept(self) -> float:
        """Intercept of a straight-line fit (first coefficient)."""
        return float(self.coef[0])

    @property
    def slope(self) -> float:
        """Slope of a straight-line fit (second coefficient)."""
        return float(self.coef[-1])


def lstsq(X, y, *, centered_r2: bool = True) -> LinearFit:
    """Ordinary least squares for ``y = X @ coef`` (no implicit intercept).

    Standard errors come from ``s^2 (X^T X)^-1`` with ``s^2 = SSE / (n - p)``.
    ``centered_r2=False`` gives the uncentred R^2 that is appropriate for
    regressions forced through the origin.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if n < p:
        raise ValueError(f"need at least {p} points, got {n}")

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum()) if centered_r2 else float(y @ y)
    r2 = 1.0 - sse / sst if sst > 0 else 1.0

    dof = n - p
    if dof > 0:
        cov = (sse / dof) * np.linalg.pinv(X.T @ X)
        stderr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        stderr = np.full(p, np.nan)
    return LinearFit(coef=coef, stderr=stderr, r2=r2, residuals=resid, n=n)


def fit_line(x, y, *, through_origin: bool = False) -> LinearFit:
    """Straight-line fit. Coefficients are ``[intercept, slope]`` or ``[slope]``."""
    x = np.asarray(x, dtype=float)
    if through_origin:
        return lstsq(x[:, None], y, centered_r2=False)
    return lstsq(np.column_stack([np.ones_like(x), x]), y)


def cumulative_trapezoid(y, x) -> np.ndarray:
    """Running trapezoidal integral of ``y`` over ``x``, starting at 0."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out

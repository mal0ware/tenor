"""The Nelson-Siegel yield curve in the Diebold-Li factorisation.

Diebold and Li (2006), section 2.2 (the paper does not number its equations):

    y_t(tau) = b1_t + b2_t * (1 - exp(-lam*tau)) / (lam*tau)
                    + b3_t * ((1 - exp(-lam*tau)) / (lam*tau) - exp(-lam*tau))

with maturity ``tau`` in months. The loadings are 1 (level), a function that
starts at 1 and decays to 0 (slope), and a hump that starts at 0, rises and
decays to 0 (curvature). With ``lam`` fixed the betas come from ordinary least
squares on each date (section 3.2).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: The decay value Diebold and Li fix for every date (footnote 5, section 3.2).
DL_LAMBDA: float = 0.0609


def _hump_argmax() -> float:
    """Return x* = lam * tau at which the curvature loading peaks.

    The loading g(x) = (1 - e^-x)/x - e^-x has g'(x) = 0 where
    e^-x (x^2 + x + 1) = 1. Bisection on [1, 3] finds the single positive root.
    """
    lo, hi = 1.0, 3.0

    def h(x: float) -> float:
        return float(np.exp(-x) * (x * x + x + 1.0) - 1.0)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if h(lo) * h(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


#: lam * tau at the peak of the curvature loading, about 1.7933.
HUMP_ARGMAX: float = _hump_argmax()


def curvature_peak(lam: float) -> float:
    """Maturity at which the curvature loading is largest, for a given decay."""
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    return HUMP_ARGMAX / lam


def lambda_for_peak(tau: float) -> float:
    """Decay that puts the peak of the curvature loading at maturity ``tau``."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return HUMP_ARGMAX / tau


def loadings(tau: npt.ArrayLike, lam: float = DL_LAMBDA) -> FloatArray:
    """Factor loadings at each maturity. Shape (len(tau), 3).

    Columns are level, slope and curvature. At ``tau = 0`` the limits
    (1, 1, 0) are returned.
    """
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    t = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    if (t < 0.0).any():
        raise ValueError("maturities must be non-negative")
    x = lam * t
    safe = np.where(x > 0.0, x, 1.0)
    slope = np.where(x > 0.0, -np.expm1(-safe) / safe, 1.0)
    curvature = slope - np.exp(-x)
    out: FloatArray = np.column_stack([np.ones_like(x), slope, curvature])
    return out


def ns_yield(beta: npt.ArrayLike, tau: npt.ArrayLike, lam: float = DL_LAMBDA) -> FloatArray:
    """Yields implied by factors ``beta``.

    ``beta`` has shape (3,) or (T, 3); the result has shape (len(tau),) or
    (T, len(tau)).
    """
    b = np.asarray(beta, dtype=np.float64)
    out: FloatArray = b @ loadings(tau, lam).T
    return out


def ns_forward(beta: npt.ArrayLike, tau: npt.ArrayLike, lam: float = DL_LAMBDA) -> FloatArray:
    """Instantaneous forward rate curve (Diebold and Li 2006, section 2.2).

    f_t(tau) = b1 + b2 * exp(-lam*tau) + b3 * lam*tau * exp(-lam*tau).
    The yield is the average of the forward rate from 0 to tau (section 2.1).
    """
    b = np.asarray(beta, dtype=np.float64)
    t = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    x = lam * t
    basis = np.column_stack([np.ones_like(x), np.exp(-x), x * np.exp(-x)])
    out: FloatArray = b @ basis.T
    return out


def fit(
    yields: npt.ArrayLike, tau: npt.ArrayLike, lam: float = DL_LAMBDA
) -> tuple[FloatArray, FloatArray]:
    """Least-squares factors for each date.

    Parameters
    ----------
    yields:
        Shape (T, n) or (n,): one curve per row.
    tau:
        The n maturities, in months.
    lam:
        Fixed decay.

    Returns
    -------
    beta:
        Shape (T, 3) (or (3,) for a single curve).
    residuals:
        yields minus fitted yields, same shape as ``yields``.

    """
    y = np.asarray(yields, dtype=np.float64)
    single = y.ndim == 1
    y2 = np.atleast_2d(y)
    design = loadings(tau, lam)
    if design.shape[0] != y2.shape[1]:
        raise ValueError("number of maturities does not match the yield columns")
    if design.shape[0] < 3:
        raise ValueError("need at least three maturities")
    if not np.isfinite(y2).all():
        raise ValueError("yields contain missing or infinite values")
    beta = y2 @ np.linalg.pinv(design).T
    resid = y2 - beta @ design.T
    if single:
        return beta[0], resid[0]
    return beta, resid

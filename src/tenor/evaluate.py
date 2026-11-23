"""Forecast-error statistics and the Diebold-Mariano test.

Diebold and Mariano (1995), section 1.1. For two forecast-error series e1 and
e2 and squared-error loss, the loss differential is d_t = e1_t^2 - e2_t^2. The
statistic is

    S1 = dbar / sqrt( 2*pi*fhat_d(0) / T )

where 2*pi*fhat_d(0) is a weighted sum of the sample autocovariances of d,

    2*pi*fhat_d(0) = sum_{|k| <= S} w(k/S) * gamma_d(k),
    gamma_d(k)     = (1/T) * sum_t (d_t - dbar)(d_{t-|k|} - dbar).

The paper proposes the uniform (rectangular) lag window, w = 1, with
truncation S = h - 1 for h-step-ahead forecasts, because optimal h-step errors
are at most (h - 1)-dependent. Under the null of equal accuracy S1 is
asymptotically standard normal. The uniform window can give a negative
variance estimate; the paper treats that as zero and rejects. A Bartlett
window is the alternative the paper names for a guaranteed positive estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def rmse(errors: npt.ArrayLike) -> float:
    """Root mean squared error."""
    e = np.asarray(errors, dtype=np.float64)
    return float(np.sqrt(np.mean(e**2)))


def autocorr(x: npt.ArrayLike, lag: int) -> float:
    """Sample autocorrelation at displacement ``lag``.

    rho(k) = sum_{t>k} (x_t - xbar)(x_{t-k} - xbar) / sum_t (x_t - xbar)^2.
    Returns nan when the series is constant or shorter than the lag.
    """
    a = np.asarray(x, dtype=np.float64)
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if a.size <= lag:
        return math.nan
    c = a - a.mean()
    denom = float(c @ c)
    if denom == 0.0:
        return math.nan
    if lag == 0:
        return 1.0
    return float(c[lag:] @ c[:-lag]) / denom


@dataclass(frozen=True)
class ErrorStats:
    """Mean, standard deviation and RMSE of a forecast-error series.

    The standard deviation divides by T, so rmse^2 = mean^2 + sd^2 exactly.
    """

    n: int
    mean: float
    sd: float
    rmse: float


def error_stats(errors: npt.ArrayLike) -> ErrorStats:
    """Summarise one forecast-error series as in Diebold and Li, Tables 4 to 6."""
    e = np.asarray(errors, dtype=np.float64)
    if e.ndim != 1 or e.size == 0:
        raise ValueError("errors must be a non-empty one-dimensional array")
    return ErrorStats(n=int(e.size), mean=float(e.mean()), sd=float(e.std()), rmse=rmse(e))


@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano statistic for squared-error loss.

    A negative statistic means the first forecast has the smaller mean squared
    error. ``pvalue`` is two-sided against the standard normal.
    """

    stat: float
    pvalue: float
    mean_diff: float
    long_run_var: float
    n: int


def long_run_variance(d: npt.ArrayLike, lags: int, window: str = "uniform") -> float:
    """Weighted sum of autocovariances of ``d`` up to ``lags``.

    ``window`` is "uniform" (weights 1) or "bartlett" (weights 1 - k/(lags+1)).
    """
    a = np.asarray(d, dtype=np.float64)
    t = a.size
    if lags < 0:
        raise ValueError("lags must be non-negative")
    if window not in ("uniform", "bartlett"):
        raise ValueError("window must be 'uniform' or 'bartlett'")
    c = a - a.mean()
    total = float(c @ c) / t
    for k in range(1, min(lags, t - 1) + 1):
        gamma = float(c[k:] @ c[:-k]) / t
        weight = 1.0 if window == "uniform" else 1.0 - k / (lags + 1.0)
        total += 2.0 * weight * gamma
    return total


def diebold_mariano(
    e1: npt.ArrayLike, e2: npt.ArrayLike, h: int = 1, window: str = "uniform"
) -> DMResult:
    """Test equal mean squared error of two forecasts made for the same dates.

    Parameters
    ----------
    e1, e2:
        Forecast errors on the same dates.
    h:
        Forecast horizon; autocovariances up to lag h - 1 enter the variance.
    window:
        "uniform" as in the paper, or "bartlett".

    """
    a = np.asarray(e1, dtype=np.float64)
    b = np.asarray(e2, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("error series must be one-dimensional and the same length")
    if h < 1:
        raise ValueError("h must be at least 1")
    t = a.size
    if t < 2:
        raise ValueError("need at least two forecasts")
    d = a**2 - b**2
    dbar = float(d.mean())
    lrv = long_run_variance(d, h - 1, window)
    if lrv <= 0.0:
        # Diebold and Mariano (1995, section 1.1): treat a negative estimate
        # as zero and reject. A zero differential is not a rejection.
        stat = 0.0 if dbar == 0.0 else math.copysign(math.inf, dbar)
        pvalue = 1.0 if dbar == 0.0 else 0.0
    else:
        stat = dbar / math.sqrt(lrv / t)
        pvalue = math.erfc(abs(stat) / math.sqrt(2.0))
    return DMResult(stat=stat, pvalue=pvalue, mean_diff=dbar, long_run_var=lrv, n=t)

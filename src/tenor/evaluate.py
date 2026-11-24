"""Forecast-error statistics and tests of equal predictive accuracy.

Diebold-Mariano (1995, section 1.1). For two forecast-error series e1 and e2
and squared-error loss, the loss differential is d_t = e1_t^2 - e2_t^2. The
statistic is

    S1 = dbar / sqrt( V / T )
    V  = sum_{|k| <= h-1} w(k) * gamma_d(k),
    gamma_d(k) = (1/T) * sum_t (d_t - dbar)(d_{t-|k|} - dbar).

The paper proposes the uniform lag window, w = 1, with truncation h - 1,
because optimal h-step errors are at most (h - 1)-dependent. The uniform
window can give a negative variance estimate; the paper treats that as zero
and rejects. A Bartlett window, w(k) = 1 - k/h, is always positive.

Harvey, Leybourne and Newbold (1997, equation 9) correct the small-sample
size of S1 at longer horizons:

    S1* = S1 * sqrt( (T + 1 - 2h + h(h-1)/T) / T )

and compare S1* with a Student t distribution with T - 1 degrees of freedom.

Where the test applies. The normal (or t) limit needs the two models not to be
nested. When one model is a restricted version of the other and the
restriction is true, the two forecasts converge on each other, the loss
differential degenerates, and under recursive estimation the statistic is not
normal (Clark and McCracken 2001). The larger model is then expected to have
the higher mean squared error, because it estimates parameters whose true
values are known under the null.

Clark and West (2007, equations 2.1 and 3.1) give a test for that nested case.
With e_s the errors of the small model, e_l those of the large one and
yhat_s, yhat_l the forecasts,

    f_t = e_s,t^2 - ( e_l,t^2 - (yhat_s,t - yhat_l,t)^2 )

The last term removes the noise the large model's extra parameters add under
the null. The statistic is the t-ratio of the mean of f, with an
autocorrelation-consistent variance for h > 1, and the test is one-sided: a
large positive value rejects equal accuracy in favour of the large model. The
test never rejects in favour of the small model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

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


def hln_factor(n: int, h: int) -> float:
    """Harvey-Leybourne-Newbold multiplier for a Diebold-Mariano statistic."""
    if n < 2 or h < 1:
        raise ValueError("need n >= 2 and h >= 1")
    inside = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if inside <= 0.0:
        raise ValueError("too few forecasts for this horizon")
    return math.sqrt(inside)


def student_t_two_sided(stat: float, df: int) -> float:
    """Two-sided p-value of ``stat`` against a Student t with ``df`` degrees."""
    if math.isinf(stat):
        return 0.0
    return float(2.0 * stats.t.sf(abs(stat), df))


def dm_hln(e1: npt.ArrayLike, e2: npt.ArrayLike, h: int = 1) -> DMResult:
    """Diebold-Mariano with the Harvey-Leybourne-Newbold correction.

    Uniform window with h - 1 lags, statistic scaled by ``hln_factor`` and
    compared with a Student t distribution with T - 1 degrees of freedom.
    """
    raw = diebold_mariano(e1, e2, h=h, window="uniform")
    stat = raw.stat if math.isinf(raw.stat) else raw.stat * hln_factor(raw.n, h)
    pvalue = raw.pvalue if raw.stat == 0.0 else student_t_two_sided(stat, raw.n - 1)
    return DMResult(
        stat=stat, pvalue=pvalue, mean_diff=raw.mean_diff, long_run_var=raw.long_run_var, n=raw.n
    )


@dataclass(frozen=True)
class CWResult:
    """Clark-West statistic for a small model nested in a large one.

    A positive statistic favours the large model. ``pvalue`` is one-sided
    against a Student t with T - 1 degrees of freedom. ``mspe_adjusted`` is the large model's mean
    squared error less the adjustment term.
    """

    stat: float
    pvalue: float
    mspe_small: float
    mspe_large: float
    mspe_adjusted: float
    n: int


def clark_west(
    e_small: npt.ArrayLike,
    e_large: npt.ArrayLike,
    f_small: npt.ArrayLike,
    f_large: npt.ArrayLike,
    h: int = 1,
) -> CWResult:
    """Test equal accuracy of nested models (Clark and West 2007).

    Parameters
    ----------
    e_small, e_large:
        Forecast errors of the restricted and the unrestricted model.
    f_small, f_large:
        The forecasts themselves, on the same dates.
    h:
        Forecast horizon. The variance of the adjusted differential uses the
        uniform window with h - 1 lags, and the statistic carries the
        Harvey-Leybourne-Newbold factor and is compared with a Student t with
        T - 1 degrees of freedom. Clark and West leave the variance estimator
        open; this one had the size closest to nominal in ``tenor.size_sim``.

    """
    es = np.asarray(e_small, dtype=np.float64)
    el = np.asarray(e_large, dtype=np.float64)
    gap = np.asarray(f_small, dtype=np.float64) - np.asarray(f_large, dtype=np.float64)
    if not (es.shape == el.shape == gap.shape) or es.ndim != 1:
        raise ValueError("inputs must be one-dimensional and the same length")
    if h < 1:
        raise ValueError("h must be at least 1")
    t = es.size
    if t < 2:
        raise ValueError("need at least two forecasts")
    adj = float(np.mean(gap**2))
    f = es**2 - (el**2 - gap**2)
    fbar = float(f.mean())
    lrv = long_run_variance(f, h - 1, "uniform")
    if fbar == 0.0:
        stat, pvalue = 0.0, 0.5
    elif lrv <= 0.0:
        stat = math.copysign(math.inf, fbar)
        pvalue = 0.0 if fbar > 0.0 else 1.0
    else:
        stat = fbar / math.sqrt(lrv / t) * hln_factor(t, h)
        pvalue = float(stats.t.sf(stat, t - 1))
    mspe_small = float(np.mean(es**2))
    mspe_large = float(np.mean(el**2))
    return CWResult(
        stat=stat,
        pvalue=pvalue,
        mspe_small=mspe_small,
        mspe_large=mspe_large,
        mspe_adjusted=mspe_large - adj,
        n=t,
    )

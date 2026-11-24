"""How often the accuracy tests reject when the null is true.

Two simulations, both seeded.

``dm_size`` draws two independent forecast-error series of equal variance.
Each is a moving sum of h standard normals, which is the dependence an h-step
error from overlapping monthly forecasts has. Equal accuracy holds by
construction, so a 5% test should reject 5% of the time.

``nested_size`` simulates the case in this repository. The series is a random
walk, the small model is "no change", and the large model is the direct AR(1)
regression estimated recursively, which contains "no change" as a special
case. The null of equal accuracy is true in population. The function reports
how often Clark-West rejects it, and how often a Diebold-Mariano test applied
to the same pair does.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from tenor import evaluate as E

FloatArray = npt.NDArray[np.float64]


def _lrv_rows(d: FloatArray, lags: int, window: str) -> FloatArray:
    """Row-wise version of ``evaluate.long_run_variance``."""
    t = d.shape[1]
    c = d - d.mean(axis=1, keepdims=True)
    total: FloatArray = (c * c).sum(axis=1) / t
    for k in range(1, min(lags, t - 1) + 1):
        gamma = (c[:, k:] * c[:, :-k]).sum(axis=1) / t
        weight = 1.0 if window == "uniform" else 1.0 - k / (lags + 1.0)
        total = total + 2.0 * weight * gamma
    return total


def dm_stats_rows(d: FloatArray, h: int, window: str = "uniform") -> FloatArray:
    """Diebold-Mariano statistic for each row of loss differentials.

    A non-positive variance estimate gives an infinite statistic with the sign
    of the mean, as in ``evaluate.diebold_mariano``.
    """
    t = d.shape[1]
    dbar = d.mean(axis=1)
    lrv = _lrv_rows(d, h - 1, window)
    ok = lrv > 0.0
    out: FloatArray = np.where(dbar == 0.0, 0.0, np.copysign(np.inf, dbar))
    out[ok] = dbar[ok] / np.sqrt(lrv[ok] / t)
    return out


def _moving_sum(z: FloatArray, h: int) -> FloatArray:
    c = np.cumsum(z, axis=1)
    out: FloatArray = c[:, h - 1 :].copy()
    out[:, 1:] -= c[:, :-h]
    return out


def dm_size(h: int, n: int, reps: int, seed: int, level: float = 0.05) -> dict[str, float]:
    """Rejection rates of three Diebold-Mariano variants under equal accuracy.

    Returns the share of ``reps`` simulations rejected two-sided at ``level``
    by the uniform window with normal critical values, the same statistic with
    the Harvey-Leybourne-Newbold correction and t critical values, and the
    Bartlett window with normal critical values.
    """
    rng = np.random.default_rng(seed)
    e1 = _moving_sum(rng.standard_normal((reps, n + h - 1)), h)
    e2 = _moving_sum(rng.standard_normal((reps, n + h - 1)), h)
    d = e1**2 - e2**2
    z = float(stats.norm.ppf(1.0 - level / 2.0))
    tcrit = float(stats.t.ppf(1.0 - level / 2.0, n - 1))
    uniform = np.abs(dm_stats_rows(d, h, "uniform"))
    bartlett = np.abs(dm_stats_rows(d, h, "bartlett"))
    return {
        "dm_uniform_normal": float(np.mean(uniform > z)),
        "dm_hln_t": float(np.mean(uniform * E.hln_factor(n, h) > tcrit)),
        "dm_bartlett_normal": float(np.mean(bartlett > z)),
    }


def recursive_ar1_forecasts(x: FloatArray, h: int, first_target: int) -> FloatArray:
    """Recursive direct AR(1) forecasts for each row of ``x``.

    Column j of the result is the forecast of x[:, first_target + j] made h
    rows earlier from the regression of x_{s+h} on (1, x_s) over all pairs
    known at that origin. Running sums make this one pass per row.
    """
    t = x.shape[1]
    lag, lead = x[:, : t - h], x[:, h:]
    n = np.arange(1, t - h + 1, dtype=np.float64)
    sx, sy = np.cumsum(lag, axis=1), np.cumsum(lead, axis=1)
    sxx, sxy = np.cumsum(lag * lag, axis=1), np.cumsum(lag * lead, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # the first column has one pair and no variance; it is never used
        g = (sxy - sx * sy / n) / (sxx - sx * sx / n)
        c = (sy - g * sx) / n
    origins = np.arange(first_target - h, t - h)
    k = origins - h  # index of the last pair known at each origin
    if k[0] < 2:
        raise ValueError("first_target leaves too few pairs for the first forecast")
    out: FloatArray = c[:, k] + g[:, k] * x[:, origins]
    return out


def nested_size(
    h: int, n: int, reps: int, seed: int, initial: int = 108, level: float = 0.05
) -> dict[str, float]:
    """Rejection rates for a random walk against a recursively estimated AR(1).

    Returns the one-sided Clark-West rejection rate at ``level``, the
    two-sided rejection rate of the Harvey-Leybourne-Newbold Diebold-Mariano
    test on the same pair, the share of all simulations in which that test
    rejects in favour of the random walk, and the share in which the AR(1) has
    the higher mean squared error.
    """
    rng = np.random.default_rng(seed)
    t = initial + n
    x = np.cumsum(rng.standard_normal((reps, t)), axis=1)
    f_large = recursive_ar1_forecasts(x, h, initial)
    f_small = x[:, initial - h : t - h]
    actual = x[:, initial:]
    e_small, e_large = actual - f_small, actual - f_large
    gap = f_small - f_large
    f = e_small**2 - (e_large**2 - gap**2)
    cw = dm_stats_rows(f, h, "uniform") * E.hln_factor(n, h)
    d = e_large**2 - e_small**2
    dm = dm_stats_rows(d, h, "uniform") * E.hln_factor(n, h)
    tcrit = float(stats.t.ppf(1.0 - level / 2.0, n - 1))
    return {
        "cw_one_sided": float(np.mean(cw > stats.t.ppf(1.0 - level, n - 1))),
        "dm_hln_t_two_sided": float(np.mean(np.abs(dm) > tcrit)),
        "dm_hln_t_favouring_random_walk": float(np.mean(dm > tcrit)),
        "ar1_has_higher_mse": float(np.mean(d.mean(axis=1) > 0.0)),
    }


def size_table(
    horizons: tuple[int, ...] = (1, 6, 12),
    sizes: tuple[int, ...] = (84, 394),
    reps: int = 5000,
    seed: int = 20261123,
) -> pd.DataFrame:
    """Rejection rates in percent at a nominal 5%, for each horizon and length."""
    rows: list[dict[str, object]] = []
    for n in sizes:
        for h in horizons:
            for design, rates in (
                ("independent errors, equal accuracy", dm_size(h, n, reps, seed + 1000 * h + n)),
                (
                    "random walk against recursive AR(1)",
                    nested_size(h, n, reps, seed + 1000 * h + n + 7),
                ),
            ):
                for test, rate in rates.items():
                    rows.append(
                        {
                            "design": design,
                            "statistic": test,
                            "horizon_months": h,
                            "forecasts": n,
                            "simulations": reps,
                            "nominal_pct": 5.0,
                            "rate_pct": 100.0 * rate,
                        }
                    )
    return pd.DataFrame(rows)

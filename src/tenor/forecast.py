"""Recursive, direct h-step forecasts of the yield curve.

Everything here follows Diebold and Li (2006), sections 3.3 and 3.4. The paper
does not number its equations, so the section is cited instead.

Nelson-Siegel with AR(1) factors (section 3.3):

    yhat_{t+h|t}(tau) = b1hat_{t+h|t} + b2hat_{t+h|t} * slope(tau) + b3hat_{t+h|t} * curv(tau)
    bihat_{t+h|t}     = c_i + g_i * bihat_t

where c_i and g_i come from regressing bihat_t on an intercept and bihat_{t-h}
(footnote 11: a direct regression of the factor at t+h on the factor at t).

Nelson-Siegel with VAR(1) factors (section 3.3): bhat_{t+h|t} = c + G * bhat_t.

Random walk (section 3.4, competitor 1): yhat_{t+h|t}(tau) = y_t(tau).

AR(1) on yield levels (section 3.4, competitor 5): yhat_{t+h|t}(tau) = c(tau) + g(tau) * y_t(tau).

Nelson-Siegel with random-walk factors is not in the paper. It sets
bihat_{t+h|t} = bihat_t, so the forecast is today's fitted curve. It is the
Nelson-Siegel AR(1) model with c_i = 0 and g_i = 1 imposed, which makes it
the nested benchmark that separates the cost of the curve fit from the cost of
forecasting the factors back towards their means.

Estimation is recursive by default: a forecast made at t uses rows from the
start of the sample up to and including t and nothing later. With ``window``
set, only the last ``window`` rows up to t are used (a rolling scheme). Every
model is evaluated on the same origins and targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tenor import nelson_siegel as ns

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Model keys, in the order used in tables.
MODELS: tuple[str, ...] = ("ns_ar", "ns_var", "ns_rw", "ar_yield", "rw")

MODEL_LABELS: dict[str, str] = {
    "ns_ar": "Nelson-Siegel, AR(1) factors",
    "ns_var": "Nelson-Siegel, VAR(1) factors",
    "ns_rw": "Nelson-Siegel, random-walk factors",
    "ar_yield": "AR(1) on yield levels",
    "rw": "Random walk",
}


def direct_ar1(x: npt.ArrayLike, h: int) -> tuple[FloatArray, FloatArray]:
    """Intercept and slope of the direct h-step regression, column by column.

    Regresses x_{s+h} on (1, x_s) for every s with s + h inside the sample.
    ``x`` has shape (T,) or (T, k); each column gets its own regression.
    """
    a = np.asarray(x, dtype=np.float64)
    a2 = a[:, None] if a.ndim == 1 else a
    t = a2.shape[0]
    if h < 1:
        raise ValueError("h must be at least 1")
    if t - h < 3:
        raise ValueError("need at least three (x_s, x_{s+h}) pairs")
    lag, lead = a2[: t - h], a2[h:]
    lag_mean, lead_mean = lag.mean(axis=0), lead.mean(axis=0)
    lag_c = lag - lag_mean
    var = (lag_c**2).sum(axis=0)
    if (var == 0.0).any():
        raise ValueError("a regressor is constant")
    g = (lag_c * (lead - lead_mean)).sum(axis=0) / var
    c = lead_mean - g * lag_mean
    return c, g


def direct_ar1_forecast(x: npt.ArrayLike, h: int) -> FloatArray:
    """Forecast of x_{T-1+h} from the direct AR(1) regression on rows 0..T-1."""
    a = np.asarray(x, dtype=np.float64)
    a2 = a[:, None] if a.ndim == 1 else a
    c, g = direct_ar1(a2, h)
    out: FloatArray = c + g * a2[-1]
    return out


def direct_var1(x: npt.ArrayLike, h: int) -> tuple[FloatArray, FloatArray]:
    """Intercept vector c and matrix G of the direct h-step VAR(1) regression.

    Regresses x_{s+h} on (1, x_s) jointly; row i of G holds the coefficients
    for variable i.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("x must have shape (T, k)")
    t, k = a.shape
    if h < 1:
        raise ValueError("h must be at least 1")
    if t - h < k + 2:
        raise ValueError("too few (x_s, x_{s+h}) pairs for a VAR(1)")
    design = np.column_stack([np.ones(t - h), a[: t - h]])
    coef, *_ = np.linalg.lstsq(design, a[h:], rcond=None)
    c: FloatArray = coef[0]
    big_g: FloatArray = coef[1:].T
    return c, big_g


def direct_var1_forecast(x: npt.ArrayLike, h: int) -> FloatArray:
    """Forecast of x_{T-1+h} from the direct VAR(1) regression on rows 0..T-1."""
    a = np.asarray(x, dtype=np.float64)
    c, big_g = direct_var1(a, h)
    out: FloatArray = c + big_g @ a[-1]
    return out


@dataclass(frozen=True)
class ForecastSet:
    """Forecasts from every model on one common set of dates.

    Attributes
    ----------
    horizon:
        h, in rows (months for a monthly panel).
    origins:
        Row index at which each forecast is made.
    targets:
        Row index being forecast, always origins + horizon.
    actual:
        Realised yields at the targets. Shape (number of forecasts, n).
    forecasts:
        One array of the same shape per model key.

    """

    horizon: int
    origins: IntArray
    targets: IntArray
    actual: FloatArray
    forecasts: dict[str, FloatArray]

    def errors(self, model: str) -> FloatArray:
        """Forecast errors y_{t+h}(tau) - yhat_{t+h|t}(tau) (section 3.4)."""
        out: FloatArray = self.actual - self.forecasts[model]
        return out


def recursive_forecasts(
    yields: npt.ArrayLike,
    tau: npt.ArrayLike,
    h: int,
    first_target: int,
    last_target: int | None = None,
    *,
    lam: float = ns.DL_LAMBDA,
    start: int = 0,
    window: int | None = None,
) -> ForecastSet:
    """Forecast every target row from ``first_target`` to ``last_target``.

    Parameters
    ----------
    yields:
        Panel with one row per period and one column per maturity. Shape (T, n).
    tau:
        The n maturities in months.
    h:
        Forecast horizon in rows.
    first_target, last_target:
        Inclusive range of target rows. The forecast for target r is made at
        origin r - h. ``last_target`` defaults to the last row.
    lam:
        Nelson-Siegel decay.
    start:
        First row of the estimation sample. Rows before it are never used.
    window:
        If given, each forecast is estimated on the last ``window`` rows up to
        its origin (never reaching before ``start``). ``None`` is recursive.

    """
    y = np.asarray(yields, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("yields must have shape (T, n)")
    t = y.shape[0]
    last = t - 1 if last_target is None else last_target
    if not (0 <= start < first_target <= last < t):
        raise ValueError("need 0 <= start < first_target <= last_target < T")
    if first_target - h < start:
        raise ValueError("first origin falls before the start of the sample")
    design = ns.loadings(tau, lam)
    # The factors on date t depend on the curve of date t only, so computing
    # them once for all dates introduces no look-ahead.
    beta, _ = ns.fit(y, tau, lam)

    targets = np.arange(first_target, last + 1, dtype=np.int64)
    origins = targets - h
    shape = (targets.size, y.shape[1])
    out = {name: np.empty(shape, dtype=np.float64) for name in MODELS}
    if window is not None and window - h < 5:
        raise ValueError("window is too short for the horizon")
    for i, origin in enumerate(origins):
        lo = start if window is None else max(start, origin + 1 - window)
        known_beta = beta[lo : origin + 1]
        known_y = y[lo : origin + 1]
        out["ns_ar"][i] = direct_ar1_forecast(known_beta, h) @ design.T
        out["ns_var"][i] = direct_var1_forecast(known_beta, h) @ design.T
        out["ns_rw"][i] = beta[origin] @ design.T
        out["ar_yield"][i] = direct_ar1_forecast(known_y, h)
        out["rw"][i] = y[origin]
    return ForecastSet(
        horizon=h, origins=origins, targets=targets, actual=y[targets].copy(), forecasts=out
    )

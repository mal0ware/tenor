"""A simulated daily yield panel with exactly three factors plus small noise.

Used by the tests and by the smoke run of the pipeline, which must work with
no network. Nothing in the committed results comes from this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tenor import nelson_siegel as ns
from tenor.data import CORE_TENORS, TENOR_MONTHS
from tenor.gsw import DL_PANEL_MONTHS


def simulate_panel(
    n_days: int = 4000,
    seed: int = 20261123,
    tenors: tuple[str, ...] = CORE_TENORS,
    noise_sd: float = 0.01,
    start: str = "1990-01-02",
) -> pd.DataFrame:
    """Simulate business-day yields from three AR(1) Nelson-Siegel factors.

    Level, slope and curvature each follow a daily AR(1) around a fixed mean.
    Yields are the Nelson-Siegel curve at the Diebold-Li decay plus independent
    noise of standard deviation ``noise_sd`` percentage points.
    """
    tau = [TENOR_MONTHS[c] for c in tenors]
    return _simulate(n_days, seed, tau, list(tenors), noise_sd, start)


def simulate_zero_panel(
    n_days: int = 6000,
    seed: int = 20261124,
    months: tuple[int, ...] = DL_PANEL_MONTHS,
    noise_sd: float = 0.01,
    start: str = "1985-01-01",
) -> pd.DataFrame:
    """Simulate a zero-coupon panel whose columns are maturities in months.

    Same process as ``simulate_panel``. The default starts in 1985 and runs
    for about 23 years, so the paper's 1994 to 2000 target window is inside it.
    """
    return _simulate(n_days, seed, [float(m) for m in months], list(months), noise_sd, start)


def _simulate(
    n_days: int,
    seed: int,
    tau: list[float],
    columns: list[str] | list[int],
    noise_sd: float,
    start: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    means = np.array([6.0, -2.0, 0.5])
    phi = np.array([0.9995, 0.998, 0.995])
    shock_sd = np.array([0.05, 0.06, 0.10])
    beta = np.empty((n_days, 3))
    state = means.copy()
    for t in range(n_days):
        state = means + phi * (state - means) + shock_sd * rng.standard_normal(3)
        beta[t] = state
    yields = ns.ns_yield(beta, tau) + noise_sd * rng.standard_normal((n_days, len(tau)))
    index = pd.bdate_range(start=start, periods=n_days, name="Date")
    return pd.DataFrame(np.round(yields, 4), index=index, columns=columns)

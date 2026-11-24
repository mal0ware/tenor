"""Build every table and figure from a par yield panel and a zero-coupon panel.

``run`` is the single entry point. It is deterministic: the same panels give
byte-identical tables. ``scripts/reproduce.py`` calls it on the Treasury and
Federal Reserve data; the tests and the CI smoke run call it on simulated
panels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from tenor import data as D
from tenor import evaluate as E
from tenor import forecast as F
from tenor import nelson_siegel as ns
from tenor import pca as P
from tenor import published
from tenor import size_sim as S

FloatArray = npt.NDArray[np.float64]

INK = "#1b1f24"
MUTED = "#6b7280"
COLOURS = {"pc1": "#12355b", "pc2": "#c2622d", "pc3": "#5b8a3a"}
HORIZON_COLOURS = {1: "#8fb8de", 6: "#3f7fbf", 12: "#12355b"}
PERIOD_COLOURS = ("#12355b", "#3f7fbf", "#c2622d", "#5b8a3a")
MODEL_COLOURS = {"ns_ar": "#12355b", "ar_yield": "#3f7fbf", "ns_rw": "#c2622d"}


@dataclass(frozen=True)
class Config:
    """Settings for the committed results.

    Attributes
    ----------
    horizons:
        Forecast horizons in months.
    zero_start:
        First month of the zero-coupon estimation sample. 1985-01 is the first
        month of Diebold and Li's sample.
    zero_first_target:
        First forecast target on the zero-coupon panel. 1994-01 is the paper's.
    zero_subperiods:
        Labels and inclusive target-month ranges that split the zero-coupon
        targets. The first is the paper's evaluation window.
    initial_months:
        Months of par yield data before the first par forecast target. 108 is
        the length of the paper's first estimation window.
    par_subperiods:
        The same kind of split for the par yield targets.
    zero_bound:
        Target months left out of the "excluding" sample.
    rolling_estimation_months:
        Window length of the rolling estimation scheme.
    rolling_window:
        Trading days per rolling principal-components window (about 5 years).
    rolling_step:
        Trading days between rolling windows.
    crisis_window:
        Dates over which the 3-month yield's daily volatility is tabulated.
    size_simulations:
        Simulations per cell of the test-size table.

    """

    horizons: tuple[int, ...] = (1, 6, 12)
    zero_start: str = "1985-01"
    zero_first_target: str = "1994-01"
    zero_subperiods: tuple[tuple[str, str, str], ...] = (
        ("1994-2000", "1994-01", "2000-12"),
        ("2001-2008", "2001-01", "2008-12"),
        ("2009-2017", "2009-01", "2017-12"),
        ("2018-2026", "2018-01", "2026-12"),
    )
    initial_months: int = 108
    par_subperiods: tuple[tuple[str, str, str], ...] = (
        ("1999-2008", "1999-01", "2008-12"),
        ("2009-2017", "2009-01", "2017-12"),
        ("2018-2026", "2018-01", "2026-12"),
    )
    zero_bound: tuple[str, str] = ("2009-01", "2017-12")
    rolling_estimation_months: int = 108
    rolling_window: int = 1250
    rolling_step: int = 21
    crisis_window: tuple[str, str] = ("2008-09-01", "2008-12-31")
    size_simulations: int = 5000


def _write_csv(frame: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(decimals)
    out.to_csv(path, index=False, lineterminator="\n", float_format=f"%.{decimals}f")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def _label_years(tau_months: float) -> str:
    if tau_months < 12:
        return f"{tau_months:g}m"
    return f"{tau_months / 12:g}y"


# --------------------------------------------------------------------------
# principal components
# --------------------------------------------------------------------------


def pca_variants(daily: pd.DataFrame, wide: pd.DataFrame | None) -> pd.DataFrame:
    """Variance shares of the first components under each data treatment."""
    monthly = D.month_end(daily)
    treatments: list[tuple[str, pd.DataFrame, bool]] = [
        ("daily changes, covariance", D.changes_bp(daily), False),
        ("daily changes, correlation", D.changes_bp(daily), True),
        ("weekly changes, covariance", D.changes_bp(D.week_end(daily)), False),
        ("monthly changes, covariance", D.changes_bp(monthly), False),
        ("monthly levels, covariance", monthly, False),
    ]
    if wide is not None and len(wide) > 10:
        treatments.append(
            ("daily changes, covariance, with 20y and 30y", D.changes_bp(wide), False)
        )
    rows = []
    for name, frame, standardize in treatments:
        res = P.pca(frame.to_numpy(), standardize=standardize)
        rows.append(
            {
                "treatment": name,
                "first_date": frame.index[0].strftime("%Y-%m-%d"),
                "last_date": frame.index[-1].strftime("%Y-%m-%d"),
                "observations": len(frame),
                "maturities": frame.shape[1],
                "pc1_pct": 100 * res.explained[0],
                "pc2_pct": 100 * res.explained[1],
                "pc3_pct": 100 * res.explained[2],
                "first_three_pct": 100 * res.explained[:3].sum(),
                "pc4_pct": 100 * res.explained[3],
            }
        )
    return pd.DataFrame(rows)


def pca_detail(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, P.PCAResult]:
    """Loadings and per-maturity explained shares for daily changes."""
    changes = D.changes_bp(daily)
    res = P.pca(changes.to_numpy())
    tau = D.tenor_months(daily.columns)
    sd_bp = np.sqrt(res.eigenvalues[:3])
    loadings = pd.DataFrame(
        {
            "maturity_months": tau,
            "pc1_loading": res.loadings[:, 0],
            "pc2_loading": res.loadings[:, 1],
            "pc3_loading": res.loadings[:, 2],
            "pc1_one_sd_bp": res.loadings[:, 0] * sd_bp[0],
            "pc2_one_sd_bp": res.loadings[:, 1] * sd_bp[1],
            "pc3_one_sd_bp": res.loadings[:, 2] * sd_bp[2],
        }
    )
    c1, c2, c3 = (P.communality(res, k) for k in (1, 2, 3))
    by_tenor = pd.DataFrame(
        {
            "maturity_months": tau,
            "sd_daily_change_bp": changes.to_numpy().std(axis=0, ddof=1),
            "explained_by_three_pct": 100 * c3,
            "pc1_share_of_explained_pct": 100 * c1 / c3,
            "pc2_share_of_explained_pct": 100 * (c2 - c1) / c3,
            "pc3_share_of_explained_pct": 100 * (c3 - c2) / c3,
        }
    )
    return loadings, by_tenor, res


def pca_rolling(daily: pd.DataFrame, window: int, step: int) -> pd.DataFrame:
    """Variance shares over rolling windows of daily changes."""
    changes = D.changes_bp(daily)
    window = min(window, len(changes))
    ends, shares = P.rolling_explained(changes.to_numpy(), window=window, step=step, k=3)
    return pd.DataFrame(
        {
            "window_end": changes.index[ends].strftime("%Y-%m-%d"),
            "pc1_pct": 100 * shares[:, 0],
            "pc2_pct": 100 * shares[:, 1],
            "pc3_pct": 100 * shares[:, 2],
            "first_three_pct": 100 * shares.sum(axis=1),
        }
    )


# --------------------------------------------------------------------------
# Nelson-Siegel
# --------------------------------------------------------------------------


def empirical_factors(monthly: pd.DataFrame) -> pd.DataFrame:
    """Level, slope and curvature as Diebold and Li define them (Table 1 note).

    Level is the 10-year yield, slope the 10-year minus the 3-month yield, and
    curvature twice the 2-year yield minus the sum of the 3-month and 10-year.
    """
    return pd.DataFrame(
        {
            "level": monthly["10 Yr"],
            "slope": monthly["10 Yr"] - monthly["3 Mo"],
            "curvature": 2 * monthly["2 Yr"] - monthly["3 Mo"] - monthly["10 Yr"],
        }
    )


def ns_tables(
    monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the curve each month and summarise residuals and factors."""
    tau = D.tenor_months(monthly.columns)
    beta, resid = ns.fit(monthly.to_numpy(), tau)
    betas = pd.DataFrame(beta, index=monthly.index, columns=["beta1", "beta2", "beta3"])

    resid_rows = []
    for j, t in enumerate(tau):
        r = resid[:, j]
        resid_rows.append(
            {
                "maturity_months": t,
                "mean": r.mean(),
                "sd": r.std(),
                "min": r.min(),
                "max": r.max(),
                "mae": np.abs(r).mean(),
                "rmse": E.rmse(r),
                "acf1": E.autocorr(r, 1),
                "acf12": E.autocorr(r, 12),
            }
        )
    residuals = pd.DataFrame(resid_rows)

    stats_rows = []
    for name in betas.columns:
        b = betas[name].to_numpy()
        stats_rows.append(
            {
                "factor": name,
                "mean": b.mean(),
                "sd": b.std(ddof=1),
                "min": b.min(),
                "max": b.max(),
                "acf1": E.autocorr(b, 1),
                "acf12": E.autocorr(b, 12),
                "acf30": E.autocorr(b, 30),
            }
        )
    factor_stats = pd.DataFrame(stats_rows)

    emp = empirical_factors(monthly)
    pcs = P.scores(P.pca(monthly.to_numpy()), monthly.to_numpy())
    pairs = [("beta1", "level", 0), ("beta2", "slope", 1), ("beta3", "curvature", 2)]
    corr_rows = []
    for beta_name, emp_name, k in pairs:
        b = betas[beta_name].to_numpy()
        corr_rows.append(
            {
                "factor": beta_name,
                "empirical_measure": emp_name,
                "correlation": float(np.corrcoef(b, emp[emp_name].to_numpy())[0, 1]),
                "paper_correlation": published.DL_FACTOR_CORRELATIONS[emp_name],
                "abs_correlation_with_pc": abs(float(np.corrcoef(b, pcs[:, k])[0, 1])),
            }
        )
    correlations = pd.DataFrame(corr_rows)
    return betas, residuals, factor_stats, correlations


def ns_residuals_zero(monthly: pd.DataFrame, tau: list[float]) -> pd.DataFrame:
    """Fit the curve to the zero-coupon panel and summarise the residuals."""
    _, resid = ns.fit(monthly.to_numpy(), tau)
    rows = [
        {
            "maturity_months": t,
            "mean": resid[:, j].mean(),
            "sd": resid[:, j].std(),
            "mae": np.abs(resid[:, j]).mean(),
            "rmse": E.rmse(resid[:, j]),
            "acf1": E.autocorr(resid[:, j], 1),
            "acf12": E.autocorr(resid[:, j], 12),
        }
        for j, t in enumerate(tau)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# forecasts
# --------------------------------------------------------------------------

#: (small, large) pairs in which the small model is a restricted large model.
NESTED_PAIRS: tuple[tuple[str, str], ...] = (
    ("rw", "ar_yield"),
    ("ns_rw", "ns_ar"),
    ("ns_rw", "ns_var"),
)

#: Models that contain the random walk as a special case. For these a
#: Diebold-Mariano test against the random walk has no normal limit under
#: recursive estimation.
NESTS_RANDOM_WALK: frozenset[str] = frozenset({"ar_yield"})


def _month_index(monthly: pd.DataFrame, month: str, side: str) -> int | None:
    """Row of the first month >= ``month`` (side "left") or last <= (side "right")."""
    periods = monthly.index.to_period("M")
    target = pd.Period(month, freq="M")
    if side == "left":
        hits = np.flatnonzero(periods >= target)
        return int(hits[0]) if hits.size else None
    hits = np.flatnonzero(periods <= target)
    return int(hits[-1]) if hits.size else None


def _error_lags(h: int) -> tuple[int, int]:
    """Displacements at which Diebold and Li report error autocorrelations."""
    return (1, 12) if h == 1 else (h, h + 12)


@dataclass
class _Rows:
    errors: list[dict[str, object]]
    versus: list[dict[str, object]]
    nested: list[dict[str, object]]


def _score(
    fs: F.ForecastSet,
    mask: npt.NDArray[np.bool_],
    tau: list[float],
    panel: str,
    sample: str,
    rows: _Rows,
) -> None:
    """Append error statistics and accuracy tests for the masked forecasts."""
    h = fs.horizon
    lag_a, lag_b = _error_lags(h)
    key = {"panel": panel, "sample": sample, "horizon_months": h}
    rw = fs.errors("rw")[mask]
    for name in F.MODELS:
        err = fs.errors(name)[mask]
        for j, t in enumerate(tau):
            s = E.error_stats(err[:, j])
            rows.errors.append(
                {
                    **key,
                    "model": name,
                    "maturity_months": t,
                    "forecasts": s.n,
                    "mean": s.mean,
                    "sd": s.sd,
                    "rmse": s.rmse,
                    "acf_lag_a": E.autocorr(err[:, j], lag_a),
                    "acf_lag_b": E.autocorr(err[:, j], lag_b),
                }
            )
            if name == "rw":
                continue
            rw_rmse = E.rmse(rw[:, j])
            uni = E.diebold_mariano(err[:, j], rw[:, j], h=h, window="uniform")
            hln = E.dm_hln(err[:, j], rw[:, j], h=h)
            bart = E.diebold_mariano(err[:, j], rw[:, j], h=h, window="bartlett")
            rows.versus.append(
                {
                    **key,
                    "model": name,
                    "maturity_months": t,
                    "forecasts": s.n,
                    "rmse": s.rmse,
                    "rmse_random_walk": rw_rmse,
                    "rmse_ratio": s.rmse / rw_rmse,
                    "nests_random_walk": name in NESTS_RANDOM_WALK,
                    "dm_uniform": uni.stat,
                    "dm_hln": hln.stat,
                    "dm_hln_p": hln.pvalue,
                    "dm_bartlett": bart.stat,
                    "dm_bartlett_p": bart.pvalue,
                }
            )
    for small, large in NESTED_PAIRS:
        e_s, e_l = fs.errors(small)[mask], fs.errors(large)[mask]
        f_s, f_l = fs.forecasts[small][mask], fs.forecasts[large][mask]
        for j, t in enumerate(tau):
            cw = E.clark_west(e_s[:, j], e_l[:, j], f_s[:, j], f_l[:, j], h=h)
            rows.nested.append(
                {
                    **key,
                    "small_model": small,
                    "large_model": large,
                    "maturity_months": t,
                    "forecasts": cw.n,
                    "rmse_small": math.sqrt(cw.mspe_small),
                    "rmse_large": math.sqrt(cw.mspe_large),
                    "rmse_ratio": math.sqrt(cw.mspe_large / cw.mspe_small),
                    "cw_stat": cw.stat,
                    "cw_p_one_sided": cw.pvalue,
                }
            )


def forecast_tables(
    panel: str,
    monthly: pd.DataFrame,
    tau: list[float],
    first_target: int,
    subperiods: tuple[tuple[str, str, str], ...],
    cfg: Config,
    window: int | None = None,
) -> tuple[_Rows, dict[str, str]]:
    """Forecast one panel and score every sample of its target months.

    The samples are "full", each of ``subperiods``, and the full sample
    without the target months in ``cfg.zero_bound``. A sample with fewer than
    24 forecasts, or one that would repeat "full", is skipped.
    """
    y = monthly.to_numpy()
    periods = monthly.index.to_period("M")
    rows = _Rows([], [], [])
    spans: dict[str, str] = {}
    if not (max(cfg.horizons) + 16 <= first_target < len(monthly)):
        return rows, spans
    spans["full"] = f"{periods[first_target]} to {periods[-1]}"
    zb_lo, zb_hi = (pd.Period(m, freq="M") for m in cfg.zero_bound)
    zb_label = f"excluding {zb_lo.year}-{zb_hi.year}"
    for h in cfg.horizons:
        fs = F.recursive_forecasts(y, tau, h, first_target=first_target, window=window)
        target_periods = periods[fs.targets]
        everything = np.ones(fs.targets.size, dtype=np.bool_)
        _score(fs, everything, tau, panel, "full", rows)
        masks: list[tuple[str, npt.NDArray[np.bool_]]] = [
            (
                label,
                np.asarray(
                    (target_periods >= pd.Period(lo, freq="M"))
                    & (target_periods <= pd.Period(hi, freq="M"))
                ),
            )
            for label, lo, hi in subperiods
        ]
        masks.append((zb_label, np.asarray((target_periods < zb_lo) | (target_periods > zb_hi))))
        for label, mask in masks:
            if mask.sum() < 24 or mask.all():
                continue
            if label == zb_label:
                spans[label] = f"{spans['full']} without {zb_lo} to {zb_hi}"
            else:
                spans[label] = f"{target_periods[mask][0]} to {target_periods[mask][-1]}"
            _score(fs, mask, tau, panel, label, rows)
    return rows, spans


def paper_comparison(errors: pd.DataFrame, versus: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Set the zero-coupon paper-window numbers beside the printed ones."""
    sub_e = errors[(errors["panel"] == "zero") & (errors["sample"] == sample)]
    sub_v = versus[(versus["panel"] == "zero") & (versus["sample"] == sample)]
    rows = []
    for model, by_h in published.DL_RMSE.items():
        for h, printed in by_h.items():
            for tau, paper_rmse in zip(published.DL_MATURITIES, printed, strict=True):
                here = sub_e[
                    (sub_e["model"] == model)
                    & (sub_e["horizon_months"] == h)
                    & (sub_e["maturity_months"] == float(tau))
                ]
                if here.empty:
                    continue
                k = published.DL_MATURITIES.index(tau)
                rmse_here = float(here["rmse"].iloc[0])
                row: dict[str, object] = {
                    "horizon_months": h,
                    "model": model,
                    "maturity_months": float(tau),
                    "forecasts": int(here["forecasts"].iloc[0]),
                    "rmse_here": rmse_here,
                    "rmse_paper": paper_rmse,
                    "rmse_gap": rmse_here - paper_rmse,
                    "ratio_to_rw_paper": paper_rmse / published.DL_RMSE["rw"][h][k],
                }
                ratio_here, dm_here, dm_hln, dm_hln_p = 1.0, math.nan, math.nan, math.nan
                if model != "rw":
                    v = sub_v[
                        (sub_v["model"] == model)
                        & (sub_v["horizon_months"] == h)
                        & (sub_v["maturity_months"] == float(tau))
                    ]
                    ratio_here = float(v["rmse_ratio"].iloc[0])
                    dm_here = float(v["dm_uniform"].iloc[0])
                    dm_hln = float(v["dm_hln"].iloc[0])
                    dm_hln_p = float(v["dm_hln_p"].iloc[0])
                row["ratio_to_rw_here"] = ratio_here
                row["dm_vs_rw_here"] = dm_here
                dm_paper = math.nan
                if model == "ns_ar" and h in published.DL_DM_VS_RW:
                    dm_paper = published.DL_DM_VS_RW[h][k]
                row["dm_vs_rw_paper"] = dm_paper
                row["dm_hln_here"] = dm_hln
                row["dm_hln_p_here"] = dm_hln_p
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    cols = [
        "horizon_months",
        "model",
        "maturity_months",
        "forecasts",
        "rmse_here",
        "rmse_paper",
        "rmse_gap",
        "ratio_to_rw_here",
        "ratio_to_rw_paper",
        "dm_vs_rw_here",
        "dm_vs_rw_paper",
        "dm_hln_here",
        "dm_hln_p_here",
    ]
    return pd.DataFrame(rows)[cols].sort_values(
        ["horizon_months", "model", "maturity_months"], kind="stable"
    )


def bill_volatility(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Standard deviation of daily 3-month yield changes, crisis months and all."""
    changes = D.changes_bp(daily)["3 Mo"]
    lo, hi = cfg.crisis_window
    rows = []
    for label, series in (
        (f"{lo} to {hi}", changes.loc[lo:hi]),
        ("whole sample", changes),
    ):
        if len(series) < 2:
            continue
        rows.append(
            {
                "period": label,
                "first_date": series.index[0].strftime("%Y-%m-%d"),
                "last_date": series.index[-1].strftime("%Y-%m-%d"),
                "daily_changes": len(series),
                "sd_3m_change_bp": float(series.std(ddof=1)),
                "largest_abs_3m_change_bp": float(series.abs().max()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

STYLE: dict[str, object] = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e3e6ea",
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "text.color": INK,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

MATURITY_TICKS: tuple[float, ...] = (3.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0)


def _maturity_axis(ax: plt.Axes) -> None:
    ax.set_xticks(list(MATURITY_TICKS))
    ax.set_xticklabels([_label_years(t) for t in MATURITY_TICKS])
    ax.set_xlabel("Maturity")


def _ratio_rows(versus: pd.DataFrame, panel: str, sample: str, model: str, h: int) -> pd.DataFrame:
    sub = versus[
        (versus["panel"] == panel)
        & (versus["sample"] == sample)
        & (versus["model"] == model)
        & (versus["horizon_months"] == h)
    ]
    return sub.sort_values("maturity_months")


def fig_rmse_ratio(versus: pd.DataFrame, spans: dict[str, str], cfg: Config, path: Path) -> None:
    """Nelson-Siegel AR(1) against the random walk: the paper's window, then the long sample.

    Left: this repository's forecasts for the paper's target months beside the
    ratios implied by the paper's printed tables. Right: the same model and
    data scored on every target month available.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2), sharey=True)
    paper_label = cfg.zero_subperiods[0][0]
    panels = [
        (axes[0], paper_label, f"The paper's window: targets {spans.get(paper_label, '')}"),
        (axes[1], "full", f"Long sample: targets {spans.get('full', '')}"),
    ]
    for ax, sample, title in panels:
        for h in cfg.horizons:
            rows = _ratio_rows(versus, "zero", sample, "ns_ar", h)
            if rows.empty:
                continue
            colour = HORIZON_COLOURS.get(h, INK)
            ax.plot(
                rows["maturity_months"],
                rows["rmse_ratio"],
                marker="o",
                color=colour,
                label=f"{h}-month horizon, this repository",
            )
            if sample == paper_label and h in published.DL_RMSE["ns_ar"]:
                ratio = [
                    a / b
                    for a, b in zip(
                        published.DL_RMSE["ns_ar"][h], published.DL_RMSE["rw"][h], strict=True
                    )
                ]
                ax.plot(
                    published.DL_MATURITIES,
                    ratio,
                    marker="s",
                    markersize=6,
                    markerfacecolor="white",
                    linestyle="--",
                    linewidth=1.1,
                    color=colour,
                    label=f"{h}-month horizon, printed in Diebold-Li",
                )
        ax.axhline(1.0, color=INK, linewidth=0.9)
        ax.set_title(title)
        _maturity_axis(ax)
    axes[0].set_ylabel("Forecast RMSE / random-walk RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: ("printed" in labels[i], i))
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 0.0),
    )
    axes[1].text(
        0.99,
        0.03,
        "Below 1: the forecast beat the random walk",
        transform=axes[1].transAxes,
        ha="right",
        color=MUTED,
        fontsize=8,
    )
    fig.suptitle(
        "Nelson-Siegel AR(1) forecasts of zero-coupon yields against no change",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    _save(fig, path)


def fig_subperiods(versus: pd.DataFrame, cfg: Config, path: Path) -> None:
    """Longest-horizon RMSE ratio by maturity, one line per period of the targets."""
    h = max(cfg.horizons)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for shade, (label, _, _) in zip(PERIOD_COLOURS, cfg.zero_subperiods, strict=False):
        rows = _ratio_rows(versus, "zero", label, "ns_ar", h)
        if rows.empty:
            continue
        ax.plot(rows["maturity_months"], rows["rmse_ratio"], marker="o", color=shade, label=label)
    ax.axhline(1.0, color=INK, linewidth=0.9)
    _maturity_axis(ax)
    ax.set_ylabel("Forecast RMSE / random-walk RMSE")
    ax.set_title(f"Nelson-Siegel AR(1), {h}-month horizon, by period of the target month")
    ax.legend(title="Targets", title_fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def fig_variants(versus: pd.DataFrame, spans: dict[str, str], cfg: Config, path: Path) -> None:
    """Longest-horizon ratio for each model, with and without the zero-bound years."""
    h = max(cfg.horizons)
    samples = ["full"] + [s for s in spans if s.startswith("excluding")]
    fig, axes = plt.subplots(1, len(samples), figsize=(5.0 * len(samples), 4.6), sharey=True)
    axes_list = [axes] if len(samples) == 1 else list(axes)
    for ax, sample in zip(axes_list, samples, strict=True):
        for model in ("ns_ar", "ar_yield", "ns_rw"):
            rows = _ratio_rows(versus, "zero", sample, model, h)
            if rows.empty:
                continue
            ax.plot(
                rows["maturity_months"],
                rows["rmse_ratio"],
                marker="o",
                color=MODEL_COLOURS[model],
                label=F.MODEL_LABELS[model],
            )
        ax.axhline(1.0, color=INK, linewidth=0.9)
        _maturity_axis(ax)
        title = "All targets" if sample == "full" else f"Targets {sample}"
        ax.set_title(title)
    axes_list[0].set_ylabel("Forecast RMSE / random-walk RMSE")
    axes_list[-1].legend(loc="upper left")
    fig.suptitle(
        f"{h}-month horizon: mean-reverting models and the factor random walk",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def fig_loadings(loadings: pd.DataFrame, shares: FloatArray, path: Path) -> None:
    """Loadings of the first three principal components of daily changes."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    names = ["level", "slope", "curvature"]
    for k, key in enumerate(["pc1", "pc2", "pc3"]):
        ax.plot(
            loadings["maturity_months"],
            loadings[f"{key}_one_sd_bp"],
            marker="o",
            color=COLOURS[key],
            label=f"PC{k + 1} ({names[k]}), {100 * shares[k]:.1f}% of variance",
        )
    ax.axhline(0.0, color=INK, linewidth=0.9)
    _maturity_axis(ax)
    ax.set_ylabel("Yield change for a one-s.d. move (basis points)")
    ax.set_title("Principal components of daily par yield changes")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def fig_rolling(rolling: pd.DataFrame, window: int, path: Path) -> None:
    """Variance shares of the first three components through time."""
    dates = pd.to_datetime(rolling["window_end"])
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.stackplot(
        dates,
        rolling["pc1_pct"],
        rolling["pc2_pct"],
        rolling["pc3_pct"],
        colors=[COLOURS["pc1"], COLOURS["pc2"], COLOURS["pc3"]],
        labels=["PC1", "PC2", "PC3"],
        alpha=0.9,
    )
    ax.set_ylim(50, 100)
    ax.set_xlabel(f"End of {window}-trading-day window")
    ax.set_ylabel("Share of variance of daily changes (%)")
    ax.set_title("Variance explained by the first three components, rolling windows")
    ax.legend(loc="lower left", ncols=3)
    fig.tight_layout()
    _save(fig, path)


def fig_factors(betas: pd.DataFrame, monthly: pd.DataFrame, path: Path) -> None:
    """Fitted Nelson-Siegel factors against the empirical level, slope and curvature."""
    emp = empirical_factors(monthly)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    panels = [
        ("beta1", 1.0, "level", "Level: beta1 and the 10-year yield"),
        ("beta2", -1.0, "slope", "Slope: -beta2 and 10-year minus 3-month"),
        ("beta3", 0.3, "curvature", "Curvature: 0.3*beta3 and 2*(2-year) - 3-month - 10-year"),
    ]
    for ax, (name, scale, emp_name, title) in zip(axes, panels, strict=True):
        ax.plot(betas.index, scale * betas[name], color=COLOURS["pc1"], label="Nelson-Siegel")
        ax.plot(
            emp.index,
            emp[emp_name],
            color=COLOURS["pc2"],
            linestyle=":",
            label="empirical measure",
        )
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Percent")
        ax.axhline(0.0, color=INK, linewidth=0.6)
    axes[0].legend()
    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _range(frame: pd.DataFrame, column: str = "rmse_ratio") -> dict[str, object]:
    return {
        "min": round(float(frame[column].min()), 3),
        "max": round(float(frame[column].max()), 3),
        "below_one": int((frame[column] < 1.0).sum()),
        "maturities": len(frame),
    }


def _headline(
    variants: pd.DataFrame,
    by_tenor: pd.DataFrame,
    rolling: pd.DataFrame,
    versus: pd.DataFrame,
    comparison: pd.DataFrame,
    spans: dict[str, dict[str, str]],
) -> dict[str, object]:
    main = variants.iloc[0]
    lo, hi = rolling["pc1_pct"].idxmin(), rolling["pc1_pct"].idxmax()
    out: dict[str, object] = {
        "pca_treatment": str(main["treatment"]),
        "pca_first_date": str(main["first_date"]),
        "pca_last_date": str(main["last_date"]),
        "pca_observations": int(main["observations"]),
        "pc1_pct": round(float(main["pc1_pct"]), 2),
        "pc2_pct": round(float(main["pc2_pct"]), 2),
        "pc3_pct": round(float(main["pc3_pct"]), 2),
        "first_three_pct": round(float(main["first_three_pct"]), 2),
        "min_maturity_explained_pct": round(float(by_tenor["explained_by_three_pct"].min()), 2),
        "rolling_pc1_min_pct": round(float(rolling.loc[lo, "pc1_pct"]), 2),
        "rolling_pc1_min_window_end": str(rolling.loc[lo, "window_end"]),
        "rolling_pc1_max_pct": round(float(rolling.loc[hi, "pc1_pct"]), 2),
        "rolling_pc1_max_window_end": str(rolling.loc[hi, "window_end"]),
        "rolling_first_three_min_pct": round(float(rolling["first_three_pct"].min()), 2),
        "rolling_first_three_max_pct": round(float(rolling["first_three_pct"].max()), 2),
        "forecast_samples": spans,
    }
    if not comparison.empty:
        rep: dict[str, object] = {}
        for (model, h), block in comparison.groupby(["model", "horizon_months"], sort=True):
            entry: dict[str, object] = {
                "max_abs_rmse_gap": round(float(block["rmse_gap"].abs().max()), 3),
                "mean_abs_rmse_gap": round(float(block["rmse_gap"].abs().mean()), 3),
            }
            if model != "rw":
                entry["ratio_here_min"] = round(float(block["ratio_to_rw_here"].min()), 3)
                entry["ratio_here_max"] = round(float(block["ratio_to_rw_here"].max()), 3)
                entry["ratio_paper_min"] = round(float(block["ratio_to_rw_paper"].min()), 3)
                entry["ratio_paper_max"] = round(float(block["ratio_to_rw_paper"].max()), 3)
                entry["here_below_one"] = int((block["ratio_to_rw_here"] < 1.0).sum())
                entry["paper_below_one"] = int((block["ratio_to_rw_paper"] < 1.0).sum())
            rep[f"{model}_h{int(h)}"] = entry
        out["paper_window_at_five_maturities"] = rep
    ratios: dict[str, object] = {}
    for (panel, sample, model, h), block in versus.groupby(
        ["panel", "sample", "model", "horizon_months"], sort=True
    ):
        if model in ("ns_ar", "ns_rw", "ar_yield"):
            ratios[f"{panel}|{sample}|{model}|h{int(h)}"] = _range(block)
    out["rmse_ratio_to_random_walk"] = ratios
    return out


def run(
    daily: pd.DataFrame,
    zero: pd.DataFrame,
    out_dir: Path,
    *,
    wide: pd.DataFrame | None = None,
    fast: bool = False,
    cfg: Config | None = None,
    vintage: dict[str, object] | None = None,
) -> dict[str, pd.DataFrame]:
    """Write every table and figure into ``out_dir`` and return the tables.

    Parameters
    ----------
    daily:
        Daily par yields in percent, indexed by date, one column per maturity
        label from ``tenor.data.TENOR_MONTHS``. Must hold the core maturities.
        Used for the principal components and a second set of forecasts.
    zero:
        Daily zero-coupon yields in percent, indexed by date, columns the
        maturities in months. Used for the main forecasts.
    out_dir:
        Destination directory, created if missing.
    wide:
        Optional second par panel that also holds the 20- and 30-year points.
    fast:
        Use a coarser step for the rolling principal-components windows and
        fewer size simulations. Everything else is the same.
    cfg:
        Settings; the defaults are the ones used for the committed results.
    vintage:
        Extra fields for ``data_vintage.json``.

    """
    cfg = cfg or Config()
    out_dir.mkdir(parents=True, exist_ok=True)
    step = cfg.rolling_step * 12 if fast else cfg.rolling_step
    monthly = D.month_end(daily)
    zero_monthly = D.month_end(zero.loc[f"{cfg.zero_start}-01" :])
    zero_tau = [float(c) for c in zero_monthly.columns]
    par_tau = D.tenor_months(monthly.columns)

    variants = pca_variants(daily, wide)
    loadings, by_tenor, res = pca_detail(daily)
    rolling = pca_rolling(daily, cfg.rolling_window, step)
    bills = bill_volatility(daily, cfg)
    betas, residuals, factor_stats, correlations = ns_tables(monthly)
    zero_residuals = ns_residuals_zero(zero_monthly, zero_tau)

    zero_first = _month_index(zero_monthly, cfg.zero_first_target, "left")
    if zero_first is None:
        raise ValueError("the zero-coupon panel ends before the first forecast target")
    zero_rows, zero_spans = forecast_tables(
        "zero", zero_monthly, zero_tau, zero_first, cfg.zero_subperiods, cfg
    )
    par_rows, par_spans = forecast_tables(
        "par", monthly, par_tau, cfg.initial_months, cfg.par_subperiods, cfg
    )
    roll_rows, _ = forecast_tables(
        "zero",
        zero_monthly,
        zero_tau,
        zero_first,
        cfg.zero_subperiods,
        cfg,
        window=cfg.rolling_estimation_months,
    )
    errors = pd.DataFrame(zero_rows.errors + par_rows.errors)
    versus = pd.DataFrame(zero_rows.versus + par_rows.versus)
    nested = pd.DataFrame(zero_rows.nested + par_rows.nested)
    rolling_scheme = pd.DataFrame(roll_rows.versus)
    comparison = paper_comparison(errors, versus, cfg.zero_subperiods[0][0])
    sizes = S.size_table(horizons=cfg.horizons, reps=500 if fast else cfg.size_simulations)

    tables: dict[str, pd.DataFrame] = {
        "pca_variance": variants,
        "pca_loadings": loadings,
        "pca_by_maturity": by_tenor,
        "pca_rolling": rolling,
        "pca_bill_volatility": bills,
        "ns_fit_residuals": residuals,
        "ns_fit_residuals_zero": zero_residuals,
        "ns_factor_stats": factor_stats,
        "ns_factor_correlations": correlations,
        "forecast_errors": errors,
        "forecast_vs_random_walk": versus,
        "forecast_nested_tests": nested,
        "forecast_rolling_scheme": rolling_scheme,
        "test_size_simulation": sizes,
    }
    if not comparison.empty:
        tables["forecast_paper_window"] = comparison
    for name, frame in tables.items():
        _write_csv(frame, out_dir / f"{name}.csv")
    factors_out = betas.round(4).reset_index()
    factors_out["Date"] = factors_out["Date"].dt.strftime("%Y-%m-%d")
    factors_out.to_csv(out_dir / "ns_factors.csv", index=False, lineterminator="\n")

    with plt.rc_context(STYLE):
        fig_rmse_ratio(versus, zero_spans, cfg, out_dir / "fig_forecast_rmse_ratio.png")
        fig_subperiods(versus, cfg, out_dir / "fig_forecast_subperiods.png")
        fig_variants(versus, zero_spans, cfg, out_dir / "fig_forecast_variants.png")
        fig_loadings(loadings, res.explained, out_dir / "fig_pca_loadings.png")
        fig_rolling(
            rolling, min(cfg.rolling_window, len(daily) - 1), out_dir / "fig_pca_rolling.png"
        )
        fig_factors(betas, monthly, out_dir / "fig_ns_factors.png")

    spans = {"zero": zero_spans, "par": par_spans}
    headline = _headline(variants, by_tenor, rolling, versus, comparison, spans)
    (out_dir / "headline.json").write_text(
        json.dumps(headline, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    record: dict[str, object] = {
        "par_panel": {
            "first_date": daily.index[0].strftime("%Y-%m-%d"),
            "last_date": daily.index[-1].strftime("%Y-%m-%d"),
            "trading_days": len(daily),
            "months": len(monthly),
            "maturities": list(daily.columns),
            "days_dropped_for_missing_values": int(daily.attrs.get("dropped_rows", 0)),
            "panel_sha256": D.panel_sha256(daily),
        },
        "zero_panel": {
            "first_date": zero.index[0].strftime("%Y-%m-%d"),
            "last_date": zero.index[-1].strftime("%Y-%m-%d"),
            "trading_days": len(zero),
            "months_used": len(zero_monthly),
            "first_month_used": str(zero_monthly.index[0].to_period("M")),
            "maturities_months": [int(c) for c in zero.columns],
            "days_dropped_for_missing_values": int(zero.attrs.get("dropped_rows", 0)),
            "panel_sha256": D.panel_sha256(zero),
        },
        "forecast_samples": spans,
        "fast": fast,
    }
    if vintage:
        record.update(vintage)
    (out_dir / "data_vintage.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return tables

"""Build every table and figure from a daily par yield panel.

``run`` is the single entry point. It is deterministic: the same panel gives
byte-identical tables. ``scripts/reproduce.py`` calls it on the Treasury data;
the tests and the CI smoke run call it on a simulated panel.
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

FloatArray = npt.NDArray[np.float64]

COLOURS = {"pc1": "#1f4e79", "pc2": "#c0504d", "pc3": "#6b8e23", "grey": "#7f7f7f"}
HORIZON_COLOURS = {1: "#9ecae1", 6: "#4292c6", 12: "#08306b"}


@dataclass(frozen=True)
class Config:
    """Settings fixed before any result was looked at.

    Attributes
    ----------
    horizons:
        Forecast horizons in months.
    initial_months:
        Months of data before the first forecast target of the main sample.
        108 is the length of the paper's first estimation window, 1985:01 to
        1993:12.
    rolling_window:
        Trading days per rolling principal-components window (about 5 years).
    rolling_step:
        Trading days between rolling windows.
    paper_window:
        First and last forecast target of the paper's own evaluation period.
    subperiods:
        Labels and inclusive target-month ranges used to split the main sample.

    """

    horizons: tuple[int, ...] = (1, 6, 12)
    initial_months: int = 108
    rolling_window: int = 1250
    rolling_step: int = 21
    paper_window: tuple[str, str] = ("1994-01", "2000-12")
    subperiods: tuple[tuple[str, str, str], ...] = (
        ("1999-2008", "1999-01", "2008-12"),
        ("2009-2017", "2009-01", "2017-12"),
        ("2018-2026", "2018-01", "2026-12"),
    )


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


# --------------------------------------------------------------------------
# forecasts
# --------------------------------------------------------------------------


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


def _score(
    fs: F.ForecastSet, mask: npt.NDArray[np.bool_], tau: list[float], sample: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Error statistics and random-walk comparisons for the masked forecasts."""
    h = fs.horizon
    lag_a, lag_b = _error_lags(h)
    err_rows: list[dict[str, object]] = []
    cmp_rows: list[dict[str, object]] = []
    rw = fs.errors("rw")[mask]
    for name in F.MODELS:
        err = fs.errors(name)[mask]
        for j, t in enumerate(tau):
            s = E.error_stats(err[:, j])
            err_rows.append(
                {
                    "sample": sample,
                    "horizon_months": h,
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
            bart = E.diebold_mariano(err[:, j], rw[:, j], h=h, window="bartlett")
            cmp_rows.append(
                {
                    "sample": sample,
                    "horizon_months": h,
                    "model": name,
                    "maturity_months": t,
                    "forecasts": s.n,
                    "rmse": s.rmse,
                    "rmse_random_walk": rw_rmse,
                    "rmse_ratio": s.rmse / rw_rmse,
                    "dm_uniform": uni.stat,
                    "dm_uniform_p": uni.pvalue,
                    "dm_bartlett": bart.stat,
                    "dm_bartlett_p": bart.pvalue,
                }
            )
    return err_rows, cmp_rows


def forecast_tables(
    monthly: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Run every forecast sample and score it against the random walk."""
    tau = D.tenor_months(monthly.columns)
    y = monthly.to_numpy()
    periods = monthly.index.to_period("M")
    err_rows: list[dict[str, object]] = []
    cmp_rows: list[dict[str, object]] = []
    spans: dict[str, str] = {}

    first_target = cfg.initial_months
    if first_target < len(monthly):
        spans["full"] = f"{periods[first_target]} to {periods[-1]}"
        for h in cfg.horizons:
            fs = F.recursive_forecasts(y, tau, h, first_target=first_target)
            everything = np.ones(fs.targets.size, dtype=np.bool_)
            e, c = _score(fs, everything, tau, "full")
            err_rows += e
            cmp_rows += c
            target_periods = periods[fs.targets]
            for label, lo, hi in cfg.subperiods:
                mask = np.asarray(
                    (target_periods >= pd.Period(lo, freq="M"))
                    & (target_periods <= pd.Period(hi, freq="M"))
                )
                if mask.sum() < 24:
                    continue
                spans[label] = f"{target_periods[mask][0]} to {target_periods[mask][-1]}"
                e, c = _score(fs, mask, tau, label)
                err_rows += e
                cmp_rows += c

    lo_idx = _month_index(monthly, cfg.paper_window[0], "left")
    hi_idx = _month_index(monthly, cfg.paper_window[1], "right")
    max_h = max(cfg.horizons)
    if lo_idx is not None and hi_idx is not None and hi_idx > lo_idx and lo_idx - max_h >= 16:
        spans["paper window"] = f"{periods[lo_idx]} to {periods[hi_idx]}"
        for h in cfg.horizons:
            fs = F.recursive_forecasts(y, tau, h, first_target=lo_idx, last_target=hi_idx)
            everything = np.ones(fs.targets.size, dtype=np.bool_)
            e, c = _score(fs, everything, tau, "paper window")
            err_rows += e
            cmp_rows += c
    return pd.DataFrame(err_rows), pd.DataFrame(cmp_rows), spans


def paper_comparison(errors: pd.DataFrame, versus: pd.DataFrame) -> pd.DataFrame:
    """Set this repository's paper-window numbers beside the printed ones."""
    rows = []
    sub_e = errors[errors["sample"] == "paper window"]
    sub_v = versus[versus["sample"] == "paper window"]
    if sub_e.empty:
        return pd.DataFrame()
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
                row: dict[str, object] = {
                    "horizon_months": h,
                    "model": model,
                    "maturity_months": float(tau),
                    "rmse_here": float(here["rmse"].iloc[0]),
                    "rmse_paper": paper_rmse,
                }
                paper_rw = published.DL_RMSE["rw"][h][published.DL_MATURITIES.index(tau)]
                row["ratio_to_rw_paper"] = paper_rmse / paper_rw
                dm_here = math.nan
                ratio_here = 1.0
                if model != "rw":
                    v = sub_v[
                        (sub_v["model"] == model)
                        & (sub_v["horizon_months"] == h)
                        & (sub_v["maturity_months"] == float(tau))
                    ]
                    ratio_here = float(v["rmse_ratio"].iloc[0])
                    dm_here = float(v["dm_uniform"].iloc[0])
                row["ratio_to_rw_here"] = ratio_here
                row["dm_vs_rw_here"] = dm_here
                dm_paper = math.nan
                if model == "ns_ar" and h in published.DL_DM_VS_RW:
                    dm_paper = published.DL_DM_VS_RW[h][published.DL_MATURITIES.index(tau)]
                row["dm_vs_rw_paper"] = dm_paper
                rows.append(row)
    cols = [
        "horizon_months",
        "model",
        "maturity_months",
        "rmse_here",
        "rmse_paper",
        "ratio_to_rw_here",
        "ratio_to_rw_paper",
        "dm_vs_rw_here",
        "dm_vs_rw_paper",
    ]
    return pd.DataFrame(rows)[cols].sort_values(
        ["horizon_months", "model", "maturity_months"], kind="stable"
    )


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def fig_rmse_ratio(versus: pd.DataFrame, spans: dict[str, str], path: Path) -> None:
    """RMSE of the Nelson-Siegel AR forecast over the random walk's, by maturity."""
    sub = versus[(versus["sample"] == "full") & (versus["model"] == "ns_ar")]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels: list[str] = []
    for h in sorted(sub["horizon_months"].unique()):
        rows = sub[sub["horizon_months"] == h].sort_values("maturity_months")
        labels = [_label_years(t) for t in rows["maturity_months"]]
        ax.plot(
            range(len(rows)),
            rows["rmse_ratio"],
            marker="o",
            color=HORIZON_COLOURS.get(int(h), "black"),
            label=f"{int(h)}-month horizon",
        )
    if labels:
        paper = published.DL_RMSE
        positions = [
            labels.index(_label_years(t))
            for t in published.DL_MATURITIES
            if _label_years(t) in labels
        ]
        if len(positions) == len(published.DL_MATURITIES):
            ratio = [a / b for a, b in zip(paper["ns_ar"][12], paper["rw"][12], strict=True)]
            ax.plot(
                positions,
                ratio,
                marker="s",
                linestyle="--",
                color=COLOURS["grey"],
                label="12-month, as printed in Diebold-Li (1994-2000)",
            )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Forecast RMSE / random-walk RMSE")
    ax.set_title(
        f"Nelson-Siegel AR(1) forecast against no change, targets {spans.get('full', '')}",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.99,
        0.02,
        "Below 1: the forecast beat the random walk",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=COLOURS["grey"],
    )
    fig.tight_layout()
    _save(fig, path)


def fig_loadings(loadings: pd.DataFrame, shares: FloatArray, path: Path) -> None:
    """Loadings of the first three principal components of daily changes."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = range(len(loadings))
    names = ["level", "slope", "curvature"]
    for k, key in enumerate(["pc1", "pc2", "pc3"]):
        ax.plot(
            x,
            loadings[f"{key}_one_sd_bp"],
            marker="o",
            color=COLOURS[key],
            label=f"PC{k + 1} ({names[k]}), {100 * shares[k]:.1f}% of variance",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_label_years(t) for t in loadings["maturity_months"]])
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield change for a one-s.d. move (basis points)")
    ax.set_title("Principal components of daily par yield changes", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def fig_rolling(rolling: pd.DataFrame, window: int, path: Path) -> None:
    """Variance shares of the first three components through time."""
    dates = pd.to_datetime(rolling["window_end"])
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.stackplot(
        dates,
        rolling["pc1_pct"],
        rolling["pc2_pct"],
        rolling["pc3_pct"],
        colors=[COLOURS["pc1"], COLOURS["pc2"], COLOURS["pc3"]],
        labels=["PC1", "PC2", "PC3"],
        alpha=0.85,
    )
    ax.set_ylim(50, 100)
    ax.set_xlabel(f"End of {window}-trading-day window")
    ax.set_ylabel("Share of variance of daily changes (%)")
    ax.set_title("Variance explained by the first three components, rolling", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncols=3)
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
        ax.plot(emp.index, emp[emp_name], color=COLOURS["pc2"], linestyle=":", label="empirical")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Percent")
        ax.axhline(0.0, color="black", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    _save(fig, path)


def fig_subperiods(versus: pd.DataFrame, cfg: Config, path: Path) -> None:
    """12-month RMSE ratio by maturity, one line per sub-period."""
    h = max(cfg.horizons)
    sub = versus[(versus["model"] == "ns_ar") & (versus["horizon_months"] == h)]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    shades = ["#fdae6b", "#e6550d", "#7f2704", "#7f7f7f"]
    samples = [label for label, _, _ in cfg.subperiods] + ["paper window"]
    labels: list[str] = []
    for shade, sample in zip(shades, samples, strict=True):
        rows = sub[sub["sample"] == sample].sort_values("maturity_months")
        if rows.empty:
            continue
        labels = [_label_years(t) for t in rows["maturity_months"]]
        name = "1994-2000 (the paper's window)" if sample == "paper window" else sample
        ax.plot(range(len(rows)), rows["rmse_ratio"], marker="o", color=shade, label=name)
    ax.axhline(1.0, color="black", linewidth=0.8)
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Forecast RMSE / random-walk RMSE")
    ax.set_title(f"Nelson-Siegel AR(1), {h}-month horizon, by period of the targets", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _headline(
    variants: pd.DataFrame, versus: pd.DataFrame, by_tenor: pd.DataFrame, spans: dict[str, str]
) -> dict[str, object]:
    main = variants.iloc[0]
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
        "forecast_targets": spans.get("full", ""),
    }
    full = versus[(versus["sample"] == "full") & (versus["model"] == "ns_ar")]
    for h in sorted(full["horizon_months"].unique()):
        rows = full[full["horizon_months"] == h]
        out[f"h{int(h)}_forecasts"] = int(rows["forecasts"].iloc[0])
        out[f"h{int(h)}_ratio_min"] = round(float(rows["rmse_ratio"].min()), 3)
        out[f"h{int(h)}_ratio_max"] = round(float(rows["rmse_ratio"].max()), 3)
        out[f"h{int(h)}_maturities_beating_rw"] = int((rows["rmse_ratio"] < 1.0).sum())
        out[f"h{int(h)}_maturities"] = len(rows)
        out[f"h{int(h)}_dm_min"] = round(float(rows["dm_uniform"].min()), 2)
        out[f"h{int(h)}_dm_max"] = round(float(rows["dm_uniform"].max()), 2)
    return out


def run(
    daily: pd.DataFrame,
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
    out_dir:
        Destination directory, created if missing.
    wide:
        Optional second panel that also holds the 20- and 30-year points.
    fast:
        Use a coarser step for the rolling windows. Everything else is the same.
    cfg:
        Settings; the defaults are the ones used for the committed results.
    vintage:
        Extra fields for ``data_vintage.json``.

    """
    cfg = cfg or Config()
    out_dir.mkdir(parents=True, exist_ok=True)
    step = cfg.rolling_step * 12 if fast else cfg.rolling_step
    monthly = D.month_end(daily)

    variants = pca_variants(daily, wide)
    loadings, by_tenor, res = pca_detail(daily)
    rolling = pca_rolling(daily, cfg.rolling_window, step)
    betas, residuals, factor_stats, correlations = ns_tables(monthly)
    errors, versus, spans = forecast_tables(monthly, cfg)
    comparison = paper_comparison(errors, versus)

    tables: dict[str, pd.DataFrame] = {
        "pca_variance": variants,
        "pca_loadings": loadings,
        "pca_by_maturity": by_tenor,
        "pca_rolling": rolling,
        "ns_fit_residuals": residuals,
        "ns_factor_stats": factor_stats,
        "ns_factor_correlations": correlations,
        "forecast_errors": errors,
        "forecast_vs_random_walk": versus,
    }
    if not comparison.empty:
        tables["forecast_paper_window"] = comparison
    for name, frame in tables.items():
        _write_csv(frame, out_dir / f"{name}.csv")
    factors_out = betas.round(4).reset_index()
    factors_out["Date"] = factors_out["Date"].dt.strftime("%Y-%m-%d")
    factors_out.to_csv(out_dir / "ns_factors.csv", index=False, lineterminator="\n")

    fig_rmse_ratio(versus, spans, out_dir / "fig_forecast_rmse_ratio.png")
    fig_loadings(loadings, res.explained, out_dir / "fig_pca_loadings.png")
    fig_rolling(rolling, min(cfg.rolling_window, len(daily) - 1), out_dir / "fig_pca_rolling.png")
    fig_factors(betas, monthly, out_dir / "fig_ns_factors.png")
    fig_subperiods(versus, cfg, out_dir / "fig_forecast_subperiods.png")

    headline = _headline(variants, versus, by_tenor, spans)
    (out_dir / "headline.json").write_text(
        json.dumps(headline, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    record: dict[str, object] = {
        "first_date": daily.index[0].strftime("%Y-%m-%d"),
        "last_date": daily.index[-1].strftime("%Y-%m-%d"),
        "trading_days": len(daily),
        "months": len(monthly),
        "maturities": list(daily.columns),
        "days_dropped_for_missing_values": int(daily.attrs.get("dropped_rows", 0)),
        "panel_sha256": D.panel_sha256(daily),
        "forecast_samples": spans,
        "fast": fast,
    }
    if vintage:
        record.update(vintage)
    (out_dir / "data_vintage.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return tables

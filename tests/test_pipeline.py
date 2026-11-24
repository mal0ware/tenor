import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tenor import data as D
from tenor import pipeline, published, synthetic

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TABLES = [
    "pca_variance",
    "pca_loadings",
    "pca_by_maturity",
    "pca_rolling",
    "pca_bill_volatility",
    "ns_fit_residuals",
    "ns_fit_residuals_zero",
    "ns_factor_stats",
    "ns_factor_correlations",
    "ns_factors",
    "forecast_errors",
    "forecast_vs_random_walk",
    "forecast_nested_tests",
    "forecast_rolling_scheme",
    "forecast_paper_window",
    "test_size_simulation",
]
EXPECTED_FIGURES = [
    "fig_forecast_rmse_ratio",
    "fig_forecast_subperiods",
    "fig_forecast_variants",
    "fig_pca_loadings",
    "fig_pca_rolling",
    "fig_ns_factors",
]
KEYS = ["panel", "sample", "horizon_months", "maturity_months"]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synthetic.simulate_panel()


@pytest.fixture(scope="module")
def zero() -> pd.DataFrame:
    return synthetic.simulate_zero_panel()


@pytest.fixture(scope="module")
def run_a(panel, zero, tmp_path_factory):
    out = tmp_path_factory.mktemp("a")
    return out, pipeline.run(panel, zero, out, fast=True)


def test_synthetic_panels_are_seeded(panel, zero):
    pd.testing.assert_frame_equal(panel, synthetic.simulate_panel())
    pd.testing.assert_frame_equal(zero, synthetic.simulate_zero_panel())
    assert not panel.equals(synthetic.simulate_panel(seed=1))
    assert zero.index[0].year == 1985 and zero.shape[1] == 17


def test_outputs_exist(run_a):
    out, _ = run_a
    for name in EXPECTED_TABLES:
        assert (out / f"{name}.csv").stat().st_size > 0, name
    for name in EXPECTED_FIGURES:
        assert (out / f"{name}.png").stat().st_size > 5000, name
    assert json.loads((out / "headline.json").read_text())["first_three_pct"] > 0
    vintage = json.loads((out / "data_vintage.json").read_text())
    assert vintage["par_panel"]["trading_days"] == 4000
    assert vintage["zero_panel"]["trading_days"] == 6000
    assert vintage["zero_panel"]["first_month_used"] == "1985-01"


def test_running_twice_gives_identical_tables(panel, zero, run_a, tmp_path):
    out_a, _ = run_a
    pipeline.run(panel, zero, tmp_path, fast=True)
    for name in EXPECTED_TABLES:
        a = (out_a / f"{name}.csv").read_bytes()
        b = (tmp_path / f"{name}.csv").read_bytes()
        assert a == b, name
    for name in ("headline.json", "data_vintage.json"):
        assert (out_a / name).read_bytes() == (tmp_path / name).read_bytes()


def test_three_factor_panel_is_explained_by_three_components(run_a):
    _, tables = run_a
    variants = tables["pca_variance"].set_index("treatment")
    # daily changes of a three-factor panel with 1 bp noise: nearly all variance
    assert variants.loc["monthly levels, covariance", "first_three_pct"] > 99.99
    assert variants.loc["daily changes, covariance", "first_three_pct"] > 95.0
    shares = tables["pca_variance"][["pc1_pct", "pc2_pct", "pc3_pct"]].to_numpy()
    assert (np.diff(shares, axis=1) <= 1e-9).all()
    by = tables["pca_by_maturity"]
    split = by[
        ["pc1_share_of_explained_pct", "pc2_share_of_explained_pct", "pc3_share_of_explained_pct"]
    ].sum(axis=1)
    np.testing.assert_allclose(split, 100.0, atol=1e-8)


def test_fit_recovers_simulated_curves_to_the_noise_level(run_a):
    _, tables = run_a
    assert tables["ns_fit_residuals"]["rmse"].max() < 0.02
    assert tables["ns_fit_residuals_zero"]["rmse"].max() < 0.02
    assert len(tables["ns_fit_residuals_zero"]) == 17
    corr = tables["ns_factor_correlations"]
    assert (corr["abs_correlation_with_pc"] > 0.5).all()


def test_random_walk_and_models_are_scored_on_the_same_forecasts(run_a):
    _, tables = run_a
    errors = tables["forecast_errors"]
    counts = errors.groupby(["panel", "sample", "horizon_months"])["forecasts"].nunique()
    assert (counts == 1).all()
    versus = tables["forecast_vs_random_walk"]
    merged = versus.merge(errors[errors["model"] == "rw"], on=KEYS, suffixes=("", "_rw"))
    assert len(merged) == len(versus)
    np.testing.assert_array_equal(merged["forecasts"], merged["forecasts_rw"])
    np.testing.assert_allclose(merged["rmse_random_walk"], merged["rmse_rw"])
    np.testing.assert_allclose(merged["rmse_ratio"], merged["rmse"] / merged["rmse_random_walk"])


def test_sample_counts(panel, zero, run_a):
    _, tables = run_a
    errors = tables["forecast_errors"]
    par = errors[(errors["panel"] == "par") & (errors["sample"] == "full")]
    assert set(par["forecasts"]) == {len(D.month_end(panel)) - 108}
    zero_months = len(D.month_end(zero))
    full = errors[(errors["panel"] == "zero") & (errors["sample"] == "full")]
    assert set(full["forecasts"]) == {zero_months - 108}  # 1985-01 to 1993-12 is 108 months
    paper = errors[(errors["panel"] == "zero") & (errors["sample"] == "1994-2000")]
    assert set(paper["forecasts"]) == {84}
    assert set(full["maturity_months"]) == {float(m) for m in zero.columns}


def test_excluding_sample_drops_exactly_the_zero_bound_months(panel, tmp_path):
    # A zero-coupon panel long enough to reach past 2017.
    long_zero = synthetic.simulate_zero_panel(n_days=9200)
    tables = pipeline.run(panel, long_zero, tmp_path, fast=True)
    errors = tables["forecast_errors"]
    z = errors[(errors["panel"] == "zero") & (errors["model"] == "rw")]
    n_full = int(z[z["sample"] == "full"]["forecasts"].iloc[0])
    n_excl = int(z[z["sample"] == "excluding 2009-2017"]["forecasts"].iloc[0])
    n_zb = int(z[z["sample"] == "2009-2017"]["forecasts"].iloc[0])
    assert n_zb == 108
    assert n_excl == n_full - n_zb
    # mean squared errors add up across the split
    for h in (1, 12):
        rows = z[(z["horizon_months"] == h) & (z["maturity_months"] == 60.0)].set_index("sample")
        total = rows.loc["full", "rmse"] ** 2 * n_full
        parts = rows.loc["excluding 2009-2017", "rmse"] ** 2 * n_excl
        parts += rows.loc["2009-2017", "rmse"] ** 2 * n_zb
        assert total == pytest.approx(parts, rel=1e-3)


def test_nested_tests_cover_the_nested_pairs_only(run_a):
    _, tables = run_a
    nested = tables["forecast_nested_tests"]
    pairs = set(zip(nested["small_model"], nested["large_model"], strict=True))
    assert pairs == set(pipeline.NESTED_PAIRS)
    errors = tables["forecast_errors"]
    merged = nested.merge(
        errors.rename(columns={"model": "small_model", "rmse": "rmse_check"}),
        on=[*KEYS, "small_model"],
    )
    assert len(merged) == len(nested)
    np.testing.assert_allclose(merged["rmse_small"], merged["rmse_check"], atol=1e-9)
    assert nested["cw_p_one_sided"].between(0, 1).all()
    versus = tables["forecast_vs_random_walk"]
    flagged = set(versus[versus["nests_random_walk"]]["model"])
    assert flagged == {"ar_yield"}


def test_mean_reversion_helps_on_a_mean_reverting_panel(run_a):
    # The simulated factors revert to fixed means, so at 12 months the AR(1)
    # factor model should beat the factor random walk, and Clark-West should
    # see it at most maturities.
    _, tables = run_a
    nested = tables["forecast_nested_tests"]
    rows = nested[
        (nested["panel"] == "zero")
        & (nested["sample"] == "full")
        & (nested["horizon_months"] == 12)
        & (nested["large_model"] == "ns_ar")
    ]
    assert (rows["cw_stat"] > 0).mean() > 0.5


def test_rolling_scheme_is_scored_on_the_same_targets(run_a):
    _, tables = run_a
    roll = tables["forecast_rolling_scheme"]
    versus = tables["forecast_vs_random_walk"]
    base = versus[versus["panel"] == "zero"]
    merged = roll.merge(base, on=[*KEYS, "model"], suffixes=("", "_recursive"))
    assert len(merged) == len(roll) == len(base)
    np.testing.assert_allclose(merged["rmse_random_walk"], merged["rmse_random_walk_recursive"])
    assert not np.allclose(merged["rmse"], merged["rmse_recursive"])


def test_paper_comparison_carries_the_printed_numbers(run_a):
    _, tables = run_a
    comp = tables["forecast_paper_window"]
    row = comp[
        (comp["model"] == "ns_ar") & (comp["horizon_months"] == 12) & (comp["maturity_months"] == 3)
    ].iloc[0]
    assert row["rmse_paper"] == 0.739
    assert row["ratio_to_rw_paper"] == pytest.approx(0.739 / 1.019)
    assert row["dm_vs_rw_paper"] == -1.65
    assert row["forecasts"] == 84
    assert row["rmse_gap"] == pytest.approx(row["rmse_here"] - 0.739)
    assert len(comp) == 5 * (3 + 3 + 3 + 1)


def test_bill_volatility_table(panel, zero, tmp_path):
    cfg = pipeline.Config(crisis_window=("1995-03-01", "1995-06-30"))
    tables = pipeline.run(panel, zero, tmp_path, fast=True, cfg=cfg)
    bills = tables["pca_bill_volatility"].set_index("period")
    changes = D.changes_bp(panel)["3 Mo"]
    assert bills.loc["whole sample", "sd_3m_change_bp"] == pytest.approx(changes.std(ddof=1))
    window = changes.loc["1995-03-01":"1995-06-30"]
    row = bills.loc["1995-03-01 to 1995-06-30"]
    assert row["daily_changes"] == len(window)
    assert row["sd_3m_change_bp"] == pytest.approx(window.std(ddof=1))


def test_published_tables_are_internally_consistent():
    # Litterman and Scheinkman, Table 2: the split across factors sums to 100
    # and the printed averages are the column means.
    rows = np.array(list(published.LS_TABLE2.values()))
    np.testing.assert_allclose(rows[:, 1:].sum(axis=1), 100.0, atol=0.11)
    np.testing.assert_allclose(rows.mean(axis=0), published.LS_TABLE2_AVERAGE, atol=0.06)
    # Diebold and Li: at 12 months the Nelson-Siegel AR model has the lowest
    # RMSE of the models kept here at every maturity, as the text says.
    for j in range(5):
        best = published.DL_RMSE["ns_ar"][12][j]
        for model in ("rw", "ar_yield", "ns_var"):
            assert best < published.DL_RMSE[model][12][j]


def _reproduce(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "reproduce.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_reproduce_script_smoke(tmp_path):
    done = _reproduce("--fast", "--synthetic", "--out", str(tmp_path / "out"))
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "out" / "forecast_vs_random_walk.csv").exists()


def test_reproduce_script_protects_the_committed_results():
    results = str(ROOT / "results")
    assert _reproduce("--synthetic").returncode != 0
    assert _reproduce("--synthetic", "--out", results).returncode != 0
    refused = _reproduce("--fast", "--out", results)
    assert refused.returncode != 0
    assert "results" in refused.stderr


def test_fast_defaults_to_the_uncommitted_directory():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import reproduce
    finally:
        sys.path.pop(0)
    assert reproduce.RESULTS_FAST.name == "results_fast"
    assert reproduce.RESULTS_FAST != reproduce.RESULTS
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "results_fast/" in ignored

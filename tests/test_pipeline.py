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
    "ns_fit_residuals",
    "ns_factor_stats",
    "ns_factor_correlations",
    "ns_factors",
    "forecast_errors",
    "forecast_vs_random_walk",
    "forecast_paper_window",
]
EXPECTED_FIGURES = [
    "fig_forecast_rmse_ratio",
    "fig_pca_loadings",
    "fig_pca_rolling",
    "fig_ns_factors",
    "fig_forecast_subperiods",
]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synthetic.simulate_panel()


@pytest.fixture(scope="module")
def run_a(panel, tmp_path_factory):
    out = tmp_path_factory.mktemp("a")
    return out, pipeline.run(panel, out, fast=True)


def test_synthetic_panel_is_seeded(panel):
    again = synthetic.simulate_panel()
    pd.testing.assert_frame_equal(panel, again)
    other = synthetic.simulate_panel(seed=1)
    assert not panel.equals(other)


def test_outputs_exist(run_a):
    out, _ = run_a
    for name in EXPECTED_TABLES:
        assert (out / f"{name}.csv").stat().st_size > 0, name
    for name in EXPECTED_FIGURES:
        assert (out / f"{name}.png").stat().st_size > 5000, name
    assert json.loads((out / "headline.json").read_text())["first_three_pct"] > 0
    assert json.loads((out / "data_vintage.json").read_text())["trading_days"] == 4000


def test_running_twice_gives_identical_tables(panel, run_a, tmp_path):
    out_a, _ = run_a
    pipeline.run(panel, tmp_path, fast=True)
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
    corr = tables["ns_factor_correlations"]
    assert (corr["abs_correlation_with_pc"] > 0.5).all()


def test_random_walk_and_models_are_scored_on_the_same_forecasts(run_a):
    _, tables = run_a
    errors = tables["forecast_errors"]
    counts = errors.groupby(["sample", "horizon_months"])["forecasts"].nunique()
    assert (counts == 1).all()
    versus = tables["forecast_vs_random_walk"]
    merged = versus.merge(
        errors[errors["model"] == "rw"],
        on=["sample", "horizon_months", "maturity_months"],
        suffixes=("", "_rw"),
    )
    assert len(merged) == len(versus)
    np.testing.assert_array_equal(merged["forecasts"], merged["forecasts_rw"])
    np.testing.assert_allclose(merged["rmse_random_walk"], merged["rmse_rw"])
    np.testing.assert_allclose(merged["rmse_ratio"], merged["rmse"] / merged["rmse_random_walk"])


def test_full_sample_counts(panel, run_a):
    _, tables = run_a
    months = len(D.month_end(panel))
    full = tables["forecast_errors"][tables["forecast_errors"]["sample"] == "full"]
    assert set(full["forecasts"]) == {months - 108}
    paper = tables["forecast_errors"][tables["forecast_errors"]["sample"] == "paper window"]
    assert set(paper["forecasts"]) == {84}


def test_paper_comparison_carries_the_printed_numbers(run_a):
    _, tables = run_a
    comp = tables["forecast_paper_window"]
    row = comp[
        (comp["model"] == "ns_ar") & (comp["horizon_months"] == 12) & (comp["maturity_months"] == 3)
    ].iloc[0]
    assert row["rmse_paper"] == 0.739
    assert row["ratio_to_rw_paper"] == pytest.approx(0.739 / 1.019)
    assert row["dm_vs_rw_paper"] == -1.65


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


def test_reproduce_script_smoke(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "reproduce.py"),
        "--fast",
        "--synthetic",
        "--out",
        str(tmp_path / "out"),
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "out" / "forecast_vs_random_walk.csv").exists()
    refuse = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce.py"), "--synthetic"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refuse.returncode != 0

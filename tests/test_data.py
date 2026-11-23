from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tenor import data as D

OLD = """Date,"3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","30 Yr"
01/31/1990,8.00,8.10,8.11,8.28,8.33,8.35,8.43,8.43,8.46
01/03/1990,7.89,7.94,7.85,7.94,7.96,7.92,8.04,7.99,8.04
01/02/1990,7.83,7.89,7.81,7.87,7.90,7.87,7.98,7.94,8.00
"""

NEW = """Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"
02/03/2026,4.19,4.26,4.28,4.32,4.32,4.38,4.51,4.63,4.65,4.71,4.80,4.92,5.34,5.31
02/02/2026,4.20,4.27,4.29,4.34,4.35,4.42,4.62,4.76,4.81,4.86,4.94,5.04,5.43,5.40
01/30/2026,4.20,N/A,4.29,4.34,4.35,4.42,4.62,,4.81,4.86,4.94,5.04,5.43,5.40
"""


def _write(tmp_path: Path) -> Path:
    (tmp_path / "par_yields_1990.csv").write_text(OLD, encoding="utf-8")
    (tmp_path / "par_yields_2026.csv").write_text(NEW, encoding="utf-8")
    return tmp_path


def test_read_one_file_sorts_dates_and_parses_floats(tmp_path):
    frame = D.read_treasury_csv(_write(tmp_path) / "par_yields_1990.csv")
    assert list(frame.index) == list(pd.to_datetime(["1990-01-02", "1990-01-03", "1990-01-31"]))
    assert frame.loc["1990-01-02", "10 Yr"] == 7.94
    assert frame.dtypes.eq("float64").all()


def test_missing_values_are_nan_and_dropped_from_the_panel(tmp_path):
    full = D.load_all(_write(tmp_path))
    assert np.isnan(full.loc["2026-01-30", "2 Yr"])
    assert np.isnan(full.loc["2026-01-30", "1.5 Month"])
    assert np.isnan(full.loc["1990-01-02", "1 Mo"])
    panel = D.select_panel(full, end=None)
    assert list(panel.columns) == list(D.CORE_TENORS)
    assert len(panel) == 5
    assert panel.attrs["dropped_rows"] == 1
    assert not panel.isna().any().any()


def test_columns_are_ordered_by_maturity(tmp_path):
    full = D.load_all(_write(tmp_path))
    months = D.tenor_months(full.columns)
    assert months == sorted(months)
    assert D.tenor_months(D.CORE_TENORS) == [3, 6, 12, 24, 36, 60, 84, 120]


def test_sample_end_cuts_later_rows(tmp_path):
    full = D.load_all(_write(tmp_path))
    assert len(D.select_panel(full, end="2026-02-02")) == 4
    assert len(D.select_panel(full, start="1990-01-03", end="1990-12-31")) == 2


def test_month_end_takes_the_last_available_day(tmp_path):
    full = D.load_all(_write(tmp_path))
    panel = D.select_panel(full, end=None)
    monthly = D.month_end(panel)
    assert list(monthly.index) == list(pd.to_datetime(["1990-01-31", "2026-02-03"]))


def test_week_end_takes_the_last_available_day():
    idx = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-11"])
    panel = pd.DataFrame({"3 Mo": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    weekly = D.week_end(panel)
    assert list(weekly["3 Mo"]) == [3.0, 5.0]


def test_changes_are_in_basis_points():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    panel = pd.DataFrame({"3 Mo": [5.00, 5.05, 4.95], "10 Yr": [4.00, 4.00, 4.10]}, index=idx)
    ch = D.changes_bp(panel)
    assert len(ch) == 2
    np.testing.assert_allclose(ch["3 Mo"], [5.0, -10.0])
    np.testing.assert_allclose(ch["10 Yr"], [0.0, 10.0])


def test_fingerprint_changes_with_the_data():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    a = pd.DataFrame({"3 Mo": [5.00, 5.05]}, index=idx)
    b = a.copy()
    b.iloc[1, 0] = 5.06
    assert D.panel_sha256(a) == D.panel_sha256(a.copy())
    assert D.panel_sha256(a) != D.panel_sha256(b)


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.load_all(tmp_path)
    bad = tmp_path / "par_yields_2000.csv"
    bad.write_text('Date,"9 Yr"\n01/03/2000,5.0\n', encoding="utf-8")
    with pytest.raises(ValueError):
        D.read_treasury_csv(bad)
    bad.write_text('Day,"1 Yr"\n01/03/2000,5.0\n', encoding="utf-8")
    with pytest.raises(ValueError):
        D.read_treasury_csv(bad)

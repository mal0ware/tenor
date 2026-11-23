"""Reading the US Treasury's daily par yield curve files.

The Treasury publishes one CSV per calendar year. The first column is the
date (month/day/year) and the others are par yields in percent, on a
bond-equivalent basis, one column per constant maturity. The set of columns
changes over the years as maturities were added or suspended.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

#: Column headers used by the Treasury, mapped to maturity in months.
TENOR_MONTHS: dict[str, float] = {
    "1 Mo": 1.0,
    "1.5 Month": 1.5,
    "2 Mo": 2.0,
    "3 Mo": 3.0,
    "4 Mo": 4.0,
    "6 Mo": 6.0,
    "1 Yr": 12.0,
    "2 Yr": 24.0,
    "3 Yr": 36.0,
    "5 Yr": 60.0,
    "7 Yr": 84.0,
    "10 Yr": 120.0,
    "20 Yr": 240.0,
    "30 Yr": 360.0,
}

#: The eight maturities quoted on every business day since January 1990.
CORE_TENORS: tuple[str, ...] = ("3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr")

#: The core set plus the 20- and 30-year points, complete from 9 February 2006.
WIDE_TENORS: tuple[str, ...] = (*CORE_TENORS, "20 Yr", "30 Yr")

FIRST_YEAR: int = 1990

#: Last date used in the committed results. Later rows are ignored so that
#: rerunning on a newer download gives the same tables.
SAMPLE_END: str = "2026-10-31"

#: First date on which every maturity in WIDE_TENORS is quoted again, after
#: the 30-year bond returned.
WIDE_START: str = "2006-02-09"


def read_treasury_csv(path: Path) -> pd.DataFrame:
    """Read one yearly file into a frame indexed by date, columns as published."""
    frame = pd.read_csv(path, na_values=["N/A", ""])
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Date" not in frame.columns:
        raise ValueError(f"{path} has no Date column")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    unknown = [c for c in frame.columns if c != "Date" and c not in TENOR_MONTHS]
    if unknown:
        raise ValueError(f"{path} has unrecognised columns: {unknown}")
    return frame.set_index("Date").sort_index().astype("float64")


def load_all(data_dir: Path) -> pd.DataFrame:
    """Concatenate every yearly file in ``data_dir``, sorted by date."""
    files = sorted(data_dir.glob("par_yields_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no par_yields_*.csv files in {data_dir}; run scripts/fetch_data.py first"
        )
    frames = [read_treasury_csv(f) for f in files]
    full = pd.concat(frames, axis=0, sort=False).sort_index()
    if full.index.has_duplicates:
        raise ValueError("duplicate dates across yearly files")
    ordered = [c for c in TENOR_MONTHS if c in full.columns]
    return full[ordered]


def select_panel(
    full: pd.DataFrame,
    tenors: tuple[str, ...] = CORE_TENORS,
    start: str | None = None,
    end: str | None = SAMPLE_END,
) -> pd.DataFrame:
    """Rows between ``start`` and ``end`` with every chosen maturity present.

    Days on which any chosen maturity is missing are dropped; the count of
    dropped days is stored in ``frame.attrs["dropped_rows"]``.
    """
    panel = full.loc[start:end, list(tenors)]
    complete = panel.dropna(how="any")
    complete.attrs["dropped_rows"] = len(panel) - len(complete)
    return complete


def tenor_months(columns: list[str] | tuple[str, ...] | pd.Index) -> list[float]:
    """Maturity in months for each column label."""
    return [TENOR_MONTHS[str(c)] for c in columns]


def month_end(panel: pd.DataFrame) -> pd.DataFrame:
    """Last available observation of each calendar month.

    The index keeps the actual date of that observation.
    """
    keys = panel.index.to_period("M")
    last_dates = panel.index.to_series().groupby(keys).max()
    return panel.loc[last_dates.to_numpy()]


def week_end(panel: pd.DataFrame) -> pd.DataFrame:
    """Last available observation of each calendar week (Monday to Sunday)."""
    keys = panel.index.to_period("W")
    last_dates = panel.index.to_series().groupby(keys).max()
    return panel.loc[last_dates.to_numpy()]


def changes_bp(panel: pd.DataFrame) -> pd.DataFrame:
    """First differences between consecutive rows, in basis points."""
    return (panel.diff().iloc[1:]) * 100.0


def panel_sha256(panel: pd.DataFrame) -> str:
    """Fingerprint of a panel's dates and values, for the data vintage record."""
    text = panel.to_csv(float_format="%.4f", lineterminator="\n", date_format="%Y-%m-%d")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

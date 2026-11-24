"""Zero-coupon yields from the Gurkaynak-Sack-Wright curve.

Gurkaynak, Sack and Wright (2007) fit a Svensson curve to off-the-run Treasury
notes and bonds on every business day from June 1961. The Federal Reserve
publishes the six fitted parameters per day in ``feds200628.csv``. With
maturity ``n`` in years, the continuously compounded zero-coupon yield is
(their equation 9)

    y(n) = b0 + b1 * (1 - exp(-n/t1)) / (n/t1)
              + b2 * ((1 - exp(-n/t1)) / (n/t1) - exp(-n/t1))
              + b3 * ((1 - exp(-n/t2)) / (n/t2) - exp(-n/t2))

Before 1980 the file sets b3 = 0 and the curve is the three-factor
Nelson-Siegel form. Because the parameters are published, the yield at any
maturity can be computed, including the seventeen maturities Diebold and Li
(2006) use. The file itself only tabulates whole years.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

#: The seventeen maturities, in months, of Diebold and Li (2006), section 3.1.
DL_PANEL_MONTHS: tuple[int, ...] = (
    *(3, 6, 9, 12, 15, 18, 21, 24, 30),
    *(36, 48, 60, 72, 84, 96, 108, 120),
)

PARAMETER_COLUMNS: tuple[str, ...] = ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")

FILE_NAME: str = "feds200628.csv"

#: First date of the panel built for the committed results: the first month
#: of Diebold and Li's sample.
SAMPLE_START: str = "1985-01-01"


def _hump(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Slope and curvature loadings at x = n / tau, with x > 0."""
    slope: FloatArray = -np.expm1(-x) / x
    curvature: FloatArray = slope - np.exp(-x)
    return slope, curvature


def svensson_yield(params: npt.ArrayLike, years: npt.ArrayLike) -> FloatArray:
    """Zero-coupon yields from Svensson parameters.

    Parameters
    ----------
    params:
        Shape (T, 6): b0, b1, b2, b3, tau1, tau2 per row. A row with b3 = 0
        ignores tau2, whatever its value.
    years:
        The n maturities, in years, all positive.

    Returns
    -------
    Yields in the units of the betas (percent in the Fed's file). Shape (T, n).

    """
    p = np.atleast_2d(np.asarray(params, dtype=np.float64))
    n = np.atleast_1d(np.asarray(years, dtype=np.float64))
    if p.shape[1] != 6:
        raise ValueError("params must have six columns")
    if (n <= 0.0).any():
        raise ValueError("maturities must be positive")
    b0, b1, b2, b3, t1, t2 = (p[:, [k]] for k in range(6))
    if (t1 <= 0.0).any():
        raise ValueError("tau1 must be positive")
    s1, c1 = _hump(n[None, :] / t1)
    out: FloatArray = b0 + b1 * s1 + b2 * c1
    second = (b3 != 0.0).ravel()
    if second.any():
        if (t2[second] <= 0.0).any():
            raise ValueError("tau2 must be positive where b3 is non-zero")
        _, c2 = _hump(n[None, :] / t2[second])
        out[second] += b3[second] * c2
    return out


def read_gsw_csv(path: Path) -> pd.DataFrame:
    """Read the Fed's file into a frame indexed by date, columns as published.

    The file starts with a note and a legend; the table begins at the line
    whose first field is ``Date``.
    """
    text = path.read_text(encoding="utf-8-sig")
    start = text.find("\nDate,")
    if start < 0:
        if not text.startswith("Date,"):
            raise ValueError(f"{path} has no header line starting with Date")
        start = -1
    frame = pd.read_csv(io.StringIO(text[start + 1 :]), na_values=["NA", ""])
    missing = [c for c in PARAMETER_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d")
    return frame.set_index("Date").sort_index().astype("float64")


def zero_panel(
    frame: pd.DataFrame,
    months: tuple[int, ...] = DL_PANEL_MONTHS,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Daily zero-coupon yields at ``months``, computed from the parameters.

    Columns are the maturities in months. Days without a full parameter set
    are dropped; the count is stored in ``frame.attrs["dropped_rows"]``.
    Values are rounded to four decimals, the precision of the Fed's own
    tabulated yields.
    """
    params = frame.loc[start:end, list(PARAMETER_COLUMNS)]
    complete = params.dropna(how="any")
    yields = svensson_yield(complete.to_numpy(), np.asarray(months, dtype=np.float64) / 12.0)
    out = pd.DataFrame(np.round(yields, 4), index=complete.index, columns=list(months))
    out.attrs["dropped_rows"] = len(params) - len(complete)
    return out


def max_gap_to_published(frame: pd.DataFrame, panel: pd.DataFrame) -> float:
    """Largest absolute gap between computed and tabulated whole-year yields.

    Compares every whole-year maturity in ``panel`` with the Fed's own SVENYxx
    column on the same dates. A value near 0.0001 (rounding) confirms the
    formula and the parameter columns were read correctly.
    """
    worst = 0.0
    for m in panel.columns:
        name = f"SVENY{int(m) // 12:02d}"
        if int(m) % 12 != 0 or name not in frame.columns:
            continue
        gap = (panel[m] - frame.loc[panel.index, name]).abs().max()
        if np.isfinite(gap):
            worst = max(worst, float(gap))
    return worst


def panel_sha256(panel: pd.DataFrame) -> str:
    """Fingerprint of a panel's dates and values."""
    text = panel.to_csv(float_format="%.4f", lineterminator="\n", date_format="%Y-%m-%d")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

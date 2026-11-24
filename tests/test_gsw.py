from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tenor import gsw
from tenor import nelson_siegel as ns

HEADER = (
    '"Note: This is not an official Federal Reserve statistical release."\n'
    "\n"
    "Series,Compounding Convention,Mnemonic(s)\n"
    "Zero-coupon yield,Continuously Compounded,SVENYXX\n"
    "\n"
    "Date,BETA0,BETA1,BETA2,BETA3,SVENY01,SVENY02,TAU1,TAU2\n"
)


def _row(date: str, p: tuple[float, ...], y1: str, y2: str) -> str:
    b0, b1, b2, b3, t1, t2 = p
    return f"{date},{b0},{b1},{b2},{b3},{y1},{y2},{t1},{t2}\n"


P_A = (5.0, -1.5, 0.8, 2.0, 1.4, 9.0)
P_B = (4.0, -1.0, -2.0, 0.0, 0.9, -999.99)  # three-factor row, tau2 unused


def _file(tmp_path: Path) -> Path:
    ya = gsw.svensson_yield(P_A, [1.0, 2.0])[0]
    yb = gsw.svensson_yield(P_B, [1.0, 2.0])[0]
    text = (
        HEADER
        + _row("1985-01-03", P_B, f"{yb[0]:.4f}", f"{yb[1]:.4f}")
        + _row("1985-01-02", P_A, f"{ya[0]:.4f}", f"{ya[1]:.4f}")
        + "1985-01-04,NA,NA,NA,NA,NA,NA,NA,NA\n"
    )
    path = tmp_path / gsw.FILE_NAME
    path.write_text(text, encoding="utf-8")
    return path


def test_svensson_by_hand_at_one_maturity():
    # n = 2 years, tau1 = 1, tau2 = 4: x1 = 2, x2 = 0.5.
    p = (3.0, 1.0, 2.0, -1.0, 1.0, 4.0)
    s1 = (1 - np.exp(-2.0)) / 2.0
    s2 = (1 - np.exp(-0.5)) / 0.5
    expected = 3.0 + 1.0 * s1 + 2.0 * (s1 - np.exp(-2.0)) - 1.0 * (s2 - np.exp(-0.5))
    assert gsw.svensson_yield(p, [2.0])[0, 0] == pytest.approx(expected, rel=1e-12)


def test_svensson_limits():
    p = (6.0, -2.0, 1.0, 0.5, 1.5, 8.0)
    assert gsw.svensson_yield(p, [1e-9])[0, 0] == pytest.approx(6.0 - 2.0, abs=1e-6)
    assert gsw.svensson_yield(p, [1e7])[0, 0] == pytest.approx(6.0, abs=1e-5)


def test_svensson_without_second_hump_is_nelson_siegel():
    # With b3 = 0 the curve is Nelson-Siegel with lam = 1 / tau1, whatever tau2 holds.
    years = np.array([0.25, 1.0, 3.0, 10.0])
    out = gsw.svensson_yield(P_B, years)[0]
    expected = ns.ns_yield(np.array(P_B[:3]), years, lam=1.0 / P_B[4])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_yield_is_the_average_of_the_forward_curve():
    # Gurkaynak, Sack and Wright, equation 8, gives the Svensson forward rate.
    b0, b1, b2, b3, t1, t2 = P_A
    n = 7.0
    u = np.linspace(0.0, n, 200001)
    forward = (
        b0
        + b1 * np.exp(-u / t1)
        + b2 * (u / t1) * np.exp(-u / t1)
        + b3 * (u / t2) * np.exp(-u / t2)
    )
    average = float(np.sum((forward[1:] + forward[:-1]) / 2.0) * (u[1] - u[0]) / n)
    assert gsw.svensson_yield(P_A, [n])[0, 0] == pytest.approx(average, abs=1e-8)


def test_read_and_build_panel(tmp_path):
    frame = gsw.read_gsw_csv(_file(tmp_path))
    assert list(frame.index) == list(pd.to_datetime(["1985-01-02", "1985-01-03", "1985-01-04"]))
    panel = gsw.zero_panel(frame, months=(3, 12, 24))
    assert list(panel.columns) == [3, 12, 24]
    assert len(panel) == 2
    assert panel.attrs["dropped_rows"] == 1
    assert panel.loc["1985-01-02", 12] == pytest.approx(
        gsw.svensson_yield(P_A, [1.0])[0, 0], abs=5e-5
    )
    assert gsw.max_gap_to_published(frame, panel) < 1.1e-4
    assert len(gsw.zero_panel(frame, months=(12,), start="1985-01-03")) == 1
    assert len(gsw.panel_sha256(panel)) == 64


def test_gap_check_detects_a_wrong_column(tmp_path):
    frame = gsw.read_gsw_csv(_file(tmp_path))
    panel = gsw.zero_panel(frame, months=(12, 24))
    frame["SVENY02"] += 0.5
    assert gsw.max_gap_to_published(frame, panel) == pytest.approx(0.5, abs=1e-3)


def test_validation(tmp_path):
    with pytest.raises(ValueError):
        gsw.svensson_yield((1.0, 1.0, 1.0), [1.0])
    with pytest.raises(ValueError):
        gsw.svensson_yield(P_A, [0.0])
    with pytest.raises(ValueError):
        gsw.svensson_yield((1.0, 1.0, 1.0, 1.0, 1.0, -1.0), [1.0])
    bad = tmp_path / "bad.csv"
    bad.write_text("nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        gsw.read_gsw_csv(bad)


def test_paper_maturities():
    assert len(gsw.DL_PANEL_MONTHS) == 17
    assert gsw.DL_PANEL_MONTHS[0] == 3 and gsw.DL_PANEL_MONTHS[-1] == 120
    assert {3, 12, 36, 60, 120} <= set(gsw.DL_PANEL_MONTHS)

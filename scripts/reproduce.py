"""Rebuild every table and figure in ``results/``.

Usage:
    uv run python scripts/fetch_data.py
    uv run python scripts/reproduce.py              # full run, writes results/
    uv run python scripts/reproduce.py --fast       # coarse run, writes results_fast/
    uv run python scripts/reproduce.py --synthetic --out <dir>
                                                    # simulated panels, no network

``results/`` holds the committed tables, so only a full run on the real data
may write there. ``--fast`` defaults to ``results_fast/``, which is not
committed, and both ``--fast`` and ``--synthetic`` refuse ``--out results``.
``--synthetic`` exists for the CI smoke run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tenor import data as D  # noqa: E402
from tenor import gsw, pipeline, synthetic  # noqa: E402

RESULTS = ROOT / "results"
RESULTS_FAST = ROOT / "results_fast"


def main() -> int:
    """Run the pipeline on the downloaded data or on simulated panels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast", action="store_true", help="coarser rolling windows, fewer simulations"
    )
    parser.add_argument("--synthetic", action="store_true", help="use simulated panels")
    parser.add_argument(
        "--out", type=Path, default=None, help="default: results/, or results_fast/ with --fast"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    started = time.perf_counter()
    out: Path = (args.out or (RESULTS_FAST if args.fast else RESULTS)).resolve()
    if out == RESULTS.resolve():
        if args.synthetic:
            parser.error("--synthetic needs --out pointing somewhere other than results/")
        if args.fast:
            parser.error("--fast may not write into results/; drop --out or point it elsewhere")
    if args.synthetic:
        daily = synthetic.simulate_panel()
        zero = synthetic.simulate_zero_panel()
        pipeline.run(daily, zero, out, fast=args.fast, vintage={"source": "simulated panels"})
    else:
        full = D.load_all(args.data_dir)
        daily = D.select_panel(full, D.CORE_TENORS)
        wide = D.select_panel(full, D.WIDE_TENORS, start=D.WIDE_START)
        gsw_path = args.data_dir / gsw.FILE_NAME
        if not gsw_path.exists():
            parser.error(f"{gsw_path} is missing; run scripts/fetch_data.py first")
        curve = gsw.read_gsw_csv(gsw_path)
        zero = gsw.zero_panel(curve, start=gsw.SAMPLE_START, end=D.SAMPLE_END)
        pipeline.run(
            daily,
            zero,
            out,
            wide=wide,
            fast=args.fast,
            vintage={
                "par_source": "US Treasury daily par yield curve rates",
                "zero_source": "Federal Reserve, Gurkaynak-Sack-Wright curve (feds200628.csv)",
                "zero_max_gap_to_published_whole_year_yields": round(
                    gsw.max_gap_to_published(curve, zero), 6
                ),
                "sample_end": D.SAMPLE_END,
            },
        )
    print(f"wrote {out} in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

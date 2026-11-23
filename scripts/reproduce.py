"""Rebuild every table and figure in ``results/``.

Usage:
    uv run python scripts/fetch_data.py
    uv run python scripts/reproduce.py              # full run, writes results/
    uv run python scripts/reproduce.py --fast       # coarser rolling windows
    uv run python scripts/reproduce.py --synthetic --out <dir>
                                                    # simulated panel, no network

``--synthetic`` exists for the CI smoke run. It refuses to write into
``results/`` so that simulated numbers can never be mistaken for the real ones.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tenor import data as D  # noqa: E402
from tenor import pipeline, synthetic  # noqa: E402


def main() -> int:
    """Run the pipeline on the Treasury data or on a simulated panel."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="coarser rolling windows")
    parser.add_argument("--synthetic", action="store_true", help="use a simulated panel")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    started = time.perf_counter()
    out: Path = args.out.resolve()
    if args.synthetic:
        if out == (ROOT / "results").resolve():
            parser.error("--synthetic needs --out pointing somewhere other than results/")
        daily = synthetic.simulate_panel()
        pipeline.run(daily, out, fast=args.fast, vintage={"source": "simulated panel"})
    else:
        full = D.load_all(args.data_dir)
        daily = D.select_panel(full, D.CORE_TENORS)
        wide = D.select_panel(full, D.WIDE_TENORS, start=D.WIDE_START)
        pipeline.run(
            daily,
            out,
            wide=wide,
            fast=args.fast,
            vintage={
                "source": "US Treasury daily par yield curve rates",
                "sample_end": D.SAMPLE_END,
            },
        )
    print(f"wrote {out} in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

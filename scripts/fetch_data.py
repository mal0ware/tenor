"""Download the two public yield data sets into ``data/``.

1. The US Treasury's daily par yield curve, one CSV per year.

Source: U.S. Department of the Treasury, "Daily Treasury Par Yield Curve
Rates", https://home.treasury.gov/resource-center/data-chart-center/interest-rates
The files are public, need no key and no account. They are written to
``data/`` (never committed) as ``par_yields_<year>.csv`` exactly as served,
plus ``manifest.json`` with the download time and a hash of each file.

2. The Federal Reserve's Gurkaynak-Sack-Wright zero-coupon curve.

Source: Board of Governors of the Federal Reserve System, "Nominal Yield
Curve", https://www.federalreserve.gov/data/nominal-yield-curve.htm, the
data file for Gurkaynak, Sack and Wright (2007), FEDS paper 2006-28. One CSV
of about 17 MB with daily Svensson parameters from June 1961. Public, no key,
no account. Written to ``data/feds200628.csv`` exactly as served.

Usage:
    uv run python scripts/fetch_data.py            # fetch years not yet on disk, refresh the last
    uv run python scripts/fetch_data.py --force    # fetch everything again
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIRST_YEAR = 1990
LAST_YEAR = 2026  # the committed results stop on 2026-10-31
URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)
GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
GSW_FILE = "feds200628.csv"
USER_AGENT = "tenor-fetch/0.2 (python-urllib)"


def fetch_year(year: int, retries: int = 4) -> bytes:
    """Return the raw CSV for one year, retrying on transient errors."""
    request = urllib.request.Request(URL.format(year=year), headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body: bytes = response.read()
            if not body.startswith(b"Date,"):
                raise ValueError(f"unexpected content for {year}: {body[:60]!r}")
            return body
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not download {year}: {last_error}")


def fetch_gsw(retries: int = 4) -> bytes:
    """Return the raw Gurkaynak-Sack-Wright CSV, retrying on transient errors."""
    request = urllib.request.Request(GSW_URL, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body: bytes = response.read()
            if b"\nDate,BETA0," not in body[:4000]:
                raise ValueError(f"unexpected content: {body[:60]!r}")
            return body
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not download {GSW_URL}: {last_error}")


def main() -> int:
    """Download the files and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download every year again")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir / "manifest.json"
    manifest: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        path = args.data_dir / f"par_yields_{year}.csv"
        if path.exists() and not args.force and year != LAST_YEAR:
            continue
        body = fetch_year(year)
        path.write_bytes(body)
        rows = body.count(b"\n") - 1 + (0 if body.endswith(b"\n") else 1)
        manifest[str(year)] = {
            "file": path.name,
            "rows": rows,
            "sha256": hashlib.sha256(body).hexdigest(),
            "downloaded_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        print(f"{year}: {rows} rows")
        time.sleep(0.3)

    gsw_path = args.data_dir / GSW_FILE
    if args.force or not gsw_path.exists() or GSW_FILE not in manifest:
        body = fetch_gsw() if args.force or not gsw_path.exists() else gsw_path.read_bytes()
        gsw_path.write_bytes(body)
        manifest[GSW_FILE] = {
            "file": GSW_FILE,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "downloaded_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        print(f"{GSW_FILE}: {len(body)} bytes")

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"data in {args.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

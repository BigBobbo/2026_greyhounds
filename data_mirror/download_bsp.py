#!/usr/bin/env python3
"""Download Betfair's free historical greyhound BSP files.

Betfair publishes one CSV per day per market at
https://promo.betfair.com/betfairsp/prices/ — win and place markets for
greyhound racing, containing the Betfair Starting Price (BSP), pre-race
traded prices (ppwap/ppmin/ppmax), morning prices and traded volumes for
every runner. This is the backbone for honest market-aware backtesting.

promo.betfair.com is geo-blocked from US IPs, so this script must run from
an Irish/UK connection (any home broadband is fine). No third-party
packages needed — plain Python 3.

Usage:
    python3 download_bsp.py                     # 2021-01-01 → today, IRE only
    python3 download_bsp.py --from 2015-01-01   # longer history
    python3 download_bsp.py --all-tracks        # keep GB tracks too

Output (next to this script):
    bsp/win_greyhound.csv.gz    combined win-market rows
    bsp/place_greyhound.csv.gz  combined place-market rows
    bsp/raw/                    one small CSV per (market, day) — lets the
                                script resume if interrupted; safe to delete
                                after the combined files are built

When it finishes, commit the two .csv.gz files to the repo (or send them
on) — they are typically a few tens of MB for 5 years.
"""

import argparse
import csv
import glob
import gzip
import io
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

BASE = "https://promo.betfair.com/betfairsp/prices"
MARKETS = ("win", "place")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bsp")
RAW_DIR = os.path.join(OUT_DIR, "raw")


def fetch(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    except urllib.error.URLError:
        return None


def day_url(market: str, d: date) -> str:
    # e.g. dwbfgreyhoundwin01012024.csv
    return f"{BASE}/dwbfgreyhound{market}{d.strftime('%d%m%Y')}.csv"


def filter_rows(raw: bytes, ire_only: bool) -> list[str]:
    """Return CSV data rows (no header), optionally Irish tracks only.

    MENU_HINT looks like "GreyhoundRacing / Shel (IRE) 30th Jul" — Irish
    tracks carry an (IRE) marker.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return []
    out = []
    for line in lines[1:]:
        if not line.strip():
            continue
        if ire_only and "(IRE)" not in line:
            continue
        out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default="2021-01-01")
    ap.add_argument("--to", dest="date_to", default=str(date.today()))
    ap.add_argument("--all-tracks", action="store_true",
                    help="keep GB tracks too (default: Irish (IRE) rows only)")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds between requests (be polite)")
    args = ap.parse_args()

    d0 = date.fromisoformat(args.date_from)
    d1 = date.fromisoformat(args.date_to)
    ire_only = not args.all_tracks
    os.makedirs(RAW_DIR, exist_ok=True)

    total_days = (d1 - d0).days + 1
    header_by_market: dict[str, str] = {}
    misses = hits = 0

    d = d0
    n = 0
    while d <= d1:
        n += 1
        for market in MARKETS:
            shard = os.path.join(RAW_DIR, f"{market}_{d.strftime('%Y%m%d')}.csv")
            if os.path.exists(shard):
                continue  # resume: already fetched
            raw = fetch(day_url(market, d))
            time.sleep(args.delay)
            if raw is None or b"," not in raw[:200]:
                # No meeting that day (or file missing) — record the miss so
                # resume doesn't refetch endlessly.
                open(shard, "w").close()
                misses += 1
                continue
            first_line = raw.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
            header_by_market.setdefault(market, first_line)
            rows = filter_rows(raw, ire_only)
            with open(shard, "w", newline="") as f:
                f.write("\n".join(rows))
            hits += 1
        if n % 50 == 0:
            print(f"{d} — {n}/{total_days} days, {hits} files with data, {misses} empty", flush=True)
        d += timedelta(days=1)

    if hits == 0 and misses > total_days:
        print("WARNING: nothing downloaded — if you are outside IE/UK the site "
              "geo-blocks; otherwise the filename pattern may have changed. "
              f"Check {day_url('win', date(2024, 1, 1))} in a browser.")
        return 1

    # Combine shards into one gz per market
    for market in MARKETS:
        out_path = os.path.join(OUT_DIR, f"{market}_greyhound.csv.gz")
        shards = sorted(glob.glob(os.path.join(RAW_DIR, f"{market}_*.csv")))
        n_rows = 0
        with gzip.open(out_path, "wt", newline="") as out:
            hdr = header_by_market.get(market)
            if hdr:
                out.write(hdr + "\n")
            for s in shards:
                with open(s) as f:
                    content = f.read().strip()
                if content:
                    out.write(content + "\n")
                    n_rows += content.count("\n") + 1
        print(f"{out_path}: {n_rows} rows from {len(shards)} day-files")

    print("\nDone. Commit the two .csv.gz files in data_mirror/bsp/ "
          "(git add data_mirror/bsp/*.csv.gz) or send them on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

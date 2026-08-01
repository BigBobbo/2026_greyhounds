#!/usr/bin/env python3
"""Download archived Betfair greyhound BSP files from the Wayback Machine.

promo.betfair.com geo-blocks non-IE/UK IPs, but the Internet Archive's
August 2024 crawl captured ~7,600 daily BSP CSVs (win + place, 2012 →
mid-2024). This pulls the raw snapshots (`id_` URLs return the original
bytes), filters to Irish tracks, and writes the same combined
win/place_greyhound.csv.gz files that download_bsp.py produces — so a
later IE/UK run of download_bsp.py for the 2024→present tail merges
seamlessly (both keep per-day shards in bsp/raw/).

Usage:
    python3 data_mirror/download_bsp_wayback.py [--concurrency 3]
"""

import argparse
import asyncio
import glob
import gzip
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "bsp")
RAW_DIR = os.path.join(OUT_DIR, "raw")
CDX_URL = (
    "https://web.archive.org/cdx/search/cdx"
    "?url=promo.betfair.com/betfairsp/prices/dwbfgreyhound*"
    "&output=json&fl=original,timestamp,statuscode"
    "&filter=statuscode:200&collapse=urlkey&limit=50000"
)
UA = {"User-Agent": "Greyhound-Research-Bot/1.0 (race prediction research)"}


def fetch_cdx() -> list[tuple[str, str]]:
    req = urllib.request.Request(CDX_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        rows = json.load(r)[1:]
    out = []
    for orig, ts, _ in rows:
        if re.search(r"dwbfgreyhound(win|place)\d{8}\.csv", orig):
            out.append((orig, ts))
    return out


async def fetch_one(orig: str, ts: str, sem: asyncio.Semaphore, delay: float,
                    stats: dict) -> None:
    m = re.search(r"dwbfgreyhound(win|place)(\d{8})\.csv", orig)
    market, dmy = m.group(1), m.group(2)
    shard = os.path.join(RAW_DIR, f"{market}_{dmy[4:]}{dmy[2:4]}{dmy[:2]}.csv")
    if os.path.exists(shard):
        stats["skipped"] += 1
        return
    url = f"https://web.archive.org/web/{ts}id_/{orig}"

    def _get() -> bytes | None:
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=90) as r:
                    return r.read()
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    async with sem:
        raw = await asyncio.to_thread(_get)
        await asyncio.sleep(delay)

    if raw is None or b"," not in raw[:200]:
        open(shard + ".fail", "w").close()
        stats["failed"] += 1
        return
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if lines and "EVENT_ID" in lines[0].upper():
        stats.setdefault("header", lines[0])
    keep = [ln for ln in lines[1:] if "(IRE)" in ln]
    with open(shard, "w") as f:
        f.write("\n".join(keep))
    stats["ok"] += 1
    if (stats["ok"] + stats["failed"]) % 200 == 0:
        done = stats["ok"] + stats["failed"] + stats["skipped"]
        rate = stats["ok"] / max(time.time() - stats["t0"], 1)
        print(f"{done}/{stats['total']} files ({stats['failed']} failed, "
              f"{rate:.1f}/s)", flush=True)


async def main(concurrency: int, delay: float) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    files = fetch_cdx()
    print(f"{len(files)} archived BSP files to fetch", flush=True)
    stats = {"ok": 0, "failed": 0, "skipped": 0, "total": len(files),
             "t0": time.time()}
    sem = asyncio.Semaphore(concurrency)
    CHUNK = 1000
    for i in range(0, len(files), CHUNK):
        await asyncio.gather(*(fetch_one(o, t, sem, delay, stats)
                               for o, t in files[i:i + CHUNK]))

    print(f"fetch done: {stats['ok']} ok, {stats['failed']} failed, "
          f"{stats['skipped']} already present", flush=True)

    header = stats.get(
        "header",
        "EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,"
        "WIN_LOSE,BSP,PPWAP,MORNINGWAP,PPMAX,PPMIN,IPMAX,IPMIN,"
        "MORNINGTRADEDVOL,PPTRADEDVOL,IPTRADEDVOL",
    )
    for market in ("win", "place"):
        out_path = os.path.join(OUT_DIR, f"{market}_greyhound.csv.gz")
        shards = sorted(glob.glob(os.path.join(RAW_DIR, f"{market}_*.csv")))
        n = 0
        with gzip.open(out_path, "wt") as out:
            out.write(header + "\n")
            for s in shards:
                content = open(s).read().strip()
                if content:
                    out.write(content + "\n")
                    n += content.count("\n") + 1
        print(f"{out_path}: {n} IRE rows from {len(shards)} day-files", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.25)
    args = ap.parse_args()
    asyncio.run(main(args.concurrency, args.delay))

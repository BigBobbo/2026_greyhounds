#!/usr/bin/env python3
"""Settle a committed paper bet sheet against production results.

Reads docs/predictions/bet_sheet_<date>.json (rows written by the sheet
generator: track, race_no, trap, dog, tissue, min_price, stake), fetches
results over the production API, and appends one row per bet to
docs/predictions/paper_ledger.csv. Idempotent per date. Stdlib only.

Settlement convention: the bet is taken at the recorded tissue price
(fixed odds at decision time); an SP-settled P&L column is kept alongside
for comparison, and tissue-vs-SP is the CLV column. Dead heats pay
stake/n at the taken price.

Usage: python3 scripts/settle_paper_round.py --date 2026-08-21
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import urllib.request
from datetime import date as date_cls, datetime, timedelta, timezone

API = "https://2026greyhounds-production.up.railway.app"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "docs", "predictions")


def get(path: str):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(API + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        return json.loads(r.read().decode() or "null")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date_cls.fromisoformat,
                    default=datetime.now(timezone.utc).date() - timedelta(days=1))
    args = ap.parse_args()
    day = args.date

    sheet_path = os.path.join(OUT, f"bet_sheet_{day}.json")
    ledger_path = os.path.join(OUT, "paper_ledger.csv")
    if not os.path.exists(sheet_path):
        print(f"[settle] no sheet json for {day}; nothing to settle")
        return 0
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            if any(row.startswith(str(day)) for row in f):
                print(f"[settle] {day} already in ledger; skipping")
                return 0

    bets = json.load(open(sheet_path))
    races = get(f"/api/predictions/races-for-date?race_date={day}")
    if isinstance(races, dict):
        races = races.get("races") or races.get("items") or []
    idx = {(r.get("track_name"), r.get("race_number")): r["id"] for r in races}

    rows, missing = [], 0
    for b in bets:
        rid = idx.get((b["track"], b["race_no"]))
        if rid is None:
            rid = next((v for (t, n), v in idx.items()
                        if n == b["race_no"] and t and b["track"].split()[0] in t),
                       None)
        result = {"won": None, "finish": None, "sp": None}
        if rid is not None:
            det = get(f"/api/races/{rid}") or {}
            entries = det.get("entries") or []
            entry = next((e for e in entries if e.get("trap") == b["trap"]), None)
            if entry and entry.get("finish_position") is not None:
                n_dh = sum(1 for e in entries if e.get("finish_position") == 1)
                won = entry["finish_position"] == 1
                result = {"won": won, "finish": entry["finish_position"],
                          "sp": entry.get("sp_decimal"), "n_dh": max(n_dh, 1)}
        if result["won"] is None:
            missing += 1
            continue
        stake = float(b["stake"])
        ndh = result.get("n_dh", 1)
        if result["won"]:
            pt = stake * (b["tissue"] - 1.0) / ndh - stake * (1 - 1 / ndh)
            ps = (stake * (result["sp"] - 1.0) / ndh - stake * (1 - 1 / ndh)
                  if result["sp"] else None)
        else:
            pt, ps = -stake, -stake if result["sp"] else None
        rows.append({
            "date": str(day), "time": b["time"], "track": b["track"],
            "race_no": b["race_no"], "trap": b["trap"], "dog": b["dog"],
            "p_blend": b["p_blend"], "tissue": b["tissue"],
            "min_price": b["min_price"], "stake": stake,
            "finish": result["finish"], "sp": result["sp"],
            "won": int(result["won"]),
            "pnl_tissue": round(pt, 2),
            "pnl_sp": round(ps, 2) if ps is not None else "",
            "clv": round(b["tissue"] / result["sp"], 3) if result["sp"] else "",
        })

    if not rows:
        print(f"[settle] {day}: no settleable bets ({missing} missing results)")
        return 0
    exists = os.path.exists(ledger_path)
    with open(ledger_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    staked = sum(r["stake"] for r in rows)
    pnl = sum(r["pnl_tissue"] for r in rows)
    print(f"[settle] {day}: {len(rows)} bets, {sum(r['won'] for r in rows)} wins, "
          f"staked {staked:.2f}, P&L(tissue) {pnl:+.2f} "
          f"({pnl / staked * 100:+.1f}%), {missing} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())

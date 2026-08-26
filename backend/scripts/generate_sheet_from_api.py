#!/usr/bin/env python3
"""Generate the daily paper bet sheet from production predictions + a
tissue-price capture file. Stdlib only.

Inputs:
  --date YYYY-MM-DD       (default: today UTC)
  --tissue PATH           JSONL: one line per race with race-level track,
                          time, and dogs[{trap,name,forecast}] — captured
                          from the GreyhoundBET card feed by the session.

Model probabilities come from the production API's saved experiment-59
predictions (created pre-race by the 11:30 cron). Tissue races are joined
to API races by dog-name matching (>=4 of 6 normalized names; ambiguity
refused). The certified rule (docs/VERDICT-2026-08-21.md):

    blend a=0.712 b=1.120 over proportionally de-vigged tissue
    edge >= 0.03 at the tissue price, price in [1.5, 12]
    p_blend <= 0.55, field-average data_completeness >= 0.6
    races already run at generation time are skipped
    quarter-Kelly at tissue, completeness-scaled, 5% per-bet cap,
    top 15 edges, 10% daily cap on the EUR 100 basis

Writes docs/predictions/bet_sheet_<date>.md and .json (the row schema
settle_paper_round.py consumes).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import sys
import urllib.request
from datetime import date as date_cls, datetime, timezone
from zoneinfo import ZoneInfo

API = "https://2026greyhounds-production.up.railway.app"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "docs", "predictions")
DUBLIN = ZoneInfo("Europe/Dublin")

ALPHA, BETA = 0.7124744889059822, 1.1195819096432853
MARGIN, MIN_PRICE, MAX_PRICE = 0.03, 1.5, 12.0
MAX_BLEND, MIN_COMPLETENESS = 0.55, 0.6
BANKROLL, KELLY_FRACTION = 100.0, 0.25
MAX_STAKE_PCT, DAILY_CAP_PCT, TOP_N = 0.05, 0.10, 15


def get(path: str):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(API + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        return json.loads(r.read().decode() or "null")


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def frac_to_dec(f: str):
    f = (f or "").strip().lower()
    if not f:
        return None
    if f in ("evs", "evens", "1/1"):
        return 2.0
    m = re.match(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", f)
    if m:
        return 1.0 + float(m.group(1)) / float(m.group(2))
    try:
        v = float(f)
        return v if v > 1 else None
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date_cls.fromisoformat,
                    default=datetime.now(timezone.utc).date())
    ap.add_argument("--tissue", required=True)
    args = ap.parse_args()
    day = args.date

    tissue_races = []
    for line in open(args.tissue):
        line = line.strip()
        if line:
            tissue_races.append(json.loads(line))

    races = get(f"/api/predictions/races-for-date?race_date={day}")
    if isinstance(races, dict):
        races = races.get("races") or races.get("items") or []
    api_races = []
    for r in races:
        preds = get(f"/api/predictions/race/{r['id']}/saved?experiment_id=59")
        if isinstance(preds, dict):
            preds = preds.get("predictions") or preds.get("items") or []
        priced = [p for p in preds if p.get("win_probability") is not None]
        if priced:
            api_races.append({
                "race_id": r["id"], "track": r.get("track_name"),
                "race_no": r.get("race_number"), "preds": priced,
            })
    print(f"[sheet] {len(api_races)} API races with predictions, "
          f"{len(tissue_races)} tissue races")

    now_dublin = datetime.now(DUBLIN)
    rows = []
    skipped = {"no_join": 0, "already_run": 0, "no_edge": 0,
               "thin_form": 0, "no_prices": 0}
    for tr in tissue_races:
        dogs = tr.get("dogs") or []
        prices_by_name = {norm(d["name"]): (d["trap"], frac_to_dec(d.get("forecast")))
                          for d in dogs}
        if len([1 for _, (_, p) in prices_by_name.items() if p]) < 5:
            skipped["no_prices"] += 1
            continue
        cands = []
        for ar in api_races:
            names = {norm(p.get("dog_name") or ""): p for p in ar["preds"]}
            hits = sum(1 for n in prices_by_name if n in names)
            if hits >= 4:
                cands.append((ar, hits))
        if len(cands) != 1:
            skipped["no_join"] += 1
            continue
        ar = cands[0][0]

        hhmm = (tr.get("time") or "")[:5]
        if hhmm:
            off = now_dublin.replace(hour=int(hhmm[:2]), minute=int(hhmm[3:5]),
                                     second=0, microsecond=0)
            if off <= now_dublin:
                skipped["already_run"] += 1
                continue

        plist = sorted(ar["preds"], key=lambda p: p.get("trap") or 0)
        comps = [p.get("data_completeness") or 0.0 for p in plist]
        if sum(comps) / len(comps) < MIN_COMPLETENESS:
            skipped["thin_form"] += 1
            continue
        pm, price = [], []
        ok = True
        for p in plist:
            n = norm(p.get("dog_name") or "")
            if n not in prices_by_name or prices_by_name[n][1] is None:
                ok = False
                break
            pm.append(float(p["win_probability"]))
            price.append(prices_by_name[n][1])
        if not ok or sum(pm) <= 0:
            skipped["no_prices"] += 1
            continue
        tot = sum(pm)
        pm = [x / tot for x in pm]
        inv = [1.0 / o for o in price]
        s_inv = sum(inv)
        mkt = [x / s_inv for x in inv]
        sc = [ALPHA * math.log(max(x, 1e-9)) + BETA * math.log(max(m, 1e-9))
              for x, m in zip(pm, mkt)]
        mx = max(sc)
        ex = [math.exp(x - mx) for x in sc]
        se = sum(ex)
        pb = [x / se for x in ex]

        any_edge = False
        for i, p in enumerate(plist):
            o = price[i]
            edge = pb[i] - 1.0 / o
            if o < MIN_PRICE or o > MAX_PRICE or edge < MARGIN or pb[i] > MAX_BLEND:
                continue
            any_edge = True
            b = o - 1.0
            f_star = (b * pb[i] - (1.0 - pb[i])) / b
            comp = p.get("data_completeness") or 1.0
            stake_pct = min(max(f_star, 0.0) * KELLY_FRACTION * comp, MAX_STAKE_PCT)
            stake = round(BANKROLL * stake_pct, 2)
            if stake < 0.5:
                continue
            rows.append({
                "time": hhmm, "track": ar["track"], "race_no": ar["race_no"],
                "trap": p.get("trap"), "dog": p.get("dog_name"),
                "p_model": round(pm[i], 3), "p_blend": round(pb[i], 3),
                "tissue": round(o, 2),
                "min_price": round(1.0 / (pb[i] - MARGIN), 2),
                "edge_at_tissue": round(edge, 4), "stake": stake,
            })
        if not any_edge:
            skipped["no_edge"] += 1

    if len(rows) > TOP_N:
        rows.sort(key=lambda r: r["edge_at_tissue"], reverse=True)
        rows = rows[:TOP_N]
    total = sum(r["stake"] for r in rows)
    cap = BANKROLL * DAILY_CAP_PCT
    scale = 1.0
    if total > cap > 0:
        scale = cap / total
        for r in rows:
            r["stake"] = round(r["stake"] * scale, 2)
    rows = [r for r in rows if r["stake"] >= 0.5]
    rows.sort(key=lambda r: (r["time"], r["track"] or "", r["race_no"] or 0))

    stamp = now_dublin.strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        f"# Bet sheet — {day} (blend rule, certified 2026-08-21)", "",
        f"Generated {stamp} | paper basis EUR {BANKROLL:.0f} | quarter-Kelly | "
        f"margin {MARGIN} | blend a={ALPHA:.3f} b={BETA:.3f} | commission 0"
        + (f" | daily cap scaled x{scale:.2f}" if scale < 1 else ""), "",
        "**Back WIN only if the app price is AT OR ABOVE the minimum price;",
        "otherwise skip. Prefer a Best-Odds-Guaranteed bookmaker.**", "",
        "| Time | Track | Race | Trap | Dog | Blend prob | Tissue | Min price | Stake |",
        "|------|-------|------|------|-----|-----------|--------|-----------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['time']} | {r['track']} | R{r['race_no']} | {r['trap']} "
            f"| **{r['dog']}** | {r['p_blend']:.1%} | {r['tissue']} "
            f"| {r['min_price']} | EUR {r['stake']:.2f} |")
    if not rows:
        lines.append("| - | no qualifying bets | | | | | | | |")
    lines += ["", f"{len(rows)} bet(s) | staked EUR "
              f"{sum(r['stake'] for r in rows):.2f} | skips: {skipped}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f"bet_sheet_{day}.md"), "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(OUT, f"bet_sheet_{day}.json"), "w"), indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

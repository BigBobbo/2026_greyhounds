#!/usr/bin/env python3
"""Replay the bet-sheet rule against settled results.

Production has been saving predictions daily since 9 August, stamped
with created_at — usually the morning before the race — so these are
genuine pre-race probabilities, not hindsight. This applies the exact
rule the bet sheet prints and settles it on real starting prices:

    top pick per race
    min_odds = 1 + (1/(p - min_edge) - 1) / (1 - commission)
    bet only if the available price >= min_odds
    win  -> + stake * (price - 1) * (1 - commission)
    lose -> - stake

Honest limits, stated because they matter:

  * SP is the price at the off. Whoever bets sees an earlier price, and
    early prices differ — outsiders usually shorten, favourites drift or
    steam. Using SP both to decide and to settle is the same
    approximation the backtest made, and it is optimistic to the extent
    that the decision needs a price you cannot know until the off.
  * Any race whose prediction was created after the off is dropped.
  * A few hundred bets is a small sample; the confidence interval is
    reported for exactly that reason.

Usage:
    python3 scripts/evaluate_bet_sheet_strategy.py --from 2026-08-09 --to 2026-08-19
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import statistics
import sys
import urllib.error
import urllib.request
from datetime import date as date_cls
from datetime import datetime, timedelta

DEFAULT_API = "https://2026greyhounds-production.up.railway.app"
DEFAULT_EXPERIMENT = 59
# Ireland is UTC+1 in summer; created_at is stored in UTC.
IRISH_OFFSET_HOURS = 1


def get(api: str, path: str):
    ctx = ssl.create_default_context()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{api.rstrip('/')}{path}", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                return json.loads(resp.read().decode() or "null")
        except urllib.error.HTTPError:
            return None
        except Exception:                              # noqa: BLE001
            if attempt == 2:
                return None
            import time
            time.sleep(2)
    return None


def as_list(x, *keys):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        for k in keys:
            if x.get(k):
                return x[k]
    return []


def min_odds_for(prob: float, cfg: dict) -> float | None:
    min_edge = float(cfg["min_edge"])
    if prob <= min_edge:
        return None
    net_floor = 1.0 / (prob - min_edge)
    odds = 1.0 + (net_floor - 1.0) / (1.0 - float(cfg["commission_rate"]))
    return max(odds, float(cfg["min_odds"]))


def stake_for(prob: float, odds: float, cfg: dict, completeness: float) -> float:
    net = 1.0 + (odds - 1.0) * (1.0 - float(cfg["commission_rate"]))
    b = net - 1.0
    edge = prob * net - 1.0
    if b <= 0 or edge <= 0:
        return 0.0
    frac = (edge / b) * float(cfg["kelly_fraction"]) * completeness
    stake = frac * float(cfg["current_bankroll"])
    return round(min(stake, float(cfg["max_stake_pct"])
                     * float(cfg["current_bankroll"])), 2)


def bootstrap_ci(pnls: list[float], stakes: list[float], n: int = 2000):
    """Race-level bootstrap of ROI — the bets are independent races."""
    if not pnls:
        return (0.0, 0.0)
    rng = random.Random(12345)
    idx = range(len(pnls))
    rois = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        s = sum(stakes[i] for i in pick)
        if s > 0:
            rois.append(100.0 * sum(pnls[i] for i in pick) / s)
    if not rois:
        return (0.0, 0.0)
    rois.sort()
    return (rois[int(0.05 * len(rois))], rois[int(0.95 * len(rois))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=date_cls.fromisoformat, required=True)
    ap.add_argument("--to", dest="end", type=date_cls.fromisoformat, required=True)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--experiment-id", type=int, default=DEFAULT_EXPERIMENT)
    args = ap.parse_args()

    cfg = get(args.api, "/api/bankroll/config")
    if not cfg or "current_bankroll" not in cfg:
        print("cannot read bankroll config", file=sys.stderr)
        return 1
    commission = float(cfg["commission_rate"])

    considered = skipped_no_sp = skipped_post_hoc = 0
    placed_pnl: list[float] = []
    placed_stake: list[float] = []
    placed_win = 0
    skipped_price = 0
    # Comparisons
    all_pnl: list[float] = []
    all_stake: list[float] = []
    all_win = 0
    fav_pnl: list[float] = []
    fav_win = 0
    fav_n = 0

    day = args.start
    while day <= args.end:
        races = as_list(get(args.api, f"/api/predictions/races-for-date?race_date={day}"),
                        "races", "items")
        for race in races:
            detail = get(args.api, f"/api/races/{race['id']}") or {}
            entries = detail.get("entries") or []
            winner = next((e for e in entries
                           if e.get("finish_position") == 1), None)
            if not winner:
                continue
            preds = as_list(get(args.api, f"/api/predictions/race/{race['id']}"
                                          f"/saved?experiment_id={args.experiment_id}"),
                            "predictions", "items")
            priced = [p for p in preds if p.get("win_probability") is not None]
            if not priced:
                continue

            rt = str(detail.get("race_time") or "")[:5]
            if rt and priced[0].get("created_at"):
                try:
                    hh, _, mm = rt.partition(":")
                    off = datetime.combine(day, datetime.min.time()) + timedelta(
                        hours=int(hh) - IRISH_OFFSET_HOURS, minutes=int(mm))
                    made = datetime.fromisoformat(priced[0]["created_at"])
                    if made >= off:
                        skipped_post_hoc += 1
                        continue
                except ValueError:
                    pass

            considered += 1
            top = max(priced, key=lambda p: p["win_probability"])
            by_trap = {e.get("trap"): e for e in entries}
            entry = by_trap.get(top.get("trap"))
            sp = (entry or {}).get("sp_decimal")
            if not sp or sp <= 1:
                skipped_no_sp += 1
                continue
            sp = float(sp)
            won = (entry or {}).get("finish_position") == 1

            prob = float(top["win_probability"])
            floor = min_odds_for(prob, cfg)
            completeness = float(top.get("data_completeness") or 1.0)

            # The sheet's rule: stake sized at the floor, bet only if the
            # price is at or above it.
            if floor is not None:
                stake = stake_for(prob, floor, cfg, completeness)
                if stake >= 0.01:
                    if sp >= floor:
                        pnl = (stake * (sp - 1.0) * (1.0 - commission)
                               if won else -stake)
                        placed_pnl.append(pnl)
                        placed_stake.append(stake)
                        placed_win += int(won)
                    else:
                        skipped_price += 1

            # Comparison: back every top pick at SP, flat 1 unit.
            all_pnl.append((sp - 1.0) * (1.0 - commission) if won else -1.0)
            all_stake.append(1.0)
            all_win += int(won)

            # Comparison: back the favourite, flat 1 unit (needs a full book).
            book = {e.get("trap"): e.get("sp_decimal") for e in entries
                    if e.get("sp_decimal")}
            if len(book) == len(entries) and book:
                fav_trap = min(book, key=book.get)
                fav_sp = float(book[fav_trap])
                fav_won = by_trap[fav_trap].get("finish_position") == 1
                fav_pnl.append((fav_sp - 1.0) * (1.0 - commission)
                               if fav_won else -1.0)
                fav_win += int(fav_won)
                fav_n += 1
        print(f"  {day}: running total {len(placed_pnl)} bets", flush=True)
        day += timedelta(days=1)

    print(f"\n{'=' * 62}")
    print(f"Races with a pre-race prediction and a result : {considered}")
    print(f"  dropped, prediction made after the off      : {skipped_post_hoc}")
    print(f"  dropped, no starting price for the pick     : {skipped_no_sp}")
    print(f"  skipped by the rule, price below minimum    : {skipped_price}")
    print(f"{'=' * 62}")

    if placed_pnl:
        total = sum(placed_pnl)
        staked = sum(placed_stake)
        roi = 100.0 * total / staked
        lo, hi = bootstrap_ci(placed_pnl, placed_stake)
        print(f"\nBET SHEET RULE — {len(placed_pnl)} bets")
        print(f"  wins            : {placed_win}/{len(placed_pnl)} "
              f"= {placed_win / len(placed_pnl):.1%}")
        print(f"  staked          : EUR {staked:.2f}")
        print(f"  profit/loss     : EUR {total:+.2f}")
        print(f"  ROI             : {roi:+.1f}%   (90% CI {lo:+.1f}% .. {hi:+.1f}%)")
    else:
        print("\nBET SHEET RULE — no qualifying bets in this window")

    if all_pnl:
        roi = 100.0 * sum(all_pnl) / sum(all_stake)
        lo, hi = bootstrap_ci(all_pnl, all_stake)
        print(f"\nEVERY TOP PICK AT SP (no price filter) — {len(all_pnl)} bets")
        print(f"  wins            : {all_win}/{len(all_pnl)} "
              f"= {all_win / len(all_pnl):.1%}")
        print(f"  ROI             : {roi:+.1f}%   (90% CI {lo:+.1f}% .. {hi:+.1f}%)")

    if fav_pnl:
        roi = 100.0 * sum(fav_pnl) / len(fav_pnl)
        lo, hi = bootstrap_ci(fav_pnl, [1.0] * len(fav_pnl))
        print(f"\nBACK THE FAVOURITE (baseline) — {fav_n} bets")
        print(f"  wins            : {fav_win}/{fav_n} = {fav_win / fav_n:.1%}")
        print(f"  ROI             : {roi:+.1f}%   (90% CI {lo:+.1f}% .. {hi:+.1f}%)")

    print("\nCaveat: SP is the price at the off. A person betting earlier "
          "sees a different price,\nso using SP both to decide and to settle "
          "is optimistic. Treat this as an upper bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

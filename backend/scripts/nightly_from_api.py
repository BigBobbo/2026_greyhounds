#!/usr/bin/env python3
"""The nightly cycle, driven by the production API.

Replaces nightly_routine.py, which rebuilt a 108MB database from dumps,
scraped the whole results gap and recomputed features every night —
roughly an hour of work that an unattended session did not reliably
finish, so no sheet was committed for over two weeks.

Production already does all of that on its own schedule and stores the
result. This reads it: a few dozen HTTP calls, seconds rather than an
hour, and no local state to go stale.

Standard library only and no database, so a fresh container can run it
with nothing installed.

Outputs, under docs/predictions/:
    <date>.csv / <date>.md          every runner's probability (the
                                    pre-registered record)
    bet_sheet_<date>.md             top pick per race with the minimum
                                    acceptable price and stake
    <yesterday>-scorecard.md        yesterday's sheet scored on results
    record.csv                      one row per scored night

Usage:
    python3 scripts/nightly_from_api.py [--date 2026-08-20]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone

DEFAULT_API = "https://2026greyhounds-production.up.railway.app"
DEFAULT_EXPERIMENT = 59

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs", "predictions")

BUCKETS = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
           (0.20, 0.30), (0.30, 0.40), (0.40, 1.01)]


# --- HTTP ---------------------------------------------------------------

def get(api: str, path: str, retries: int = 3):
    """GET JSON, retrying transient failures (the app sleeps when idle)."""
    url = f"{api.rstrip('/')}{path}"
    ctx = ssl.create_default_context()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                return json.loads(resp.read().decode() or "null")
        except Exception as e:                       # noqa: BLE001
            last = e
            if attempt < retries - 1:
                import time
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {path} failed after {retries} tries: {last}")


def races_for(api: str, day: date_cls) -> list:
    rows = get(api, f"/api/predictions/races-for-date?race_date={day}")
    if isinstance(rows, dict):
        rows = rows.get("races") or rows.get("items") or []
    return rows or []


def race_detail(api: str, race_id: int) -> dict:
    return get(api, f"/api/races/{race_id}") or {}


def predictions_for(api: str, race_id: int, experiment_id: int) -> list:
    rows = get(api, f"/api/predictions/race/{race_id}/saved"
                    f"?experiment_id={experiment_id}")
    if isinstance(rows, dict):
        rows = rows.get("predictions") or rows.get("items") or []
    return rows or []


# --- Today's sheets -----------------------------------------------------

def build_sheets(api: str, day: date_cls, experiment_id: int,
                 cfg: dict) -> dict:
    races = races_for(api, day)
    if not races:
        return {"races": 0, "runners": 0, "bets": 0,
                "note": f"no races published for {day}"}

    generated_at = datetime.now(timezone.utc)
    stamp = generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []
    bets: list[dict] = []
    missing = 0

    for race in sorted(races, key=lambda r: (r.get("track_name") or "",
                                             r.get("race_number") or 0)):
        race_id = race["id"]
        detail = race_detail(api, race_id)
        race_time = str(detail.get("race_time") or "")[:5]
        preds = predictions_for(api, race_id, experiment_id)
        priced = [p for p in preds if p.get("win_probability") is not None]
        if not priced:
            missing += 1
            continue
        priced.sort(key=lambda p: p["win_probability"], reverse=True)

        for rank, p in enumerate(priced, 1):
            wp = float(p["win_probability"])
            rows.append({
                "race_id": race_id,
                "date": str(day),
                "time": race_time,
                "track": race.get("track_name"),
                "race_no": race.get("race_number"),
                "trap": p.get("trap"),
                "dog": p.get("dog_name"),
                "win_prob": round(wp, 4),
                "fair_odds": round(1.0 / wp, 2) if wp > 0 else None,
                "rank": rank,
                "confidence": p.get("confidence_tier"),
                "data_completeness": round(
                    float(p.get("data_completeness") or 0.0), 3),
                "generated_at": stamp,
            })

        bet = bet_for_race(priced[0], cfg)
        if bet:
            bet.update(time=race_time, track=race.get("track_name"),
                       race_no=race.get("race_number"))
            bets.append(bet)

    if not rows:
        return {"races": 0, "runners": 0, "bets": 0,
                "note": f"{len(races)} races published but none predicted yet"}

    bets.sort(key=lambda b: (b["time"] or "99", b["track"] or "", b["race_no"]))
    bets = cap_daily_exposure(bets, cfg)

    os.makedirs(OUT_DIR, exist_ok=True)
    write_prediction_csv(day, rows)
    write_prediction_md(day, rows, experiment_id, generated_at)
    write_bet_sheet(day, bets, cfg, experiment_id, generated_at)

    return {
        "races": len({r["race_id"] for r in rows}),
        "runners": len(rows),
        "bets": len(bets),
        "staked": round(sum(b["stake"] for b in bets), 2),
        "races_without_predictions": missing,
        "top": [f"{b['track']} R{b['race_no']} {b['dog']} "
                f"(min {b['min_odds']})" for b in bets[:3]],
    }


def bet_for_race(top: dict, cfg: dict) -> dict | None:
    """The backtested strategy: the model's top pick, priced so the edge
    survives commission. Prints a minimum acceptable price rather than
    assuming a fill — no live odds feed is needed or available."""
    wp = float(top["win_probability"])
    min_edge = float(cfg["min_edge"])
    commission = float(cfg["commission_rate"])
    if wp <= min_edge:
        return None
    net_floor = 1.0 / (wp - min_edge)
    min_odds = 1.0 + (net_floor - 1.0) / (1.0 - commission)
    min_odds = max(min_odds, float(cfg["min_odds"]))

    # Kelly at exactly the floor price — any better price only helps.
    net_odds = 1.0 + (min_odds - 1.0) * (1.0 - commission)
    b = net_odds - 1.0
    if b <= 0:
        return None
    edge = wp * net_odds - 1.0
    if edge <= 0:
        return None
    fraction = (edge / b) * float(cfg["kelly_fraction"])
    fraction *= float(top.get("data_completeness") or 1.0)
    stake = fraction * float(cfg["current_bankroll"])
    stake = min(stake, float(cfg["max_stake_pct"]) * float(cfg["current_bankroll"]))
    stake = round(stake, 2)
    if stake < 0.01:
        return None
    return {
        "dog": top.get("dog_name"), "trap": top.get("trap"),
        "win_prob": round(wp, 3), "min_odds": round(min_odds, 2),
        "stake": stake, "confidence": top.get("confidence_tier"),
    }


def cap_daily_exposure(bets: list[dict], cfg: dict) -> list[dict]:
    cap = float(cfg["max_daily_exposure_pct"]) * float(cfg["current_bankroll"])
    total = sum(b["stake"] for b in bets)
    if total <= cap or total <= 0:
        return bets
    scale = cap / total
    kept = []
    for b in bets:
        b = dict(b)
        b["stake"] = round(b["stake"] * scale, 2)
        if b["stake"] >= 0.01:
            kept.append(b)
    return kept


# --- Writers ------------------------------------------------------------

def write_prediction_csv(day: date_cls, rows: list[dict]) -> None:
    path = os.path.join(OUT_DIR, f"{day}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_md(day, rows, experiment_id, generated_at) -> None:
    n_races = len({r["race_id"] for r in rows})
    lines = [
        f"# Prediction sheet — {day}",
        "",
        f"Experiment {experiment_id} · model-only probabilities · generated "
        f"{generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"{n_races} races, {len(rows)} runners. **Bold** = model's top pick. "
        "Fair odds = 1 / probability.",
        "",
    ]
    last = None
    for r in rows:
        key = (r["time"], r["track"], r["race_no"])
        if key != last:
            if last is not None:
                lines.append("")
            last = key
            lines += [
                f"### {r['time']} — {r['track']} R{r['race_no']}",
                "",
                "| Trap | Dog | Win prob | Fair odds | Confidence | Data |",
                "|------|-----|----------|-----------|------------|------|",
            ]
        dog = f"**{r['dog']}**" if r["rank"] == 1 else r["dog"]
        lines.append(
            f"| {r['trap']} | {dog} | {r['win_prob']:.1%} | {r['fair_odds']} "
            f"| {r['confidence']} | {r['data_completeness']:.0%} |"
        )
    with open(os.path.join(OUT_DIR, f"{day}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_bet_sheet(day, bets, cfg, experiment_id, generated_at) -> None:
    bankroll = float(cfg["current_bankroll"])
    lines = [
        f"# Bet sheet — {day}",
        "",
        f"Bankroll €{bankroll:.0f} · quarter-Kelly · day cap "
        f"€{bankroll * float(cfg['max_daily_exposure_pct']):.0f} · "
        f"experiment {experiment_id} · generated "
        f"{generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        "**Rule: back WIN only if the live price is AT OR ABOVE the minimum "
        "odds. If it is lower — skip, no exceptions.**",
        "",
        "Prices are not fetched automatically: check each one in your betting "
        "app. A price above the minimum only makes the bet better.",
        "",
        "| Time | Track | Race | Trap | Dog | Min odds | Stake |",
        "|------|-------|------|------|-----|----------|-------|",
    ]
    for b in bets:
        lines.append(
            f"| {b['time']} | {b['track']} | R{b['race_no']} | {b['trap']} "
            f"| **{b['dog']}** | {b['min_odds']} | €{b['stake']:.2f} |"
        )
    if not bets:
        lines.append("| — | no qualifying bets today | | | | | |")
    lines += [
        "",
        f"{len(bets)} bet(s) · total staked if every price is available: "
        f"€{sum(b['stake'] for b in bets):.2f}",
    ]
    with open(os.path.join(OUT_DIR, f"bet_sheet_{day}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# --- Scoring ------------------------------------------------------------

def score_day(api: str, day: date_cls) -> dict | None:
    """Score a previously committed sheet against results. Methodology is
    unchanged from score_prediction_sheet.py: strict cohort only, measured
    against the model's own expectation and a uniform-field baseline."""
    csv_path = os.path.join(OUT_DIR, f"{day}.csv")
    card_path = os.path.join(OUT_DIR, f"{day}-scorecard.md")
    if not os.path.exists(csv_path) or os.path.exists(card_path):
        return None

    import math
    from collections import defaultdict

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    by_race: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_race[int(r["race_id"])].append(r)

    generated_at = None
    if rows[0].get("generated_at"):
        try:
            generated_at = datetime.strptime(
                rows[0]["generated_at"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    winners: dict[int, int] = {}
    starts: dict[int, datetime] = {}
    mismatched: list[str] = []
    for race_id in by_race:
        detail = race_detail(api, race_id)
        # A race id only means anything within the database that issued
        # it. A sheet written from a different database points at
        # unrelated races, and scoring it produces plausible-looking
        # nonsense rather than an error — so verify identity first.
        row = by_race[race_id][0]
        same = (
            str(detail.get("race_date")) == row["date"]
            and (detail.get("track_name") or "") == (row["track"] or "")
            and str(detail.get("race_number")) == str(row["race_no"])
        )
        if not same:
            mismatched.append(
                f"id {race_id}: sheet says {row['date']} {row['track']} "
                f"R{row['race_no']}, API says {detail.get('race_date')} "
                f"{detail.get('track_name')} R{detail.get('race_number')}"
            )
            continue
        for e in detail.get("entries") or []:
            if e.get("finish_position") == 1 and e.get("trap") is not None:
                winners[race_id] = e["trap"]
        rt = str(detail.get("race_time") or "")[:5]
        if rt:
            hh, _, mm = rt.partition(":")
            try:
                # Irish local time is UTC+1 in summer; the sheet is stamped
                # in UTC, so shift before comparing.
                starts[race_id] = datetime.combine(
                    day, datetime.min.time(), tzinfo=timezone.utc
                ) + timedelta(hours=int(hh) - 1, minutes=int(mm))
            except ValueError:
                pass

    if mismatched:
        print(f"[nightly] REFUSING to score {day}: {len(mismatched)} of "
              f"{len(by_race)} race ids do not match the API's races. This "
              "sheet was written against a different database, so scoring "
              "it would compare predictions to unrelated races.",
              file=sys.stderr)
        for line in mismatched[:3]:
            print(f"    {line}", file=sys.stderr)
        return None

    def strict(race_id: int) -> bool:
        if generated_at is None or race_id not in starts:
            return True
        return starts[race_id] > generated_at

    scored = [rid for rid in by_race if rid in winners and strict(rid)]
    excluded = len([rid for rid in by_race if rid in winners]) - len(scored)
    if not scored:
        return None

    hits = 0
    expected = 0.0
    logloss = base_logloss = 0.0
    brier = base_brier = 0.0
    runners = 0
    bucket = {b: [0, 0.0, 0] for b in BUCKETS}

    for rid in scored:
        preds = by_race[rid]
        field = len(preds)
        win_trap = winners[rid]
        probs = {int(p["trap"]): float(p["win_prob"]) for p in preds}
        total = sum(probs.values()) or 1.0
        probs = {t: p / total for t, p in probs.items()}
        top_trap = max(probs, key=probs.get)
        expected += probs[top_trap]
        if top_trap == win_trap:
            hits += 1
        if win_trap in probs:
            logloss += -math.log(max(probs[win_trap], 1e-9))
            base_logloss += -math.log(1.0 / field)
            for trap, p in probs.items():
                won = 1.0 if trap == win_trap else 0.0
                brier += (p - won) ** 2
                base_brier += (1.0 / field - won) ** 2
                runners += 1
                for b in BUCKETS:
                    if b[0] <= p < b[1]:
                        bucket[b][0] += int(won)
                        bucket[b][1] += p
                        bucket[b][2] += 1
                        break

    n = len(scored)
    summary = {
        "date": str(day), "races": n, "hits": hits,
        "hit_rate": round(hits / n, 4),
        "expected_hit_rate": round(expected / n, 4),
        "log_loss": round(logloss / n, 4),
        "log_loss_uniform": round(base_logloss / n, 4),
        "brier": round(brier / runners, 4) if runners else "",
        "brier_uniform": round(base_brier / runners, 4) if runners else "",
        "excluded_post_start": excluded,
    }

    lines = [
        f"# Scorecard — {day}", "",
        f"Scored {date_cls.today()} against results from the production API. "
        "Strict cohort only: races that started after the sheet was "
        f"generated ({excluded} excluded).", "",
        f"- Top-pick hit rate: **{hits}/{n} = {hits / n:.1%}** "
        f"(model expected {expected / n:.1%})",
        f"- Winner log loss: **{logloss / n:.4f}** "
        f"(uniform baseline {base_logloss / n:.4f})",
        f"- Brier per runner: **{brier / runners:.4f}** "
        f"(uniform {base_brier / runners:.4f})" if runners else "",
        "", "| Predicted | Observed | n |", "|---|---|---|",
    ]
    for b in BUCKETS:
        wins, sump, cnt = bucket[b]
        if cnt:
            lines.append(f"| {sump / cnt:.1%} | {wins / cnt:.1%} | {cnt} |")
    with open(card_path, "w") as f:
        f.write("\n".join(x for x in lines if x != "") + "\n")

    append_record(summary)
    return summary


def append_record(summary: dict) -> None:
    """Append one night to record.csv, honouring an existing header so a
    schema change cannot silently shift columns in the running record."""
    record = os.path.join(OUT_DIR, "record.csv")
    fields = list(summary.keys())
    if os.path.exists(record):
        with open(record) as f:
            header = f.readline().strip()
        if header:
            fields = header.split(",")
    row = {k: summary.get(k, "") for k in fields}
    exists = os.path.exists(record)
    with open(record, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# --- Entry point --------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date_cls.fromisoformat, default=None)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--experiment-id", type=int, default=DEFAULT_EXPERIMENT)
    args = ap.parse_args()

    day = args.date or datetime.now(timezone.utc).date()
    print(f"[nightly] {day} via {args.api}", flush=True)

    try:
        cfg = get(args.api, "/api/bankroll/config") or {}
    except Exception as e:                            # noqa: BLE001
        print(f"[nightly] FAILED: cannot reach the app: {e}", file=sys.stderr)
        return 1
    if "current_bankroll" not in cfg:
        print("[nightly] FAILED: bankroll config missing", file=sys.stderr)
        return 1

    scored = score_day(args.api, day - timedelta(days=1))
    if scored:
        print(f"[nightly] scored {scored['date']}: "
              f"{scored['hits']}/{scored['races']} top picks "
              f"({scored['hit_rate']:.1%} vs expected "
              f"{scored['expected_hit_rate']:.1%}), log loss "
              f"{scored['log_loss']} vs uniform {scored['log_loss_uniform']}")
    else:
        print("[nightly] nothing to score for yesterday")

    result = build_sheets(args.api, day, args.experiment_id, cfg)
    if result.get("note"):
        print(f"[nightly] {result['note']}")
    else:
        print(f"[nightly] sheet: {result['races']} races, "
              f"{result['runners']} runners; bet sheet: {result['bets']} bets, "
              f"€{result['staked']:.2f} staked")
        for line in result.get("top", []):
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

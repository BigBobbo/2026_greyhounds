"""Score a committed prediction sheet against scraped results.

Methodology is fixed here BEFORE results exist (pre-registered):

  - Races count only if a winner (finish_position == 1) is recorded.
  - The strict cohort excludes any race whose local start time was at or
    before the sheet's generation timestamp (post-start predictions were
    still made blind — results weren't in the DB — but strictness costs
    nothing and removes the argument).
  - Headline metrics: top-pick hit rate vs the model's own expected hit
    rate (mean top-pick probability), winner log loss vs the
    uniform-field baseline, Brier score vs uniform, and a reliability
    table (predicted-probability buckets vs observed win frequency).

Usage (from backend/):
    DATABASE_URL=... python3 scripts/score_prediction_sheet.py \
        --csv ../docs/predictions/2026-08-01.csv

Import surface: ``score(csv_path)`` returns ``(report_text, summary)``
where summary holds the strict-cohort headline numbers for the running
record. The methodology above is unchanged by the refactor.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal  # noqa: E402
from app.models.race import Race  # noqa: E402
from app.models.race_entry import RaceEntry  # noqa: E402

DUBLIN = ZoneInfo("Europe/Dublin")

BUCKETS = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
           (0.20, 0.30), (0.30, 0.40), (0.40, 1.01)]


def score(csv_path: str):
    """Score a sheet. Returns (report_text, summary_dict)."""
    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return "Empty sheet", {}
    generated_at = None
    if rows[0].get("generated_at"):
        generated_at = datetime.strptime(
            rows[0]["generated_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)

    by_race: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_race[int(r["race_id"])].append(r)

    db = SessionLocal()
    winners: dict[int, int] = {}   # race_id -> winning trap
    for race_id in by_race:
        w = (
            db.query(RaceEntry.trap)
            .filter(RaceEntry.race_id == race_id,
                    RaceEntry.finish_position == 1)
            .first()
        )
        if w and w.trap is not None:
            winners[race_id] = w.trap
    race_rows = {r.id: r for r in db.query(Race).filter(
        Race.id.in_(list(by_race.keys()))).all()}
    db.close()

    def is_strict(race_id: int) -> bool:
        if generated_at is None:
            return True
        race = race_rows.get(race_id)
        if race is None or race.race_time is None:
            return False
        start = datetime.combine(race.race_date, race.race_time,
                                 tzinfo=DUBLIN)
        return start > generated_at

    def report(cohort: str, race_ids: list[int]) -> dict:
        scored = [rid for rid in race_ids if rid in winners]
        if not scored:
            out(f"\n== {cohort}: no races with results yet ==")
            return {}
        hits = 0
        exp_hits = 0.0
        logloss = 0.0
        base_logloss = 0.0
        brier = 0.0
        base_brier = 0.0
        n_runners = 0
        bucket_stats = {b: [0, 0.0] for b in BUCKETS}
        bucket_n = {b: 0 for b in BUCKETS}
        tier_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])

        for rid in scored:
            preds = by_race[rid]
            field = len(preds)
            win_trap = winners[rid]
            probs = {int(p["trap"]): float(p["win_prob"]) for p in preds}
            total = sum(probs.values()) or 1.0
            probs = {t: p / total for t, p in probs.items()}

            top = max(preds, key=lambda p: float(p["win_prob"]))
            top_trap = int(top["trap"])
            exp_hits += probs[top_trap]
            if top_trap == win_trap:
                hits += 1
            tier = top["confidence"] or "?"
            tier_stats[tier][1] += 1
            if top_trap == win_trap:
                tier_stats[tier][0] += 1

            p_win = probs.get(win_trap)
            if p_win is None:
                continue  # winner wasn't in the sheet (reserve ran) — skip
            logloss += -math.log(max(p_win, 1e-9))
            base_logloss += -math.log(1.0 / field)
            for t, p in probs.items():
                won = 1.0 if t == win_trap else 0.0
                brier += (p - won) ** 2
                base_brier += (1.0 / field - won) ** 2
                n_runners += 1
                for b in BUCKETS:
                    if b[0] <= p < b[1]:
                        bucket_n[b] += 1
                        bucket_stats[b][1] += p
                        if won:
                            bucket_stats[b][0] += 1
                        break

        n = len(scored)
        out(f"\n== {cohort}: {n} races scored ==")
        out(f"Top-pick hit rate : {hits}/{n} = {hits / n:.1%} "
            f"(model expected {exp_hits / n:.1%})")
        out(f"Winner log loss   : {logloss / n:.4f} "
            f"(uniform baseline {base_logloss / n:.4f})")
        out(f"Brier / runner    : {brier / n_runners:.4f} "
            f"(uniform baseline {base_brier / n_runners:.4f})")
        out("Reliability (predicted bucket -> observed win rate):")
        for b in BUCKETS:
            cnt = bucket_n[b]
            if not cnt:
                continue
            wins, sump = bucket_stats[b]
            out(f"  {b[0]:>4.0%}–{b[1]:>4.0%}: predicted {sump / cnt:.1%}  "
                f"observed {wins / cnt:.1%}  (n={cnt})")
        out("Top-pick hit rate by confidence tier:")
        for tier in ("strong", "moderate", "weak", "avoid"):
            wins, cnt = tier_stats.get(tier, (0, 0))
            if cnt:
                out(f"  {tier:>8s}: {wins}/{cnt} = {wins / cnt:.1%}")
        return {
            "races": n, "hits": hits,
            "hit_rate": round(hits / n, 4),
            "expected_hit_rate": round(exp_hits / n, 4),
            "log_loss": round(logloss / n, 4),
            "log_loss_uniform": round(base_logloss / n, 4),
            "brier": round(brier / n_runners, 4),
            "brier_uniform": round(base_brier / n_runners, 4),
        }

    all_ids = list(by_race.keys())
    report("ALL RACES", all_ids)
    strict = [rid for rid in all_ids if is_strict(rid)]
    summary = {}
    if generated_at is not None and len(strict) < len(all_ids):
        summary = report(
            f"STRICT PRE-RACE (start after {generated_at:%H:%M} UTC, "
            f"{len(all_ids) - len(strict)} excluded)", strict)
    else:
        # Sheet generated before every race: the full set IS the strict set.
        summary = report("STRICT PRE-RACE (= all races)", strict) \
            if generated_at is not None else {}
    missing = [rid for rid in all_ids if rid not in winners]
    if missing:
        out(f"\n{len(missing)} race(s) still without results.")
    summary["races_missing_results"] = len(missing)
    return "\n".join(lines), summary


def main(csv_path: str) -> None:
    text, _ = score(csv_path)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    main(args.csv)

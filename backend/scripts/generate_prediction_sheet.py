"""Generate a full prediction sheet for a day's cards — every runner, no
betting filter.

Unlike the bet sheet (which lists only stake-worthy picks), this dumps the
model's calibrated win probability, rank, confidence tier and data
completeness for EVERY runner in EVERY scheduled race, so predictions can
be scored against results afterwards. Output is committed markdown + CSV
with a generation timestamp: git history makes the predictions
tamper-evident pre-registrations.

Usage (from backend/):
    DATABASE_URL=... python3 scripts/generate_prediction_sheet.py \
        --experiment-id 1 [--date 2026-08-01] [--out-dir ../docs/predictions]
"""

import argparse
import csv
import os
import sys
from datetime import date as date_cls
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal  # noqa: E402
from app.models.race import Race  # noqa: E402
from app.models.race_entry import RaceEntry  # noqa: E402
from app.models.track import Track  # noqa: E402


def main(experiment_id: int, target_date: date_cls, out_dir: str) -> None:
    from app.services.prediction_service import (
        compute_features_for_entries,
        predict_race,
    )

    db = SessionLocal()
    races = (
        db.query(Race, Track.name.label("track_name"))
        .join(Track, Race.track_id == Track.id)
        .filter(Race.race_date == target_date)
        .filter(Race.status == "scheduled")
        .order_by(Race.race_time.asc().nullslast(), Track.name, Race.race_number)
        .all()
    )
    if not races:
        print(f"No scheduled races for {target_date} — run the card scrape first.")
        return

    race_ids = [race.id for race, _ in races]
    all_entry_ids = [
        e.id for e in db.query(RaceEntry.id)
        .filter(RaceEntry.race_id.in_(race_ids)).all()
    ]
    print(f"Computing features for {len(all_entry_ids)} entries across "
          f"{len(race_ids)} races...", file=sys.stderr)
    features = compute_features_for_entries(
        db, all_entry_ids, [], include_builtin=True, include_elo=True,
    )

    generated_at = datetime.utcnow()
    rows = []
    skipped = 0
    for race, track_name in races:
        try:
            preds = predict_race(db, experiment_id, race.id,
                                 precomputed_features=features)
        except Exception as e:
            skipped += 1
            print(f"  ! race {race.id} ({track_name} R{race.race_number}): {e}",
                  file=sys.stderr)
            continue
        ranked = sorted(
            (p for p in preds if p.get("win_probability") is not None),
            key=lambda p: p["win_probability"], reverse=True,
        )
        for rank, p in enumerate(ranked, 1):
            wp = p["win_probability"]
            rows.append({
                "race_id": race.id,
                "date": str(target_date),
                "time": str(race.race_time or "")[:5],
                "track": track_name,
                "race_no": race.race_number,
                "trap": p["trap"],
                "dog": p["dog_name"],
                "win_prob": round(wp, 4),
                "fair_odds": round(1.0 / wp, 2) if wp > 0 else None,
                "rank": rank,
                "confidence": p.get("confidence_tier"),
                "data_completeness": round(p.get("data_completeness") or 0.0, 3),
                "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{target_date}.csv")
    md_path = os.path.join(out_dir, f"{target_date}.md")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_races = len({r["race_id"] for r in rows})
    lines = [
        f"# Prediction sheet — {target_date}",
        "",
        f"Experiment {experiment_id} · model-only probabilities (no market "
        f"blend — no exchange odds feed yet) · generated "
        f"{generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"{n_races} races, {len(rows)} runners. **Bold** = model's top pick. "
        "Fair odds = 1 / probability: the model calls value anything priced "
        "above it. Races that started before the generation timestamp "
        "should be excluded from strict pre-race scoring.",
        "",
    ]
    current = None
    for r in rows:
        key = (r["time"], r["track"], r["race_no"])
        if key != current:
            current = key
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
        if r["rank"] == max(x["rank"] for x in rows if x["race_id"] == r["race_id"]):
            lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"{n_races} races predicted ({skipped} skipped), {len(rows)} runners")
    print(f"Written: {md_path} and {csv_path}")
    db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", type=int, required=True)
    ap.add_argument("--date", type=lambda s: date_cls.fromisoformat(s),
                    default=date_cls.today())
    ap.add_argument("--out-dir", default=os.path.join("..", "docs", "predictions"))
    args = ap.parse_args()
    main(args.experiment_id, args.date, args.out_dir)

"""Generate the daily bet sheet — odds-conditional instructions.

No live odds feed is required: for each qualifying dog the sheet prints
the MINIMUM decimal odds at which the bet has the required edge, and the
stake to place at that price. The person executing checks the live price
on their betting app and simply skips anything trading below the printed
minimum. This is deliberately robust to the executor's timing: if the
price is bigger than the minimum, the bet is at least as good as sized.

    min_odds(p) = 1 / (p - min_edge), floored at the config's min_odds
    stake       = canonical Kelly at exactly min_odds (conservative:
                  any better price only increases the edge)

Bets are filtered by model confidence tier and capped by the daily
exposure allocator. Output: markdown to stdout and data/bet_sheet_<date>.md.

Usage (from backend/):
    DATABASE_URL=... python3 scripts/generate_bet_sheet.py \
        --experiment-id 12 [--date 2026-08-02]
"""

import argparse
import os
import sys
from datetime import date as date_cls
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal  # noqa: E402
from app.models.race import Race  # noqa: E402
from app.models.track import Track  # noqa: E402


def main(experiment_id: int, target_date: date_cls, min_confidence: str) -> None:
    from dataclasses import replace

    from app.services.prediction_service import predict_race
    from ml.staking import StakingConfig, allocate_daily, kelly_stake

    db = SessionLocal()
    cfg = StakingConfig.from_db(db)

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

    tier_rank = {"strong": 3, "moderate": 2, "weak": 1, "avoid": 0}
    min_tier = tier_rank.get(min_confidence, 2)

    # Compute features ONCE for every entry racing that day — the per-race
    # path would rebuild the full as-of aggregate index per race (minutes
    # each); the shared matrix brings the whole day to one pass.
    from app.models.race_entry import RaceEntry
    from app.services.prediction_service import compute_features_for_entries

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

    candidates = []
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
        # Mirror the BACKTESTED strategy exactly: the model's top pick per
        # race, nothing else. Listing every dog with a theoretical price
        # floor produced 366 micro-bets — technically +EV lines, but not
        # the strategy the evaluation validated, and unexecutable by hand.
        top = None
        for p in preds:
            wp = p.get("win_probability")
            if wp is not None and (top is None or wp > top.get("win_probability", 0)):
                top = p
        for p in ([top] if top else []):
            wp = p.get("win_probability")
            if wp is None or wp <= cfg.min_edge:
                continue
            if tier_rank.get(p.get("confidence_tier", "avoid"), 0) < min_tier:
                continue
            # Minimum acceptable price for the required edge, grossed up so
            # the edge survives commission: at gross odds X the net price is
            # 1 + (X-1)(1-c), and THAT must clear 1/(p - min_edge).
            net_floor = 1.0 / (wp - cfg.min_edge)
            min_odds = 1.0 + (net_floor - 1.0) / (1.0 - cfg.commission_rate)
            min_odds = max(min_odds, cfg.min_odds)
            rec = kelly_stake(wp, min_odds, cfg, completeness=p.get("data_completeness") or 1.0)
            if not rec.get("bet"):
                continue
            candidates.append({
                "time": str(race.race_time or "")[:5],
                "track": track_name,
                "race_no": race.race_number,
                "dog": p["dog_name"],
                "trap": p["trap"],
                "win_prob": round(wp, 3),
                "min_odds": round(min_odds, 2),
                "stake": rec["stake"],
                "confidence": p.get("confidence_tier"),
            })

    candidates.sort(key=lambda c: (c["time"] or "99", c["track"], c["race_no"]))
    candidates = allocate_daily(candidates, cfg)

    lines = [
        f"# Bet sheet — {target_date}",
        "",
        f"Bankroll €{cfg.bankroll:.0f} · quarter-Kelly · day cap "
        f"€{cfg.bankroll * cfg.max_daily_exposure_pct:.0f} · "
        f"experiment {experiment_id}",
        "",
        "**Rule: back WIN only if the live price is AT OR ABOVE the minimum "
        "odds. If it's lower — skip, no exceptions.**",
        "",
        "| Time | Track | Race | Trap | Dog | Min odds | Stake |",
        "|------|-------|------|------|-----|----------|-------|",
    ]
    for c in candidates:
        lines.append(
            f"| {c['time']} | {c['track']} | R{c['race_no']} | {c['trap']} "
            f"| **{c['dog']}** | {c['min_odds']} | €{c['stake']:.2f} |"
        )
    if not candidates:
        lines.append("| — | no qualifying bets today | | | | | |")
    lines += [
        "",
        f"{len(candidates)} bet(s) · total staked if all match: "
        f"€{sum(c['stake'] for c in candidates):.2f} · generated "
        f"{datetime.utcnow():%Y-%m-%d %H:%M} UTC",
    ]

    sheet = "\n".join(lines)
    out = os.path.join("data", f"bet_sheet_{target_date}.md")
    os.makedirs("data", exist_ok=True)
    with open(out, "w") as f:
        f.write(sheet + "\n")
    print(sheet)
    print(f"\nWritten to {out}", file=sys.stderr)
    db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", type=int, required=True)
    ap.add_argument("--date", type=lambda s: date_cls.fromisoformat(s),
                    default=date_cls.today())
    ap.add_argument("--min-confidence", default="moderate")
    args = ap.parse_args()
    main(args.experiment_id, args.date, args.min_confidence)

"""Generate the live bet sheet — model blended with live exchange prices.

This is the sheet the whole market layer was built for. Its predecessor,
``generate_bet_sheet.py``, prints odds-conditional instructions ("back if
the price is at or above X") because there was no price feed; it stakes
off raw model probabilities alone. The honest evaluation showed that is
the weaker half of the system: the fitted Benter blend put beta = 1.12 on
the market against alpha = 0.71 on the model, meaning the market knows
things the model does not, and value comes from where the two disagree —
not from the model's opinion in isolation.

With the Betfair feed running, each race here goes:

    model probabilities  ─┐
                          ├─> blend (alpha, beta from the trained bundle)
    de-vigged exchange   ─┘        │
    book                           v
                            joint Kelly against the ACTUAL back price
                            (mutually exclusive outcomes solved together)
                                   │
                                   v
                            daily exposure cap

Races whose exchange book is incomplete or stale fall back to model-only
probabilities and are labelled as such in the output — never silently
blended against a half-book.

Usage (from backend/):
    DATABASE_URL=... python3 scripts/generate_live_bet_sheet.py \
        --experiment-id 12 [--date 2026-08-18] [--min-confidence moderate]
"""

import argparse
import csv
import os
import sys
from datetime import date as date_cls
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal  # noqa: E402
from app.models.experiment import Experiment  # noqa: E402
from app.models.race import Race  # noqa: E402
from app.models.race_entry import RaceEntry  # noqa: E402
from app.models.track import Track  # noqa: E402

TIER_RANK = {"strong": 3, "moderate": 2, "weak": 1, "avoid": 0}


def blend_params(db, experiment_id: int) -> tuple[float, float]:
    """alpha/beta for the blend, from the experiment's stored metrics or,
    failing that, the model artifact itself.

    Refuses to guess: a missing blend is a hard error rather than a
    silent alpha=1/beta=0 fallback, which would quietly ship a model-only
    sheet under a blended-sheet heading.
    """
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if exp is None:
        raise SystemExit(f"experiment {experiment_id} not found")

    metrics = exp.metrics or {}
    alpha, beta = metrics.get("blend_alpha"), metrics.get("blend_beta")
    if alpha is None or beta is None:
        path = exp.model_path
        if path and os.path.exists(path):
            import joblib
            bundle = joblib.load(path)
            alpha = bundle.get("blend_alpha", alpha)
            beta = bundle.get("blend_beta", beta)
    if alpha is None or beta is None:
        raise SystemExit(
            f"experiment {experiment_id} has no fitted blend (alpha/beta). "
            "Refit with scripts/local_retrain_eval.py, or use "
            "generate_bet_sheet.py for the model-only odds-conditional sheet."
        )
    return float(alpha), float(beta)


def main(experiment_id: int, target_date: date_cls, min_confidence: str,
         max_age_minutes: int) -> int:
    from app.services.prediction_service import (
        compute_features_for_entries, predict_race,
    )
    from ml.market import blend_race, devig_book, latest_prices_for_races
    from ml.staking import StakingConfig, allocate_daily, race_kelly

    db = SessionLocal()
    cfg = StakingConfig.from_db(db)
    alpha, beta = blend_params(db, experiment_id)
    min_tier = TIER_RANK.get(min_confidence, 2)

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
        return 1

    race_ids = [race.id for race, _ in races]
    entries = (
        db.query(RaceEntry.id, RaceEntry.race_id, RaceEntry.dog_id)
        .filter(RaceEntry.race_id.in_(race_ids)).all()
    )
    entry_to_dog = {e.id: e.dog_id for e in entries}
    runners_per_race: dict[int, int] = {}
    for e in entries:
        runners_per_race[e.race_id] = runners_per_race.get(e.race_id, 0) + 1

    prices = latest_prices_for_races(
        db, race_ids, max_age_minutes=max_age_minutes,
    )
    priced_races = len(prices)
    print(f"Exchange prices found for {priced_races}/{len(race_ids)} races "
          f"(max age {max_age_minutes} min)", file=sys.stderr)

    # One shared feature matrix for the whole day; the per-race path would
    # rebuild the as-of aggregate index per race.
    print(f"Computing features for {len(entries)} entries across "
          f"{len(race_ids)} races...", file=sys.stderr)
    features = compute_features_for_entries(
        db, [e.id for e in entries], [], include_builtin=True, include_elo=True,
    )

    bets: list[dict] = []
    rows: list[dict] = []
    skipped = 0
    blended_races = 0

    for race, track_name in races:
        try:
            preds = predict_race(db, experiment_id, race.id,
                                 precomputed_features=features)
        except Exception as e:
            skipped += 1
            print(f"  ! race {race.id} ({track_name} R{race.race_number}): {e}",
                  file=sys.stderr)
            continue

        model_by_entry = {
            p["race_entry_id"]: p["win_probability"] for p in preds
            if p.get("win_probability") is not None
        }
        if not model_by_entry:
            continue

        # Market book, keyed to the same entry ids as the model output.
        race_prices = prices.get(race.id, {})
        book_by_entry = {
            eid: race_prices[entry_to_dog[eid]]["odds"]
            for eid in model_by_entry
            if entry_to_dog.get(eid) in race_prices
        }
        market = devig_book(
            book_by_entry, expected_runners=runners_per_race.get(race.id),
        )
        probs, blended = blend_race(model_by_entry, market, alpha, beta)
        if blended:
            blended_races += 1

        by_entry = {p["race_entry_id"]: p for p in preds}
        candidates = [
            {
                "id": eid,
                "win_prob": probs[eid],
                "odds_decimal": book_by_entry.get(eid),
                "completeness": by_entry[eid].get("data_completeness") or 1.0,
            }
            for eid in probs
            if book_by_entry.get(eid)  # no price = nothing to bet into
        ]
        # Only stake races where the blend actually happened. Falling back
        # to model-only probabilities but still betting them would quietly
        # mix two strategies in one sheet — and model-only is the half the
        # evaluation showed to be weaker. Those races stay in the CSV for
        # inspection; they just don't get money.
        staked = race_kelly(candidates, cfg) if (candidates and blended) else {}

        for c in candidates:
            p = by_entry[c["id"]]
            rec = staked.get(c["id"], {})
            row = {
                "time": str(race.race_time or "")[:5],
                "track": track_name,
                "race_no": race.race_number,
                "trap": p["trap"],
                "dog": p["dog_name"],
                "model_prob": round(model_by_entry[c["id"]], 4),
                "market_prob": round(market[c["id"]], 4) if market else None,
                "blend_prob": round(c["win_prob"], 4),
                "price": c["odds_decimal"],
                "edge": rec.get("edge"),
                "stake": rec.get("stake") if rec.get("bet") else 0.0,
                "blended": blended,
                "confidence": p.get("confidence_tier"),
            }
            rows.append(row)
            if not rec.get("bet"):
                continue
            if TIER_RANK.get(p.get("confidence_tier", "avoid"), 0) < min_tier:
                continue
            bets.append(row)

    bets.sort(key=lambda b: (b["time"] or "99", b["track"], b["race_no"]))
    bets = allocate_daily(bets, cfg)

    generated_at = datetime.utcnow()
    lines = [
        f"# Live bet sheet — {target_date}",
        "",
        f"Experiment {experiment_id} · Benter blend alpha={alpha:.3f} "
        f"beta={beta:.3f} · bankroll €{cfg.bankroll:.0f} · "
        f"{cfg.kelly_fraction:.2f}-Kelly · day cap "
        f"€{cfg.bankroll * cfg.max_daily_exposure_pct:.0f} · generated "
        f"{generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"Exchange book usable in {blended_races}/{len(race_ids)} races; the "
        "rest fell back to model-only probabilities and are excluded from "
        "staking (no live price = nothing to bet into).",
        "",
        "**Prices move. Back at or above the listed price; if it has "
        "shortened below it, skip the bet.**",
        "",
        "| Time | Track | Race | Trap | Dog | Model | Market | Blend | Price | Edge | Stake |",
        "|------|-------|------|------|-----|-------|--------|-------|-------|------|-------|",
    ]
    for b in bets:
        lines.append(
            f"| {b['time']} | {b['track']} | R{b['race_no']} | {b['trap']} "
            f"| **{b['dog']}** | {b['model_prob']:.1%} "
            f"| {b['market_prob']:.1%} | {b['blend_prob']:.1%} "
            f"| {b['price']} | {b['edge']:+.3f} | €{b['stake']:.2f} |"
        )
    if not bets:
        lines.append("| — | no qualifying bets | | | | | | | | | |")
    lines += [
        "",
        f"{len(bets)} bet(s) · total staked €{sum(b['stake'] for b in bets):.2f} "
        f"· {skipped} race(s) skipped on prediction errors",
    ]

    sheet = "\n".join(lines)
    os.makedirs("data", exist_ok=True)
    md_path = os.path.join("data", f"live_bet_sheet_{target_date}.md")
    with open(md_path, "w") as f:
        f.write(sheet + "\n")
    if rows:
        csv_path = os.path.join("data", f"live_bet_sheet_{target_date}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Full candidate table written to {csv_path}", file=sys.stderr)

    print(sheet)
    print(f"\nWritten to {md_path}", file=sys.stderr)
    db.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", type=int, required=True)
    ap.add_argument("--date", type=lambda s: date_cls.fromisoformat(s),
                    default=date_cls.today())
    ap.add_argument("--min-confidence", default="moderate")
    ap.add_argument("--max-age-minutes", type=int, default=45,
                    help="ignore exchange snapshots older than this")
    args = ap.parse_args()
    raise SystemExit(main(args.experiment_id, args.date, args.min_confidence,
                          args.max_age_minutes))

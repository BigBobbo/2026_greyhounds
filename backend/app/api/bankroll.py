"""Bankroll management API: track bets, manage bankroll, view history."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.bankroll import BankrollConfig, BetRecord
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.dog import Dog
from app.models.track import Track

router = APIRouter(prefix="/bankroll", tags=["bankroll"])


# --- Schemas ---

class BankrollConfigUpdate(BaseModel):
    initial_bankroll: float | None = None
    current_bankroll: float | None = None
    kelly_fraction: float | None = None
    min_edge: float | None = None
    max_stake_pct: float | None = None
    commission_rate: float | None = None
    min_odds: float | None = None
    max_daily_exposure_pct: float | None = None


class PlaceBetRequest(BaseModel):
    race_entry_id: int
    experiment_id: int
    win_probability: float
    odds_decimal: float | None = None
    edge: float | None = None
    confidence_tier: str | None = None
    stake: float
    stake_method: str = "kelly"
    # Combo betting (optional). When `bet_type` is "forecast" or "trio",
    # `legs` is the ordered list of race_entry_ids backing the bet
    # ([1st, 2nd] or [1st, 2nd, 3rd]) and `combo_probability` is the
    # model's combo P. The single `race_entry_id` field above stays as
    # the primary leg so the existing race-detail join logic still works.
    bet_type: str = "win"
    legs: list[int] | None = None
    combo_probability: float | None = None


class SettleBetRequest(BaseModel):
    # Preferred: the explicit outcome. The old UI encoded "lost" as
    # actual_position=2, which the place-bet rule (position <= 2 wins)
    # read as a WIN — clicking "Lost" on a place bet credited the
    # bankroll. An explicit result cannot be misread.
    result: str | None = None  # "won" | "lost" | "void"
    # Alternative: the dog's real finishing position; the handler derives
    # won/lost from the bet type.
    actual_position: int | None = None
    # Forecast/trio: the ordered list of finishing race_entry_ids the
    # race actually produced. The handler compares this against the
    # bet's `legs_json` to decide whether the combo hit.
    actual_finishing_order: list[int] | None = None


# Standard each-way place terms at Irish greyhound tracks: places pay a
# quarter of the win odds. A place bet's profit is stake * (odds-1) * 1/4,
# not the full win odds the old code paid.
PLACE_TERMS_FRACTION = 0.25


def _placings_for_field(num_runners: int | None) -> int:
    """Number of paid places for a field. Six-dog greyhound races pay 2;
    smaller fields typically pay win-only but we keep 2 as the floor the
    UI expects. Show bets always use 3."""
    if num_runners is not None and num_runners < 5:
        return 1
    return 2


def _apply_settlement(
    record: BetRecord,
    config: BankrollConfig | None,
    *,
    won: bool,
    void: bool = False,
) -> None:
    """Book a settlement onto the record and the bankroll ledger.

    Stake was already deducted at placement, so: void returns the stake;
    a win returns stake + profit; a loss moves nothing further.
    """
    if void:
        record.outcome = "void"
        record.profit = 0.0
    elif won:
        odds = record.odds_decimal
        bet_type = (record.bet_type or "win").lower()
        if bet_type in ("place", "show"):
            profit = record.stake * (odds - 1) * PLACE_TERMS_FRACTION
        else:
            profit = record.stake * (odds - 1)
        record.outcome = "won"
        record.profit = round(profit, 2)
    else:
        record.outcome = "lost"
        record.profit = -record.stake

    record.settled_at = datetime.utcnow()
    if config:
        if record.outcome == "void":
            config.current_bankroll += record.stake
        elif record.outcome == "won":
            config.current_bankroll += record.stake + record.profit
        record.bankroll_after = config.current_bankroll


# --- Config ---

@router.get("/config")
def get_config(db: Session = Depends(get_db)):
    """Get the current bankroll configuration. Creates default if none exists."""
    config = db.query(BankrollConfig).first()
    if not config:
        config = BankrollConfig(initial_bankroll=100.0, current_bankroll=100.0)
        db.add(config)
        db.commit()
        db.refresh(config)
    return {
        "id": config.id,
        "initial_bankroll": config.initial_bankroll,
        "current_bankroll": config.current_bankroll,
        "kelly_fraction": config.kelly_fraction,
        "min_edge": config.min_edge,
        "max_stake_pct": config.max_stake_pct,
        "commission_rate": getattr(config, "commission_rate", 0.05),
        "min_odds": getattr(config, "min_odds", 1.5),
        "max_daily_exposure_pct": getattr(config, "max_daily_exposure_pct", 0.10),
    }


@router.put("/config")
def update_config(update: BankrollConfigUpdate, db: Session = Depends(get_db)):
    """Update bankroll configuration."""
    config = db.query(BankrollConfig).first()
    if not config:
        config = BankrollConfig()
        db.add(config)

    if update.initial_bankroll is not None:
        config.initial_bankroll = update.initial_bankroll
    if update.current_bankroll is not None:
        config.current_bankroll = update.current_bankroll
    if update.kelly_fraction is not None:
        config.kelly_fraction = update.kelly_fraction
    if update.min_edge is not None:
        config.min_edge = update.min_edge
    if update.max_stake_pct is not None:
        config.max_stake_pct = update.max_stake_pct
    if update.commission_rate is not None:
        config.commission_rate = update.commission_rate
    if update.min_odds is not None:
        config.min_odds = update.min_odds
    if update.max_daily_exposure_pct is not None:
        config.max_daily_exposure_pct = update.max_daily_exposure_pct

    db.commit()
    db.refresh(config)
    return {
        "id": config.id,
        "initial_bankroll": config.initial_bankroll,
        "current_bankroll": config.current_bankroll,
        "kelly_fraction": config.kelly_fraction,
        "min_edge": config.min_edge,
        "max_stake_pct": config.max_stake_pct,
        "commission_rate": getattr(config, "commission_rate", 0.05),
        "min_odds": getattr(config, "min_odds", 1.5),
        "max_daily_exposure_pct": getattr(config, "max_daily_exposure_pct", 0.10),
    }


@router.post("/reset")
def reset_bankroll(db: Session = Depends(get_db)):
    """Reset bankroll to initial value and clear all bet records."""
    config = db.query(BankrollConfig).first()
    if config:
        config.current_bankroll = config.initial_bankroll
    db.query(BetRecord).delete()
    db.commit()
    return {"status": "reset", "bankroll": config.current_bankroll if config else 100.0}


# --- Bets ---

@router.post("/bets")
def place_bet(bet: PlaceBetRequest, db: Session = Depends(get_db)):
    """Record a new bet."""
    config = db.query(BankrollConfig).first()
    if not config:
        config = BankrollConfig(initial_bankroll=100.0, current_bankroll=100.0)
        db.add(config)
        db.commit()
        db.refresh(config)

    # Get race info
    entry = (
        db.query(
            RaceEntry, Dog.name.label("dog_name"),
            Race.race_date, Race.race_number, Race.grade,
            Track.name.label("track_name"),
        )
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Track, Race.track_id == Track.id)
        .filter(RaceEntry.id == bet.race_entry_id)
        .first()
    )

    if not entry:
        raise HTTPException(status_code=404, detail="Race entry not found")

    implied_prob = (1.0 / bet.odds_decimal) if bet.odds_decimal and bet.odds_decimal > 1 else None

    # Combo bets: validate the legs list shape and serialise it for
    # storage. Single-dog bets ("win", "place", "show") leave legs null.
    bet_type = bet.bet_type or "win"
    legs_json: str | None = None
    if bet_type in ("forecast", "trio"):
        if not bet.legs:
            raise HTTPException(
                status_code=422,
                detail=f"{bet_type} bets require a `legs` list",
            )
        expected = 2 if bet_type == "forecast" else 3
        if len(bet.legs) != expected:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{bet_type} bets need {expected} legs, got {len(bet.legs)}"
                ),
            )
        if len(set(bet.legs)) != len(bet.legs):
            raise HTTPException(
                status_code=422, detail="legs must be distinct race_entry_ids",
            )
        if bet.race_entry_id != bet.legs[0]:
            # The primary join column (race_entry_id) is conventionally
            # the winning leg so race-detail joins still surface combo
            # bets next to the dog the model thinks finishes 1st.
            raise HTTPException(
                status_code=422,
                detail="race_entry_id must equal the first leg for combo bets",
            )
        import json as _json
        legs_json = _json.dumps(bet.legs)

    record = BetRecord(
        race_entry_id=bet.race_entry_id,
        experiment_id=bet.experiment_id,
        dog_name=entry.dog_name,
        track_name=entry.track_name,
        race_date=str(entry.race_date),
        race_number=entry.race_number,
        trap=entry.RaceEntry.trap,
        grade=entry.grade,
        bet_type=bet_type,
        legs_json=legs_json,
        combo_probability=bet.combo_probability,
        win_probability=bet.win_probability,
        odds_decimal=bet.odds_decimal,
        implied_prob=implied_prob,
        edge=bet.edge,
        confidence_tier=bet.confidence_tier,
        stake=bet.stake,
        stake_method=bet.stake_method,
        bankroll_before=config.current_bankroll,
        outcome="pending",
    )
    db.add(record)

    # Deduct stake from bankroll
    config.current_bankroll -= bet.stake

    db.commit()
    db.refresh(record)
    return {"id": record.id, "status": "placed", "bankroll": config.current_bankroll}


@router.post("/bets/{bet_id}/settle")
def settle_bet(bet_id: int, settle: SettleBetRequest, db: Session = Depends(get_db)):
    """Settle a pending bet with the actual result."""
    record = db.query(BetRecord).filter(BetRecord.id == bet_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Bet not found")
    if record.outcome != "pending":
        raise HTTPException(status_code=400, detail="Bet already settled")

    config = db.query(BankrollConfig).first()
    bet_type = (record.bet_type or "win").lower()

    if settle.result is not None:
        result = settle.result.lower()
        if result not in ("won", "lost", "void"):
            raise HTTPException(
                status_code=422,
                detail="result must be one of 'won', 'lost', 'void'",
            )
        if result == "void":
            _apply_settlement(record, config, won=False, void=True)
        elif result == "lost":
            _apply_settlement(record, config, won=False)
        else:
            if not record.odds_decimal or record.odds_decimal <= 1.0:
                # The old code silently booked NULL-odds winners as LOSSES.
                # A win cannot be paid without a price — make the caller
                # supply one instead of corrupting the ledger.
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Bet has no odds_decimal — a win cannot be settled "
                        "without a price. Update the bet's odds first."
                    ),
                )
            _apply_settlement(record, config, won=True)
    else:
        won = False
        if bet_type == "win":
            if settle.actual_position is None:
                raise HTTPException(
                    status_code=422,
                    detail="Provide `result` or `actual_position`",
                )
            won = settle.actual_position == 1
        elif bet_type in ("place", "show"):
            if settle.actual_position is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Place/show bets need the dog's real finishing "
                        "position (or an explicit `result`)"
                    ),
                )
            limit = 3 if bet_type == "show" else 2
            won = settle.actual_position <= limit
        elif bet_type in ("forecast", "trio"):
            # Combo bets need the actual finishing order — compare it leg by
            # leg to the legs the bet was placed on. For a forecast the
            # first two finishers must match in order; for a trio the first
            # three. Reverse forecasts (CB) aren't a separate bet here —
            # they'd be staked as two separate forecast records.
            if not settle.actual_finishing_order:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"{bet_type} bets must be settled with an "
                        f"`actual_finishing_order` list (or explicit `result`)"
                    ),
                )
            import json as _json
            legs = _json.loads(record.legs_json) if record.legs_json else []
            n = 2 if bet_type == "forecast" else 3
            if len(settle.actual_finishing_order) >= n and len(legs) == n:
                won = list(settle.actual_finishing_order[:n]) == list(legs)

        if won and (not record.odds_decimal or record.odds_decimal <= 1.0):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Bet has no odds_decimal — a win cannot be settled "
                    "without a price. Update the bet's odds first."
                ),
            )
        _apply_settlement(record, config, won=won)

    db.commit()
    return {
        "id": record.id,
        "outcome": record.outcome,
        "profit": record.profit,
        "bankroll": config.current_bankroll if config else None,
    }


@router.post("/reconcile")
def reconcile_bets(db: Session = Depends(get_db)):
    """Auto-settle pending bets against the scraped results already in the
    database. A pending bet whose race has resulted is settled from the
    entry's real finishing position; a dog with no finishing position in a
    resulted race is treated as a non-runner and the bet voided (stake
    refunded). Combo bets settle against the race's actual finishing order.
    """
    import json as _json

    config = db.query(BankrollConfig).first()
    pending = db.query(BetRecord).filter(BetRecord.outcome == "pending").all()

    settled, skipped = [], 0
    for record in pending:
        entry = (
            db.query(RaceEntry).filter(RaceEntry.id == record.race_entry_id).first()
        )
        if not entry:
            skipped += 1
            continue
        race = db.query(Race).filter(Race.id == entry.race_id).first()
        if not race or race.status != "resulted":
            skipped += 1
            continue

        bet_type = (record.bet_type or "win").lower()

        if bet_type in ("forecast", "trio"):
            legs = _json.loads(record.legs_json) if record.legs_json else []
            n = 2 if bet_type == "forecast" else 3
            finishers = (
                db.query(RaceEntry)
                .filter(
                    RaceEntry.race_id == race.id,
                    RaceEntry.finish_position.isnot(None),
                )
                .order_by(RaceEntry.finish_position)
                .all()
            )
            order = [e.id for e in finishers]
            if len(order) < n or len(legs) != n:
                skipped += 1
                continue
            won = order[:n] == list(legs)
            if won and (not record.odds_decimal or record.odds_decimal <= 1.0):
                skipped += 1  # needs a price before it can be paid
                continue
            _apply_settlement(record, config, won=won)
        else:
            pos = entry.finish_position
            if pos is None:
                # Resulted race, no position for this dog: non-runner.
                _apply_settlement(record, config, won=False, void=True)
            else:
                if bet_type == "win":
                    won = pos == 1
                elif bet_type == "show":
                    won = pos <= 3
                else:  # place
                    won = pos <= _placings_for_field(race.num_runners)
                if won and (not record.odds_decimal or record.odds_decimal <= 1.0):
                    skipped += 1
                    continue
                _apply_settlement(record, config, won=won)

        settled.append({
            "id": record.id,
            "dog_name": record.dog_name,
            "outcome": record.outcome,
            "profit": record.profit,
        })

    db.commit()
    return {
        "settled": settled,
        "settled_count": len(settled),
        "still_pending": skipped,
        "bankroll": config.current_bankroll if config else None,
    }


@router.delete("/bets/{bet_id}")
def delete_bet(bet_id: int, db: Session = Depends(get_db)):
    """Delete a pending bet and refund the stake."""
    record = db.query(BetRecord).filter(BetRecord.id == bet_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Bet not found")

    config = db.query(BankrollConfig).first()
    if record.outcome == "pending" and config:
        config.current_bankroll += record.stake

    db.delete(record)
    db.commit()
    return {"status": "deleted", "bankroll": config.current_bankroll if config else None}


@router.get("/bets")
def list_bets(
    status: str | None = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db),
):
    """List all bet records, optionally filtered by status."""
    query = db.query(BetRecord)
    if status:
        query = query.filter(BetRecord.outcome == status)
    bets = query.order_by(BetRecord.created_at.desc()).limit(limit).all()

    import json as _json
    out: list[dict[str, Any]] = []
    for b in bets:
        legs: list[int] | None = None
        if b.legs_json:
            try:
                legs = _json.loads(b.legs_json)
            except (ValueError, TypeError):
                legs = None
        out.append({
            "id": b.id,
            "race_entry_id": b.race_entry_id,
            "experiment_id": b.experiment_id,
            "dog_name": b.dog_name,
            "track_name": b.track_name,
            "race_date": b.race_date,
            "race_number": b.race_number,
            "trap": b.trap,
            "grade": b.grade,
            "bet_type": b.bet_type or "win",
            "legs": legs,
            "combo_probability": b.combo_probability,
            "win_probability": b.win_probability,
            "odds_decimal": b.odds_decimal,
            "edge": b.edge,
            "confidence_tier": b.confidence_tier,
            "stake": b.stake,
            "stake_method": b.stake_method,
            "bankroll_before": b.bankroll_before,
            "outcome": b.outcome,
            "profit": b.profit,
            "bankroll_after": b.bankroll_after,
            "settled_at": str(b.settled_at) if b.settled_at else None,
            "created_at": str(b.created_at) if b.created_at else None,
        })
    return out


@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    """Get bankroll summary stats."""
    config = db.query(BankrollConfig).first()
    if not config:
        config = BankrollConfig(initial_bankroll=100.0, current_bankroll=100.0)
        db.add(config)
        db.commit()
        db.refresh(config)

    total_bets = db.query(func.count(BetRecord.id)).scalar() or 0
    settled_bets = db.query(func.count(BetRecord.id)).filter(BetRecord.outcome != "pending").scalar() or 0
    pending_bets = db.query(func.count(BetRecord.id)).filter(BetRecord.outcome == "pending").scalar() or 0
    wins = db.query(func.count(BetRecord.id)).filter(BetRecord.outcome == "won").scalar() or 0
    losses = db.query(func.count(BetRecord.id)).filter(BetRecord.outcome == "lost").scalar() or 0
    total_profit = db.query(func.sum(BetRecord.profit)).filter(BetRecord.outcome != "pending").scalar() or 0
    total_staked = db.query(func.sum(BetRecord.stake)).filter(BetRecord.outcome != "pending").scalar() or 0

    # Cumulative P&L over time
    settled = (
        db.query(BetRecord)
        .filter(BetRecord.outcome != "pending")
        .order_by(BetRecord.settled_at)
        .all()
    )
    cumulative_pnl = []
    running = 0.0
    for b in settled:
        running += (b.profit or 0)
        cumulative_pnl.append({
            "bet": len(cumulative_pnl) + 1,
            "pnl": round(running, 2),
            "date": b.race_date,
            "dog": b.dog_name,
        })

    # Current streak
    streak_type = None
    streak_count = 0
    for b in reversed(settled):
        if streak_type is None:
            streak_type = b.outcome
            streak_count = 1
        elif b.outcome == streak_type:
            streak_count += 1
        else:
            break

    return {
        "initial_bankroll": config.initial_bankroll,
        "current_bankroll": round(config.current_bankroll, 2),
        "total_pnl": round(float(total_profit), 2),
        "total_pnl_pct": round(float(total_profit) / config.initial_bankroll * 100, 1) if config.initial_bankroll > 0 else 0,
        "roi": round(float(total_profit) / max(float(total_staked), 1) * 100, 1),
        "total_bets": total_bets,
        "settled_bets": settled_bets,
        "pending_bets": pending_bets,
        "wins": wins,
        "losses": losses,
        # Voids are settled but neither won nor lost — exclude them from
        # the strike-rate denominator.
        "strike_rate": round(wins / max(wins + losses, 1) * 100, 1),
        "total_staked": round(float(total_staked), 2),
        "avg_stake": round(float(total_staked) / max(settled_bets, 1), 2),
        "streak": f"{streak_count} {'win' if streak_type == 'won' else 'loss'}{'es' if streak_count != 1 and streak_type == 'lost' else 's' if streak_count != 1 else ''}" if streak_type else "N/A",
        "streak_type": streak_type,
        "cumulative_pnl": cumulative_pnl,
    }

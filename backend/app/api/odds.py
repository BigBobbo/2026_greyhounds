"""Betting odds API endpoints — BSP import and live odds."""

import asyncio
import logging
import traceback
from datetime import date, datetime, timedelta
from threading import Thread
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.dog import Dog
from app.models.odds import OddsSnapshot
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.scrape_log import ScrapeLog
from app.models.track import Track

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/odds", tags=["odds"])


# ── Response schemas ──────────────────────────────────────────────────


class OddsSnapshotResponse(BaseModel):
    id: int
    race_id: int
    dog_id: int
    dog_name: str | None = None
    bookmaker: str
    odds_decimal: float
    implied_prob: float | None
    scraped_at: datetime
    is_sp: bool

    model_config = {"from_attributes": True}


class OddsStatusResponse(BaseModel):
    total_snapshots: int
    bsp_snapshots: int
    live_snapshots: int
    races_with_odds: int
    dogs_with_odds: int
    oldest_snapshot: datetime | None
    newest_snapshot: datetime | None


class BspImportRequest(BaseModel):
    start_date: str
    end_date: str
    irish_only: bool = True


class BspImportResponse(BaseModel):
    message: str
    log_id: int | None = None


class LiveOddsRequest(BaseModel):
    hours_ahead: int = 24
    irish_only: bool = True


class RaceOddsResponse(BaseModel):
    race_id: int
    race_date: date
    track_name: str | None = None
    race_number: int | None = None
    distance_m: int | None = None
    grade: str | None = None
    odds: list[OddsSnapshotResponse] = []


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get("/status", response_model=OddsStatusResponse)
def odds_status(db: Session = Depends(get_db)):
    """Overview of odds data in the database."""
    total = db.query(OddsSnapshot).count()
    bsp = db.query(OddsSnapshot).filter(OddsSnapshot.is_sp.is_(True)).count()
    live = total - bsp
    races_with = db.query(func.count(func.distinct(OddsSnapshot.race_id))).scalar()
    dogs_with = db.query(func.count(func.distinct(OddsSnapshot.dog_id))).scalar()

    oldest = db.query(func.min(OddsSnapshot.scraped_at)).scalar()
    newest = db.query(func.max(OddsSnapshot.scraped_at)).scalar()

    return OddsStatusResponse(
        total_snapshots=total,
        bsp_snapshots=bsp,
        live_snapshots=live,
        races_with_odds=races_with or 0,
        dogs_with_odds=dogs_with or 0,
        oldest_snapshot=oldest,
        newest_snapshot=newest,
    )


@router.get("/race/{race_id}", response_model=RaceOddsResponse)
def get_race_odds(race_id: int, db: Session = Depends(get_db)):
    """Get all odds snapshots for a specific race."""
    race = db.query(Race).filter(Race.id == race_id).first()
    if not race:
        return RaceOddsResponse(race_id=race_id, race_date=date.today())

    track = db.query(Track).filter(Track.id == race.track_id).first()

    snapshots = (
        db.query(OddsSnapshot)
        .filter(OddsSnapshot.race_id == race_id)
        .order_by(OddsSnapshot.scraped_at.desc())
        .all()
    )

    odds_list = []
    for snap in snapshots:
        dog = db.query(Dog).filter(Dog.id == snap.dog_id).first()
        odds_list.append(OddsSnapshotResponse(
            id=snap.id,
            race_id=snap.race_id,
            dog_id=snap.dog_id,
            dog_name=dog.name if dog else None,
            bookmaker=snap.bookmaker,
            odds_decimal=snap.odds_decimal,
            implied_prob=snap.implied_prob,
            scraped_at=snap.scraped_at,
            is_sp=snap.is_sp,
        ))

    return RaceOddsResponse(
        race_id=race_id,
        race_date=race.race_date,
        track_name=track.name if track else None,
        race_number=race.race_number,
        distance_m=race.distance_m,
        grade=race.grade,
        odds=odds_list,
    )


@router.get("/dog/{dog_id}", response_model=list[OddsSnapshotResponse])
def get_dog_odds(
    dog_id: int,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db),
):
    """Get odds history for a specific dog."""
    snapshots = (
        db.query(OddsSnapshot)
        .filter(OddsSnapshot.dog_id == dog_id)
        .order_by(OddsSnapshot.scraped_at.desc())
        .limit(limit)
        .all()
    )

    dog = db.query(Dog).filter(Dog.id == dog_id).first()
    dog_name = dog.name if dog else None

    return [
        OddsSnapshotResponse(
            id=s.id,
            race_id=s.race_id,
            dog_id=s.dog_id,
            dog_name=dog_name,
            bookmaker=s.bookmaker,
            odds_decimal=s.odds_decimal,
            implied_prob=s.implied_prob,
            scraped_at=s.scraped_at,
            is_sp=s.is_sp,
        )
        for s in snapshots
    ]


@router.post("/import-bsp", response_model=BspImportResponse)
def import_bsp(req: BspImportRequest, db: Session = Depends(get_db)):
    """
    Import Betfair BSP historical data for a date range.

    Downloads CSV files from promo.betfair.com/betfairsp/prices and matches
    records to existing races/dogs in the database.
    """
    start = date.fromisoformat(req.start_date)
    end = date.fromisoformat(req.end_date)

    if (end - start).days > 365:
        return BspImportResponse(message="Date range too large — max 365 days per request")

    log = ScrapeLog(
        spider_name="betfair_bsp",
        source=f"BSP import {start} to {end}",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    log_id = log.id

    def _run_import():
        from scraping.betfair_bsp import fetch_bsp_date_range
        from scraping.odds_pipeline import upsert_bsp_odds

        db_import = SessionLocal()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            records = loop.run_until_complete(
                fetch_bsp_date_range(start, end, irish_only=req.irish_only)
            )
            loop.close()

            stats = upsert_bsp_odds(db_import, records)

            log_entry = db_import.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_entry:
                log_entry.status = "success"
                log_entry.records_scraped = stats["matched"]
                log_entry.records_new = stats["inserted"]
                log_entry.completed_at = datetime.utcnow()
                db_import.commit()

        except Exception as e:
            logger.error("BSP import failed: %s\n%s", e, traceback.format_exc())
            log_entry = db_import.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_entry:
                log_entry.status = "failed"
                log_entry.error_message = f"{type(e).__name__}: {e}"
                log_entry.completed_at = datetime.utcnow()
                db_import.commit()
        finally:
            db_import.close()

    Thread(target=_run_import, daemon=True).start()

    return BspImportResponse(
        message=f"BSP import started for {start} to {end}",
        log_id=log_id,
    )


@router.post("/fetch-live", response_model=BspImportResponse)
def fetch_live_odds(req: LiveOddsRequest, db: Session = Depends(get_db)):
    """
    Fetch current live odds from the Betfair Exchange.

    Requires Betfair API credentials to be configured in .env.
    """
    from scraping.betfair_exchange import BetfairClient

    client = BetfairClient()
    if not client.is_configured():
        return BspImportResponse(
            message="Betfair API credentials not configured. "
            "Set BETFAIR_API_KEY, BETFAIR_USERNAME, and BETFAIR_PASSWORD in .env"
        )

    log = ScrapeLog(
        spider_name="betfair_exchange",
        source=f"Live odds fetch (next {req.hours_ahead}h)",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    log_id = log.id

    def _run_fetch():
        from scraping.odds_pipeline import upsert_live_odds

        db_fetch = SessionLocal()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _do_fetch():
                bf = BetfairClient()
                logged_in = await bf.login()
                if not logged_in:
                    raise RuntimeError("Betfair login failed")
                try:
                    return await bf.fetch_live_odds(req.hours_ahead, req.irish_only)
                finally:
                    await bf.close()

            records = loop.run_until_complete(_do_fetch())
            loop.close()

            stats = upsert_live_odds(db_fetch, records)

            log_entry = db_fetch.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_entry:
                log_entry.status = "success"
                log_entry.records_scraped = stats["matched"]
                log_entry.records_new = stats["inserted"]
                log_entry.completed_at = datetime.utcnow()
                db_fetch.commit()

        except Exception as e:
            logger.error("Live odds fetch failed: %s\n%s", e, traceback.format_exc())
            log_entry = db_fetch.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_entry:
                log_entry.status = "failed"
                log_entry.error_message = f"{type(e).__name__}: {e}"
                log_entry.completed_at = datetime.utcnow()
                db_fetch.commit()
        finally:
            db_fetch.close()

    Thread(target=_run_fetch, daemon=True).start()

    return BspImportResponse(
        message=f"Live odds fetch started (next {req.hours_ahead}h)",
        log_id=log_id,
    )


@router.get("/value-bets")
def find_value_bets(
    min_edge: float = Query(default=0.1, description="Minimum edge (predicted prob - implied prob)"),
    db: Session = Depends(get_db),
):
    """
    Find value bets by comparing prediction probabilities to market odds.

    A value bet is where the model's predicted win probability exceeds
    the market's implied probability by at least min_edge.
    """
    from app.models.prediction import Prediction

    # Get latest odds and predictions for scheduled races
    results = (
        db.query(
            Race, RaceEntry, Dog, OddsSnapshot, Prediction,
        )
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .join(Dog, Dog.id == RaceEntry.dog_id)
        .join(OddsSnapshot, (OddsSnapshot.race_id == Race.id) & (OddsSnapshot.dog_id == Dog.id))
        .join(Prediction, Prediction.race_entry_id == RaceEntry.id)
        .filter(Race.status == "scheduled")
        .all()
    )

    value_bets = []
    for race, entry, dog, odds, pred in results:
        if not pred.win_probability or not odds.implied_prob:
            continue

        edge = pred.win_probability - odds.implied_prob
        if edge >= min_edge:
            track = db.query(Track).filter(Track.id == race.track_id).first()
            value_bets.append({
                "race_id": race.id,
                "race_date": race.race_date.isoformat(),
                "track": track.name if track else None,
                "race_number": race.race_number,
                "dog_name": dog.name,
                "trap": entry.trap,
                "predicted_win_prob": round(pred.win_probability, 4),
                "market_odds": odds.odds_decimal,
                "market_implied_prob": round(odds.implied_prob, 4),
                "edge": round(edge, 4),
                "bookmaker": odds.bookmaker,
            })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return {"value_bets": value_bets, "count": len(value_bets), "min_edge": min_edge}

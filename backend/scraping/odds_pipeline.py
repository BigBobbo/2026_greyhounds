"""
Database pipeline for writing betting odds data into the OddsSnapshot table.

Handles:
- Betfair BSP historical data → OddsSnapshot records
- Betfair Exchange live odds → OddsSnapshot records
- Matching odds to existing Race + Dog records by name/track/date
"""

import logging
import re
from datetime import datetime, date
from typing import Any

from sqlalchemy.orm import Session

from app.models.dog import Dog
from app.models.odds import OddsSnapshot
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track
from scraping.db_pipeline import normalize_name

logger = logging.getLogger(__name__)


def _parse_event_date(date_str: str) -> date | None:
    """Parse Betfair EVENT_DT format (various formats)."""
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _find_matching_race(
    db: Session,
    track_code: str,
    race_date: date,
    distance_m: int | None,
    race_time: str | None,
) -> Race | None:
    """Find a race in the DB matching the BSP record's track/date/distance/time."""
    track = db.query(Track).filter(Track.code == track_code).first()
    if not track:
        return None

    query = db.query(Race).filter(
        Race.track_id == track.id,
        Race.race_date == race_date,
    )

    if distance_m:
        query = query.filter(Race.distance_m == distance_m)

    races = query.all()

    if len(races) == 1:
        return races[0]

    # If multiple races match, try narrowing by time
    if race_time and len(races) > 1:
        for race in races:
            if race.race_time and race.race_time.strftime("%H:%M") == race_time:
                return race

    # If still ambiguous, return None (can't match confidently)
    if len(races) > 1:
        logger.debug(
            "Ambiguous race match: %s %s %sm — %d candidates",
            track_code, race_date, distance_m, len(races),
        )
    return races[0] if len(races) == 1 else None


def _find_dog_by_name(db: Session, name: str) -> Dog | None:
    """Find a dog by normalized name."""
    norm = normalize_name(name)
    return db.query(Dog).filter(Dog.name == norm).first()


def _find_race_entry(db: Session, race_id: int, dog_id: int | None = None, trap: int | None = None) -> RaceEntry | None:
    """Find a race entry by race + dog or race + trap."""
    if dog_id:
        entry = db.query(RaceEntry).filter(
            RaceEntry.race_id == race_id,
            RaceEntry.dog_id == dog_id,
        ).first()
        if entry:
            return entry

    if trap:
        return db.query(RaceEntry).filter(
            RaceEntry.race_id == race_id,
            RaceEntry.trap == trap,
        ).first()

    return None


def upsert_bsp_odds(
    db: Session,
    bsp_records: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Write Betfair BSP records into the OddsSnapshot table.

    Attempts to match each BSP record to an existing Race + Dog in the DB.
    Records that can't be matched are skipped (logged at debug level).

    Returns stats: {"matched", "skipped_no_race", "skipped_no_dog", "inserted", "duplicate"}
    """
    stats = {
        "matched": 0,
        "skipped_no_race": 0,
        "skipped_no_dog": 0,
        "inserted": 0,
        "duplicate": 0,
    }

    for record in bsp_records:
        track_code = record.get("gri_track_code")
        if not track_code:
            stats["skipped_no_race"] += 1
            continue

        event_date = _parse_event_date(record.get("event_date", ""))
        if not event_date:
            stats["skipped_no_race"] += 1
            continue

        race = _find_matching_race(
            db, track_code, event_date,
            record.get("distance_m"),
            record.get("race_time"),
        )
        if not race:
            stats["skipped_no_race"] += 1
            continue

        dog_name = record.get("dog_name", "")
        dog = _find_dog_by_name(db, dog_name)

        # If no dog match by name, try matching by trap in the race entry
        dog_id = dog.id if dog else None
        trap = record.get("trap")

        if not dog_id and trap:
            entry = _find_race_entry(db, race.id, trap=trap)
            if entry:
                dog_id = entry.dog_id

        if not dog_id:
            stats["skipped_no_dog"] += 1
            continue

        stats["matched"] += 1

        # Check for duplicate
        existing = db.query(OddsSnapshot).filter(
            OddsSnapshot.race_id == race.id,
            OddsSnapshot.dog_id == dog_id,
            OddsSnapshot.bookmaker == "betfair_bsp",
            OddsSnapshot.is_sp.is_(True),
        ).first()

        if existing:
            stats["duplicate"] += 1
            continue

        bsp = record.get("bsp_decimal")
        if not bsp:
            continue

        snapshot = OddsSnapshot(
            race_id=race.id,
            dog_id=dog_id,
            bookmaker="betfair_bsp",
            odds_decimal=bsp,
            implied_prob=record.get("implied_prob"),
            scraped_at=datetime.utcnow(),
            is_sp=True,
        )
        db.add(snapshot)
        stats["inserted"] += 1

    db.commit()
    logger.info(
        "BSP upsert: %d matched, %d inserted, %d duplicates, %d no-race, %d no-dog",
        stats["matched"], stats["inserted"], stats["duplicate"],
        stats["skipped_no_race"], stats["skipped_no_dog"],
    )
    return stats


def upsert_live_odds(
    db: Session,
    live_records: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Write Betfair Exchange live odds into the OddsSnapshot table.

    These are pre-race or in-play odds, not starting prices.
    Multiple snapshots per dog/race are expected (time series of odds movement).

    Returns stats: {"matched", "skipped", "inserted"}
    """
    stats = {"matched": 0, "skipped": 0, "inserted": 0}
    now = datetime.utcnow()

    for record in live_records:
        dog_name = record.get("dog_name", "")
        dog = _find_dog_by_name(db, dog_name)

        if not dog:
            stats["skipped"] += 1
            continue

        # Try to find the race via dog's upcoming entries
        # For live odds, we match by dog + upcoming race date
        entries = (
            db.query(RaceEntry)
            .join(Race)
            .filter(
                RaceEntry.dog_id == dog.id,
                Race.status == "scheduled",
            )
            .all()
        )

        if not entries:
            stats["skipped"] += 1
            continue

        # Use the first scheduled entry (most likely the upcoming race)
        entry = entries[0]
        stats["matched"] += 1

        odds = record.get("odds_decimal")
        if not odds:
            continue

        snapshot = OddsSnapshot(
            race_id=entry.race_id,
            dog_id=dog.id,
            bookmaker=record.get("bookmaker", "betfair_exchange"),
            odds_decimal=odds,
            implied_prob=record.get("implied_prob"),
            scraped_at=now,
            is_sp=False,
        )
        db.add(snapshot)
        stats["inserted"] += 1

    db.commit()
    logger.info("Live odds upsert: %d matched, %d inserted, %d skipped", stats["matched"], stats["inserted"], stats["skipped"])
    return stats

"""
Database pipeline for writing scraped race data into the database.

Handles:
- Track lookup by code
- Dog find-or-create by normalized name + trainer
- Race upsert (idempotent on track + date + race_number)
- RaceEntry upsert (idempotent on race + trap)
"""

import logging
import re
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Normalize a dog/trainer name for matching."""
    name = name.strip().upper()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[''`]", "'", name)
    return name


def find_or_create_dog(
    db: Session, name: str, trainer_name: str | None = None,
    sire: str | None = None, dam: str | None = None,
) -> Dog:
    """Find existing dog by normalized name, or create new."""
    norm_name = normalize_name(name)

    dog = db.query(Dog).filter(Dog.name == norm_name).first()
    if dog:
        # Update missing fields
        if sire and not dog.sire:
            dog.sire = sire
        if dam and not dog.dam:
            dog.dam = dam
        if trainer_name and not dog.trainer_name:
            dog.trainer_name = normalize_name(trainer_name)
        return dog

    # Create new dog
    dog = Dog(
        name=norm_name,
        sire=sire,
        dam=dam,
        trainer_name=normalize_name(trainer_name) if trainer_name else None,
    )
    db.add(dog)
    db.flush()
    return dog


def upsert_race(
    db: Session,
    track: Track,
    race_data: dict[str, Any],
) -> Race:
    """Upsert a race record. Returns existing or new Race."""
    race_date_val = race_data["race_date"]
    race_number = race_data.get("race_number")

    existing = (
        db.query(Race)
        .filter(
            Race.track_id == track.id,
            Race.race_date == race_date_val,
            Race.race_number == race_number,
        )
        .first()
    )

    if existing:
        # Update fields if we have better data
        if race_data.get("distance_m") and not existing.distance_m:
            existing.distance_m = race_data["distance_m"]
        if race_data.get("grade") and not existing.grade:
            existing.grade = race_data["grade"]
        if race_data.get("going") and not existing.going:
            existing.going = race_data["going"]
        if race_data.get("prize_money") and not existing.prize_money:
            existing.prize_money = race_data["prize_money"]
        if race_data.get("race_time") and not existing.race_time:
            existing.race_time = race_data["race_time"]
        # Mark as resulted if we have finish data
        if any(e.get("finish_position") for e in race_data.get("entries", [])):
            existing.status = "resulted"
        return existing

    race = Race(
        track_id=track.id,
        race_date=race_date_val,
        race_number=race_number,
        race_time=race_data.get("race_time"),
        distance_m=race_data.get("distance_m") or 0,
        grade=race_data.get("grade"),
        race_type=race_data.get("race_type", "flat"),
        going=race_data.get("going"),
        prize_money=race_data.get("prize_money"),
        num_runners=len(race_data.get("entries", [])),
        source="gri",
        status="resulted" if any(e.get("finish_position") for e in race_data.get("entries", [])) else "scheduled",
    )
    db.add(race)
    db.flush()
    return race


def upsert_race_entry(
    db: Session,
    race: Race,
    entry_data: dict[str, Any],
) -> RaceEntry | None:
    """Upsert a race entry. Returns existing or new RaceEntry."""
    dog_name = entry_data.get("dog_name")
    trap = entry_data.get("trap")

    if not dog_name or not trap:
        logger.warning("Entry missing dog_name or trap: %s", entry_data)
        return None

    dog = find_or_create_dog(
        db, dog_name,
        trainer_name=entry_data.get("trainer_name"),
        sire=entry_data.get("sire_name"),
        dam=entry_data.get("dam_name"),
    )

    existing = (
        db.query(RaceEntry)
        .filter(RaceEntry.race_id == race.id, RaceEntry.trap == trap)
        .first()
    )

    if existing:
        # Update with new data if available
        if entry_data.get("finish_position") is not None:
            existing.finish_position = entry_data["finish_position"]
        if entry_data.get("finish_time") is not None:
            existing.finish_time = entry_data["finish_time"]
        if entry_data.get("starting_price") and not existing.starting_price:
            existing.starting_price = entry_data["starting_price"]
            existing.sp_decimal = entry_data.get("sp_decimal")
        if entry_data.get("weight_kg") and not existing.weight_kg:
            existing.weight_kg = entry_data["weight_kg"]
        if entry_data.get("comment") and not existing.comment:
            existing.comment = entry_data["comment"]
        return existing

    entry = RaceEntry(
        race_id=race.id,
        dog_id=dog.id,
        trap=trap,
        finish_position=entry_data.get("finish_position"),
        finish_time=entry_data.get("finish_time"),
        sectional_time=entry_data.get("sectional_time"),
        beaten_distance=entry_data.get("beaten_distance"),
        weight_kg=entry_data.get("weight_kg"),
        starting_price=entry_data.get("starting_price"),
        sp_decimal=entry_data.get("sp_decimal"),
        comment=entry_data.get("comment"),
        grade_at_entry=entry_data.get("grade_at_entry"),
    )
    db.add(entry)
    db.flush()
    return entry


def upsert_race_results(
    db: Session,
    scraped_races: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Upsert a batch of scraped race results into the database.

    Returns stats: {"races_new", "races_updated", "entries_new", "entries_updated", "dogs_new"}
    """
    stats = {
        "races_new": 0,
        "races_updated": 0,
        "entries_new": 0,
        "entries_updated": 0,
        "dogs_new": 0,
    }

    dogs_before = db.query(Dog).count()

    for race_data in scraped_races:
        track_code = race_data.get("track_code")
        if not track_code:
            continue

        track = db.query(Track).filter(Track.code == track_code).first()
        if not track:
            logger.warning("Unknown track code: %s", track_code)
            continue

        # Check if race exists
        existing_race = (
            db.query(Race)
            .filter(
                Race.track_id == track.id,
                Race.race_date == race_data["race_date"],
                Race.race_number == race_data.get("race_number"),
            )
            .first()
        )

        race = upsert_race(db, track, race_data)

        if existing_race:
            stats["races_updated"] += 1
        else:
            stats["races_new"] += 1

        for entry_data in race_data.get("entries", []):
            existing_entry = None
            if existing_race:
                existing_entry = (
                    db.query(RaceEntry)
                    .filter(
                        RaceEntry.race_id == race.id,
                        RaceEntry.trap == entry_data.get("trap"),
                    )
                    .first()
                )

            upsert_race_entry(db, race, entry_data)

            if existing_entry:
                stats["entries_updated"] += 1
            else:
                stats["entries_new"] += 1

    db.commit()

    dogs_after = db.query(Dog).count()
    stats["dogs_new"] = dogs_after - dogs_before

    logger.info(
        "Upsert complete: %d new races, %d updated, %d new entries, %d new dogs",
        stats["races_new"],
        stats["races_updated"],
        stats["entries_new"],
        stats["dogs_new"],
    )

    return stats

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
from datetime import date, datetime
from typing import Any

from sqlalchemy import func
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


def _last_resulted_race_date(
    db: Session, dog_id: int, before_date: date,
) -> date | None:
    """Most recent resulted-race date for a dog strictly before `before_date`.

    Used at scrape time to populate `RaceEntry.days_since_last` for upcoming
    cards: GRI doesn't print this on the card page but it's trivially
    derivable from the dog's prior resulted races. Pre-filling the column
    keeps the row dense for analytics/UI/feature code that reads the
    entry directly, instead of relying on every consumer to recompute.
    """
    return (
        db.query(func.max(Race.race_date))
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.dog_id == dog_id,
            Race.race_date < before_date,
            Race.status == "resulted",
        )
        .scalar()
    )


def find_or_create_dog(
    db: Session, name: str, trainer_name: str | None = None,
    sire: str | None = None, dam: str | None = None,
) -> Dog:
    """Find existing dog by normalized name, or create new.

    Greyhound names recur across generations. Matching by name alone merges
    two different dogs into one blended form history, so when BOTH the
    stored dog and the scraped entry carry a full pedigree (sire and dam)
    and BOTH differ, we treat it as a different animal with the same name
    and create a separate record. Partial or missing pedigree keeps the
    conservative name match — a spelling variant in one parent must not
    fork a dog's history in two.
    """
    norm_name = normalize_name(name)

    candidates = db.query(Dog).filter(Dog.name == norm_name).all()
    scraped_sire = normalize_name(sire) if sire else None
    scraped_dam = normalize_name(dam) if dam else None

    match: Dog | None = None
    for dog in candidates:
        stored_sire = normalize_name(dog.sire) if dog.sire else None
        stored_dam = normalize_name(dog.dam) if dog.dam else None
        if (
            scraped_sire and scraped_dam and stored_sire and stored_dam
            and scraped_sire != stored_sire and scraped_dam != stored_dam
        ):
            continue  # same name, different parents on both sides: not this dog
        match = dog
        break

    if match:
        if sire and not match.sire:
            match.sire = sire
        if dam and not match.dam:
            match.dam = dam
        if trainer_name and not match.trainer_name:
            match.trainer_name = normalize_name(trainer_name)
        return match

    if candidates:
        logger.info(
            "Namesake split: creating a second dog named %r (scraped sire=%r "
            "dam=%r differs from all %d stored namesakes)",
            norm_name, sire, dam, len(candidates),
        )

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
    scrape_log_id: int | None = None,
    scraped_at: datetime | None = None,
) -> Race:
    """Upsert a race record. Returns existing or new Race."""
    race_date_val = race_data["race_date"]
    race_number = race_data.get("race_number")
    stamp = scraped_at or datetime.utcnow()

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
        existing.last_scraped_at = stamp
        if scrape_log_id is not None:
            existing.last_scrape_log_id = scrape_log_id
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
        last_scraped_at=stamp,
        last_scrape_log_id=scrape_log_id,
    )
    db.add(race)
    db.flush()
    return race


def upsert_race_entry(
    db: Session,
    race: Race,
    entry_data: dict[str, Any],
    scrape_log_id: int | None = None,
    scraped_at: datetime | None = None,
) -> RaceEntry | None:
    """Upsert a race entry. Returns existing or new RaceEntry."""
    dog_name = entry_data.get("dog_name")
    trap = entry_data.get("trap")
    stamp = scraped_at or datetime.utcnow()

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

    # Backfill derivable fields the card page doesn't carry but we already
    # know from the race row / dog history. Both are computable pre-race
    # and have to be re-derived at predict time otherwise — populating
    # them at scrape time keeps the row dense and the predict-time strict
    # mode stops misclassifying these as missing.
    grade_at_entry = entry_data.get("grade_at_entry") or race.grade

    days_since_last = entry_data.get("days_since_last")
    if days_since_last is None and race.race_date is not None:
        last_date = _last_resulted_race_date(db, dog.id, race.race_date)
        if last_date is not None:
            days_since_last = (race.race_date - last_date).days

    if existing:
        # The (race, trap) row exists — but the dog in it may be wrong.
        # Cards are scraped days ahead; reserves are substituted on the
        # night, and GRI occasionally amends a runner's identity after
        # publication. The freshly scraped page is authoritative for who
        # actually ran in this trap: reassign dog_id when it differs, or
        # the result (position, time, SP, comment) is attached to a dog
        # that never ran — poisoning its form history and the training
        # labels built from it.
        if existing.dog_id != dog.id:
            logger.info(
                "Trap %s in race %s: dog changed %s -> %s (%r) on re-scrape "
                "(reserve substitution or GRI amendment)",
                trap, race.id, existing.dog_id, dog.id, dog.name,
            )
            existing.dog_id = dog.id

        # Result fields: the scraped page is the source of truth, so a
        # present value always wins — GRI corrections (amended SP, weight,
        # comment) must be able to overwrite the first-scraped value. Only
        # absent scraped values leave the stored one alone.
        if entry_data.get("finish_position") is not None:
            existing.finish_position = entry_data["finish_position"]
        if entry_data.get("finish_time") is not None:
            existing.finish_time = entry_data["finish_time"]
        if entry_data.get("starting_price"):
            existing.starting_price = entry_data["starting_price"]
            if entry_data.get("sp_decimal") is not None:
                existing.sp_decimal = entry_data["sp_decimal"]
        if entry_data.get("weight_kg"):
            existing.weight_kg = entry_data["weight_kg"]
        if entry_data.get("comment"):
            existing.comment = entry_data["comment"]
        if entry_data.get("beaten_distance") is not None:
            existing.beaten_distance = entry_data["beaten_distance"]
        if entry_data.get("sectional_time") is not None:
            existing.sectional_time = entry_data["sectional_time"]
        # Backfill derivable fields if they were left NULL at first scrape.
        if grade_at_entry and not existing.grade_at_entry:
            existing.grade_at_entry = grade_at_entry
        if days_since_last is not None and existing.days_since_last is None:
            existing.days_since_last = days_since_last
        existing.last_scraped_at = stamp
        if scrape_log_id is not None:
            existing.last_scrape_log_id = scrape_log_id
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
        grade_at_entry=grade_at_entry,
        days_since_last=days_since_last,
        last_scraped_at=stamp,
        last_scrape_log_id=scrape_log_id,
    )
    db.add(entry)
    db.flush()
    return entry


def upsert_race_results(
    db: Session,
    scraped_races: list[dict[str, Any]],
    scrape_log_id: int | None = None,
) -> dict[str, int]:
    """
    Upsert a batch of scraped race results into the database.

    `scrape_log_id` is stamped onto every upserted Race and RaceEntry along
    with a `last_scraped_at` timestamp, so each row knows which scrape job
    last touched it.

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
    stamp = datetime.utcnow()

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

        race = upsert_race(db, track, race_data, scrape_log_id=scrape_log_id, scraped_at=stamp)

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

            upsert_race_entry(db, race, entry_data, scrape_log_id=scrape_log_id, scraped_at=stamp)

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

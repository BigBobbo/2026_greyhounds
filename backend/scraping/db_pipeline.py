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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track

logger = logging.getLogger(__name__)

# Dogs whose history ordering was disturbed by an out-of-order insert (an
# entry dated EARLIER than the dog's latest already-stored resulted entry).
# Track-by-track backfills do this constantly: their later entries were given
# days_since_last values computed without knowledge of the newly inserted
# earlier race. Backfill runners drain this set via
# `pop_out_of_order_dogs()` and run `recompute_days_since_last()` when it is
# non-empty.
_out_of_order_dog_ids: set[int] = set()


def pop_out_of_order_dogs() -> set[int]:
    """Return and clear the set of dogs flagged by out-of-order inserts."""
    global _out_of_order_dog_ids
    flagged = _out_of_order_dog_ids
    _out_of_order_dog_ids = set()
    return flagged


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
            RaceEntry.scratched.isnot(True),
        )
        .scalar()
    )


def find_or_create_dog(
    db: Session, name: str, trainer_name: str | None = None,
    sire: str | None = None, dam: str | None = None,
    gri_id: str | None = None,
) -> Dog:
    """Resolve a dog's identity, creating a new row when needed.

    Resolution order:
      1. By `gri_id` (GRI's stable per-dog identifier) when provided.
      2. By normalized name. When the name-match has no gri_id yet and one
         was scraped, it is backfilled onto the legacy row. When the
         name-match carries a DIFFERENT non-null gri_id, it is a different
         dog that happens to share the name — a new row is created.
    """
    norm_name = normalize_name(name)

    def _fill_missing(d: Dog) -> Dog:
        if sire and not d.sire:
            d.sire = sire
        if dam and not d.dam:
            d.dam = dam
        if trainer_name and not d.trainer_name:
            d.trainer_name = normalize_name(trainer_name)
        return d

    # 1. GRI id is authoritative when present.
    if gri_id:
        dog = db.query(Dog).filter(Dog.gri_id == gri_id).first()
        if dog:
            return _fill_missing(dog)

    # 2. Fall back to normalized-name matching (legacy behaviour).
    dog = db.query(Dog).filter(Dog.name == norm_name).first()
    if dog:
        if gri_id and dog.gri_id is None:
            # Legacy name-only row: adopt the scraped GRI id.
            dog.gri_id = gri_id
        if not gri_id or dog.gri_id == gri_id:
            return _fill_missing(dog)
        logger.warning(
            "Dog name collision: %r already exists with gri_id=%s but the "
            "scraped entry has gri_id=%s — treating as a different dog and "
            "creating a new row",
            norm_name, dog.gri_id, gri_id,
        )

    # Create new dog. A concurrent writer may race us on the unique gri_id
    # index — use a savepoint so an IntegrityError only rolls back this
    # insert, then re-select the winning row.
    new_dog = Dog(
        name=norm_name,
        sire=sire,
        dam=dam,
        trainer_name=normalize_name(trainer_name) if trainer_name else None,
        gri_id=gri_id,
    )
    try:
        with db.begin_nested():
            db.add(new_dog)
            db.flush()
    except IntegrityError:
        logger.warning(
            "IntegrityError inserting dog %r (gri_id=%s) — re-selecting",
            norm_name, gri_id,
        )
        if gri_id:
            dog = db.query(Dog).filter(Dog.gri_id == gri_id).first()
        else:
            dog = db.query(Dog).filter(Dog.name == norm_name).first()
        if dog is None:
            raise
        return _fill_missing(dog)
    return new_dog


def upsert_race(
    db: Session,
    track: Track,
    race_data: dict[str, Any],
    scrape_log_id: int | None = None,
    scraped_at: datetime | None = None,
) -> Race | None:
    """Upsert a race record. Returns existing or new Race.

    Returns None (and does not create a race) when a NEW race has no parsed
    distance — storing distance_m=0 would poison distance-based features.
    """
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

    distance_val = race_data.get("distance_m")
    if not distance_val:
        logger.warning(
            "Skipping race with no parsed distance: track=%s date=%s race=%s",
            track.code, race_date_val, race_number,
        )
        return None

    race = Race(
        track_id=track.id,
        race_date=race_date_val,
        race_number=race_number,
        race_time=race_data.get("race_time"),
        distance_m=distance_val,
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
        gri_id=entry_data.get("gri_id"),
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
        # Reconcile identity: the card may have listed dog A but the results
        # show dog B in the same trap (reserve substitution). The freshly
        # resolved dog wins — the entry is reassigned, not duplicated.
        if existing.dog_id != dog.id:
            old_dog = db.get(Dog, existing.dog_id)
            logger.warning(
                "reserve substitution at race %s trap %s: %s -> %s",
                race.id, trap,
                old_dog.name if old_dog else f"dog_id={existing.dog_id}",
                dog.name,
            )
            existing.dog_id = dog.id
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
        # Backfill derivable fields if they were left NULL at first scrape.
        if grade_at_entry and not existing.grade_at_entry:
            existing.grade_at_entry = grade_at_entry
        if days_since_last is not None and existing.days_since_last is None:
            existing.days_since_last = days_since_last
        existing.last_scraped_at = stamp
        if scrape_log_id is not None:
            existing.last_scrape_log_id = scrape_log_id
        return existing

    # Out-of-order insert detection (track-by-track backfill corruption):
    # when this NEW entry is dated earlier than the dog's latest existing
    # resulted entry, every later entry's days_since_last may have been
    # computed without knowledge of this race. Flag the dog so the backfill
    # runner can heal with recompute_days_since_last() at the end.
    if race.race_date is not None:
        latest_resulted = (
            db.query(func.max(Race.race_date))
            .join(RaceEntry, RaceEntry.race_id == Race.id)
            .filter(
                RaceEntry.dog_id == dog.id,
                Race.status == "resulted",
                RaceEntry.scratched.isnot(True),
            )
            .scalar()
        )
        if latest_resulted is not None and race.race_date < latest_resulted:
            _out_of_order_dog_ids.add(dog.id)

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
        if race is None:
            # Race skipped (e.g. missing distance) — don't count it or
            # attempt to write its entries.
            continue

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

        # When this scrape carries RESULTS, any previously carded entry whose
        # trap is absent from the results did not run — mark it scratched and
        # recompute num_runners from the non-scratched entries.
        scraped_entries = race_data.get("entries", [])
        has_results = any(
            e.get("finish_position") is not None for e in scraped_entries
        )
        if has_results:
            result_traps = {e.get("trap") for e in scraped_entries if e.get("trap")}
            stored_entries = (
                db.query(RaceEntry).filter(RaceEntry.race_id == race.id).all()
            )
            for stored in stored_entries:
                if stored.trap not in result_traps:
                    if stored.scratched is not True:
                        stored.scratched = True
                        logger.warning(
                            "Marking race %s trap %s (dog_id=%s) as scratched: "
                            "trap absent from results",
                            race.id, stored.trap, stored.dog_id,
                        )
                elif stored.scratched:
                    # Trap reappeared in results — it ran after all.
                    stored.scratched = False
            race.num_runners = sum(
                1 for stored in stored_entries if stored.scratched is not True
            )

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


def recompute_days_since_last(db: Session) -> int:
    """Heal `RaceEntry.days_since_last` values corrupted by out-of-order
    inserts (e.g. track-by-track backfills).

    One pass over all resulted+scheduled entries ordered by (dog, date,
    entry id), recomputing each non-scratched entry's days_since_last as the
    gap to the dog's previous RESULTED race date (None for debutants) and
    OVERWRITING wrong values. Scratched entries never count as a previous
    race and are not themselves rewritten.

    Uses a single bulk query + Python loop and commits at the end. Returns
    the number of entries changed.
    """
    rows = (
        db.query(
            RaceEntry.id,
            RaceEntry.dog_id,
            Race.race_date,
            Race.status,
            RaceEntry.days_since_last,
            RaceEntry.scratched,
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(Race.status.in_(("resulted", "scheduled")))
        .order_by(RaceEntry.dog_id, Race.race_date, RaceEntry.id)
        .all()
    )

    updates: list[dict[str, Any]] = []
    prev_dog_id: int | None = None
    # Latest resulted date seen for this dog, and the latest one strictly
    # before it — needed so a same-day double doesn't count itself as the
    # "previous" race.
    max_date: date | None = None
    prev_max_date: date | None = None

    for entry_id, dog_id, race_date, status, current_val, scratched in rows:
        if dog_id != prev_dog_id:
            prev_dog_id = dog_id
            max_date = None
            prev_max_date = None

        if race_date is None:
            continue

        # Previous resulted date strictly before this entry's race date.
        if max_date is None:
            prior = None
        elif max_date < race_date:
            prior = max_date
        else:  # max_date == race_date (same-day double)
            prior = prev_max_date

        if scratched is not True:
            expected = (race_date - prior).days if prior is not None else None
            if current_val != expected:
                updates.append({"id": entry_id, "days_since_last": expected})

            if status == "resulted":
                if max_date is None or race_date > max_date:
                    prev_max_date = max_date
                    max_date = race_date

    if updates:
        db.bulk_update_mappings(RaceEntry, updates)
        db.commit()
        logger.info("recompute_days_since_last: corrected %d entries", len(updates))
    return len(updates)

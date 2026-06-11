"""DB pipeline identity / reconciliation / healing tests (audit E4/E5/E6).

Runs against the migrated temp test database (tests/conftest.py points
DATABASE_URL at it before app modules import). Every row created here is
deleted in the fixture teardown so other test modules are unaffected.
"""

from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal
from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track
from scraping.db_pipeline import (
    find_or_create_dog,
    pop_out_of_order_dogs,
    recompute_days_since_last,
    upsert_race_results,
)

TRACK_A = "ZZA"
TRACK_B = "ZZB"
NAME_PREFIX = "PIPETEST "


@pytest.fixture
def db():
    s = SessionLocal()
    # Make sure the two synthetic tracks exist.
    for code, name in ((TRACK_A, "Pipetest Park"), (TRACK_B, "Pipetest Downs")):
        if s.query(Track).filter(Track.code == code).first() is None:
            s.add(Track(name=name, code=code, active=True))
    s.commit()
    try:
        yield s
        s.rollback()
    finally:
        # Delete everything this module created: entries -> races -> dogs -> tracks.
        track_ids = [
            t.id for t in s.query(Track).filter(Track.code.in_([TRACK_A, TRACK_B]))
        ]
        race_ids = [
            r.id for r in s.query(Race).filter(Race.track_id.in_(track_ids))
        ]
        if race_ids:
            s.query(RaceEntry).filter(RaceEntry.race_id.in_(race_ids)).delete(
                synchronize_session=False
            )
            s.query(Race).filter(Race.id.in_(race_ids)).delete(
                synchronize_session=False
            )
        s.query(Dog).filter(Dog.name.like(f"{NAME_PREFIX}%")).delete(
            synchronize_session=False
        )
        s.query(Track).filter(Track.id.in_(track_ids)).delete(
            synchronize_session=False
        )
        s.commit()
        pop_out_of_order_dogs()  # don't leak module-level flags
        s.close()


def _race_data(track_code, race_date, race_number, entries):
    return {
        "track_code": track_code,
        "race_date": race_date,
        "race_number": race_number,
        "distance_m": 525,
        "grade": "A3",
        "race_type": "flat",
        "entries": entries,
    }


def _get_race(db, track_code, race_date, race_number) -> Race:
    track = db.query(Track).filter(Track.code == track_code).one()
    return (
        db.query(Race)
        .filter(
            Race.track_id == track.id,
            Race.race_date == race_date,
            Race.race_number == race_number,
        )
        .one()
    )


# ---------------------------------------------------------------------------
# E5 — dog identity by GRI id
# ---------------------------------------------------------------------------


def test_same_name_different_gri_ids_creates_two_dogs(db):
    d1 = find_or_create_dog(db, f"{NAME_PREFIX}SHARED NAME", gri_id="500001")
    db.commit()
    d2 = find_or_create_dog(db, f"{NAME_PREFIX}SHARED NAME", gri_id="500002")
    db.commit()

    assert d1.id != d2.id
    assert d1.name == d2.name == f"{NAME_PREFIX}SHARED NAME"
    assert {d1.gri_id, d2.gri_id} == {"500001", "500002"}
    # Resolving again by gri_id hits the right row each time
    assert find_or_create_dog(db, f"{NAME_PREFIX}SHARED NAME", gri_id="500001").id == d1.id
    assert find_or_create_dog(db, f"{NAME_PREFIX}SHARED NAME", gri_id="500002").id == d2.id


def test_rescrape_backfills_gri_id_onto_legacy_dog(db):
    legacy = find_or_create_dog(db, f"{NAME_PREFIX}LEGACY HOUND")
    db.commit()
    assert legacy.gri_id is None

    again = find_or_create_dog(db, f"{NAME_PREFIX}LEGACY HOUND", gri_id="500003")
    db.commit()

    assert again.id == legacy.id
    assert again.gri_id == "500003"
    # No duplicate row was created
    count = db.query(Dog).filter(Dog.name == f"{NAME_PREFIX}LEGACY HOUND").count()
    assert count == 1


def test_gri_id_resolution_wins_over_name(db):
    d = find_or_create_dog(db, f"{NAME_PREFIX}ORIGINAL NAME", gri_id="500004")
    db.commit()
    # Same GRI id under a different printed name resolves to the same row.
    same = find_or_create_dog(db, f"{NAME_PREFIX}RENAMED DOG", gri_id="500004")
    assert same.id == d.id


def test_unique_index_rejects_duplicate_gri_id(db):
    db.add(Dog(name=f"{NAME_PREFIX}UNIQUE ONE", gri_id="500005"))
    db.commit()
    db.add(Dog(name=f"{NAME_PREFIX}UNIQUE TWO", gri_id="500005"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_multiple_null_gri_ids_allowed(db):
    """SQLite unique indexes must permit many NULLs (legacy name-only dogs)."""
    db.add(Dog(name=f"{NAME_PREFIX}NULL ONE"))
    db.add(Dog(name=f"{NAME_PREFIX}NULL TWO"))
    db.commit()


# ---------------------------------------------------------------------------
# E6 — dog_id reconciliation + scratched flagging
# ---------------------------------------------------------------------------


def test_reserve_substitution_reassigns_entry(db, caplog):
    rd = date(2026, 6, 1)
    # Card scrape: dog A carded in trap 3.
    upsert_race_results(db, [_race_data(TRACK_A, rd, 1, [
        {"trap": 3, "dog_name": f"{NAME_PREFIX}DOG A", "gri_id": "600001"},
    ])])

    race = _get_race(db, TRACK_A, rd, 1)
    entry = db.query(RaceEntry).filter(RaceEntry.race_id == race.id).one()
    dog_a = db.query(Dog).filter(Dog.gri_id == "600001").one()
    assert entry.dog_id == dog_a.id

    # Results scrape: dog B actually ran in trap 3 (reserve substitution).
    with caplog.at_level("WARNING"):
        upsert_race_results(db, [_race_data(TRACK_A, rd, 1, [
            {"trap": 3, "dog_name": f"{NAME_PREFIX}DOG B", "gri_id": "600002",
             "finish_position": 1, "finish_time": 29.0},
        ])])

    db.expire_all()
    entry = db.query(RaceEntry).filter(RaceEntry.race_id == race.id).one()
    dog_b = db.query(Dog).filter(Dog.gri_id == "600002").one()
    assert entry.dog_id == dog_b.id
    assert entry.finish_position == 1
    assert any("reserve substitution" in r.message for r in caplog.records)


def test_missing_trap_marked_scratched_and_num_runners_recomputed(db):
    rd = date(2026, 6, 2)
    carded = [
        {"trap": t, "dog_name": f"{NAME_PREFIX}CARD {t}", "gri_id": f"61000{t}"}
        for t in range(1, 7)
    ]
    upsert_race_results(db, [_race_data(TRACK_A, rd, 1, carded)])

    race = _get_race(db, TRACK_A, rd, 1)
    assert race.num_runners == 6
    assert race.status == "scheduled"

    # Results arrive with only 5 traps — trap 4 was scratched.
    results = [
        {"trap": t, "dog_name": f"{NAME_PREFIX}CARD {t}", "gri_id": f"61000{t}",
         "finish_position": i + 1, "finish_time": 29.0 + i / 10}
        for i, t in enumerate([1, 2, 3, 5, 6])
    ]
    upsert_race_results(db, [_race_data(TRACK_A, rd, 1, results)])

    db.expire_all()
    race = _get_race(db, TRACK_A, rd, 1)
    entries = {
        e.trap: e
        for e in db.query(RaceEntry).filter(RaceEntry.race_id == race.id)
    }
    assert race.status == "resulted"
    assert entries[4].scratched is True
    assert entries[4].finish_position is None
    for t in (1, 2, 3, 5, 6):
        assert entries[t].scratched is not True
    assert race.num_runners == 5


# ---------------------------------------------------------------------------
# E4 — days_since_last healing after out-of-order (track-by-track) inserts
# ---------------------------------------------------------------------------


def test_recompute_days_since_last_heals_out_of_order_backfill(db):
    pop_out_of_order_dogs()  # start clean
    name = f"{NAME_PREFIX}E4 TRAVELLER"
    gid = "700001"

    def entry(fp=1):
        return {"trap": 1, "dog_name": name, "gri_id": gid,
                "finish_position": fp, "finish_time": 29.0}

    # Track A backfilled first: January then March races.
    upsert_race_results(db, [_race_data(TRACK_A, date(2026, 1, 1), 1, [entry()])])
    upsert_race_results(db, [_race_data(TRACK_A, date(2026, 3, 1), 1, [entry()])])
    # Track B backfilled afterwards: the February race lands out of order.
    upsert_race_results(db, [_race_data(TRACK_B, date(2026, 2, 1), 1, [entry()])])

    dog = db.query(Dog).filter(Dog.gri_id == gid).one()

    def days_by_date():
        rows = (
            db.query(Race.race_date, RaceEntry.days_since_last)
            .join(RaceEntry, RaceEntry.race_id == Race.id)
            .filter(RaceEntry.dog_id == dog.id)
            .all()
        )
        return {rd: d for rd, d in rows}

    # The March entry was computed when only January existed: 59 days (wrong).
    before = days_by_date()
    assert before[date(2026, 1, 1)] is None
    assert before[date(2026, 2, 1)] == 31
    assert before[date(2026, 3, 1)] == 59  # corrupted by out-of-order insert

    # The out-of-order insert was flagged for the backfill runner.
    flagged = pop_out_of_order_dogs()
    assert dog.id in flagged

    changed = recompute_days_since_last(db)
    assert changed == 1

    db.expire_all()
    after = days_by_date()
    assert after[date(2026, 1, 1)] is None  # debutant
    assert after[date(2026, 2, 1)] == 31
    assert after[date(2026, 3, 1)] == 28  # healed: gap to Feb, not Jan

    # Idempotent: a second pass changes nothing.
    assert recompute_days_since_last(db) == 0


def test_recompute_ignores_scratched_previous_races(db):
    name = f"{NAME_PREFIX}E4 SCRATCHER"
    gid = "700002"

    def entry(fp=1, **extra):
        return {"trap": 1, "dog_name": name, "gri_id": gid,
                "finish_position": fp, "finish_time": 29.0, **extra}

    upsert_race_results(db, [_race_data(TRACK_A, date(2026, 4, 1), 1, [entry()])])
    upsert_race_results(db, [_race_data(TRACK_A, date(2026, 4, 15), 1, [entry()])])
    upsert_race_results(db, [_race_data(TRACK_A, date(2026, 5, 1), 1, [entry()])])

    # Mark the mid-month run scratched: it must not count as a previous race.
    dog = db.query(Dog).filter(Dog.gri_id == gid).one()
    mid_entry = (
        db.query(RaceEntry)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.dog_id == dog.id, Race.race_date == date(2026, 4, 15))
        .one()
    )
    mid_entry.scratched = True
    db.commit()

    recompute_days_since_last(db)
    db.expire_all()

    may_entry = (
        db.query(RaceEntry)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.dog_id == dog.id, Race.race_date == date(2026, 5, 1))
        .one()
    )
    # Gap measured to Apr 1 (30 days), skipping the scratched Apr 15 run.
    assert may_entry.days_since_last == 30

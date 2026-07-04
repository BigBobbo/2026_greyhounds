"""Re-scrapes heal bad data + void sweep tests (audit task E10).

Runs against the migrated temp test database; every row created here is
deleted again in the fixture teardown.
"""

from datetime import date, timedelta

import pytest

from app.database import SessionLocal
from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track
from scraping.db_pipeline import (
    pop_out_of_order_dogs,
    upsert_race_results,
    void_stale_scheduled_races,
)

TRACK = "ZZH"
NAME_PREFIX = "HEALTEST "
RACE_DATE = date(2026, 6, 1)


@pytest.fixture
def db():
    s = SessionLocal()
    if s.query(Track).filter(Track.code == TRACK).first() is None:
        s.add(Track(name="Healtest Park", code=TRACK, active=True))
        s.commit()
    try:
        yield s
        s.rollback()
    finally:
        track_ids = [t.id for t in s.query(Track).filter(Track.code == TRACK)]
        race_ids = [r.id for r in s.query(Race).filter(Race.track_id.in_(track_ids))]
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


def _results_race(entries, race_number=1, **overrides):
    race = {
        "track_code": TRACK,
        "race_date": RACE_DATE,
        "race_number": race_number,
        "distance_m": 525,
        "grade": "A3",
        "race_type": "flat",
        "going": None,
        "prize_money": None,
        "entries": entries,
    }
    race.update(overrides)
    return race


def _entry(trap, name, position, **overrides):
    e = {
        "trap": trap,
        "dog_name": f"{NAME_PREFIX}{name}",
        "gri_id": f"9919{trap}",
        "finish_position": position,
        "finish_time": 29.50,
        "weight_kg": 30.0,
        "starting_price": "5/2",
        "sp_decimal": 3.5,
        "comment": "Crd1",
    }
    e.update(overrides)
    return e


def _stored(s, race_number=1):
    track = s.query(Track).filter(Track.code == TRACK).one()
    race = (
        s.query(Race)
        .filter(Race.track_id == track.id, Race.race_number == race_number)
        .one()
    )
    entries = {
        e.trap: e
        for e in s.query(RaceEntry).filter(RaceEntry.race_id == race.id).all()
    }
    return race, entries


def test_results_rescrape_overwrites_corrected_fields(db):
    first = _results_race(
        [
            _entry(1, "ALPHA", 1, weight_kg=29.0, starting_price="5/2", sp_decimal=3.5),
            _entry(2, "BETA", 2),
        ],
        going="-10", prize_money=300.0,
    )
    upsert_race_results(db, [first])

    # GRI corrected the result page: weight, SP, time, going, grade changed.
    corrected = _results_race(
        [
            _entry(
                1, "ALPHA", 1,
                weight_kg=31.2, starting_price="3/1", sp_decimal=4.0,
                finish_time=29.41, comment="EP,Led",
            ),
            _entry(2, "BETA", 2),
        ],
        going="-20", grade="A2",
    )
    upsert_race_results(db, [corrected])

    db.expire_all()
    race, entries = _stored(db)
    e1 = entries[1]
    assert e1.weight_kg == 31.2
    assert e1.starting_price == "3/1"
    assert e1.sp_decimal == 4.0
    assert e1.finish_time == 29.41
    assert e1.comment == "EP,Led"
    assert race.going == "-20"
    assert race.grade == "A2"
    # Absent (None) values never clobber stored data.
    assert race.prize_money == 300.0


def test_card_rescrape_keeps_fill_only_semantics(db):
    upsert_race_results(db, [_results_race([_entry(1, "GAMMA", 1, weight_kg=29.0)])])

    # A later CARD scrape (no finish positions) with conflicting values must
    # not clobber result-derived data — results are stronger provenance.
    card = _results_race(
        [
            {
                "trap": 1,
                "dog_name": f"{NAME_PREFIX}GAMMA",
                "gri_id": "99191",
                "weight_kg": 35.0,
                "comment": "card noise",
            }
        ],
        grade="A9",
    )
    upsert_race_results(db, [card])

    db.expire_all()
    race, entries = _stored(db)
    assert entries[1].weight_kg == 29.0
    assert entries[1].comment == "Crd1"
    assert race.grade == "A3"
    assert race.status == "resulted"


def test_void_stale_scheduled_races_flags_old_and_keeps_recent(db):
    track = db.query(Track).filter(Track.code == TRACK).one()
    today = date.today()
    old_sched = Race(
        track_id=track.id, race_date=today - timedelta(days=10),
        race_number=51, distance_m=525, status="scheduled",
    )
    recent_sched = Race(
        track_id=track.id, race_date=today - timedelta(days=1),
        race_number=52, distance_m=525, status="scheduled",
    )
    old_resulted = Race(
        track_id=track.id, race_date=today - timedelta(days=10),
        race_number=53, distance_m=525, status="resulted",
    )
    db.add_all([old_sched, recent_sched, old_resulted])
    db.commit()

    voided = void_stale_scheduled_races(db, older_than_days=3)
    assert voided >= 1  # >= : other modules' leftover stale rows may sweep too

    db.expire_all()
    assert old_sched.status == "void"
    assert recent_sched.status == "scheduled"
    assert old_resulted.status == "resulted"

    # Idempotent for already-voided rows.
    void_stale_scheduled_races(db, older_than_days=3)
    db.expire_all()
    assert old_sched.status == "void"
    assert recent_sched.status == "scheduled"

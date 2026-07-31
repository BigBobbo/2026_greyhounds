"""Regression tests: population aggregates must be point-in-time.

The trap/trainer/sire/track aggregate features used to be computed over ALL
resulted races — including races run after the entry being featurised — which
leaked future outcomes into training rows. These tests build a database whose
aggregate rates flip sharply at a known date and assert that each entry only
ever sees data from strictly before its own race date.

Scenario: six dogs (one per trap, each with its own trainer) race daily.
Phase 1 (days 1-40): trap 1 always wins. A probe race runs on day 50 with the
same pattern. Phase 2 (days 51-90): trap 1 always finishes last. A final probe
runs on day 100.

  * As of the day-50 probe, trap 1's win rate is 40/40 = 1.0.
  * As of the day-100 probe, it is 41/81.
  * An all-time (leaky) computation would give both probes the same value.
"""

from datetime import date, time, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track
from ml.race_features import compute_builtin_features_batch


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()


START = date(2025, 1, 1)


def _add_race(db, track, day_offset, race_number):
    race = Race(
        track_id=track.id,
        race_date=START + timedelta(days=day_offset),
        race_time=time(19, 30),
        race_number=race_number,
        distance_m=525,
        grade="A3",
        race_type="flat",
        going="standard",
        num_runners=6,
        status="resulted",
    )
    db.add(race)
    return race


def _add_result(db, race, dogs, winner_trap):
    """Trap `winner_trap` wins; remaining traps fill positions in trap order."""
    positions = {}
    pos = 2
    for trap in range(1, 7):
        if trap == winner_trap:
            positions[trap] = 1
        else:
            positions[trap] = pos
            pos += 1
    for trap in range(1, 7):
        p = positions[trap]
        db.add(RaceEntry(
            race_id=race.id,
            dog_id=dogs[trap - 1].id,
            trap=trap,
            finish_position=p,
            finish_time=29.0 + 0.1 * (p - 1),
            weight_kg=32.0,
        ))


def _seed(db):
    track = Track(name="Testville", code="TST", active=True)
    db.add(track)
    db.commit()

    dogs = []
    for trap in range(1, 7):
        d = Dog(
            name=f"Dog{trap}",
            trainer_name=f"Trainer{trap}",
            sire=f"Sire{trap}",
            dam=f"Dam{trap}",
            birth_date=date(2023, 1, 1),
        )
        db.add(d)
        dogs.append(d)
    db.commit()

    race_no = 0

    # Phase 1: days 1-40, trap 1 always wins
    for day in range(1, 41):
        race_no += 1
        race = _add_race(db, track, day, race_no)
        db.commit()
        _add_result(db, race, dogs, winner_trap=1)
    db.commit()

    # Probe A on day 50 (also resulted, trap 1 wins)
    race_no += 1
    probe_a = _add_race(db, track, 50, race_no)
    db.commit()
    _add_result(db, probe_a, dogs, winner_trap=1)
    db.commit()

    # Phase 2: days 51-90, trap 1 always loses (trap 2 wins)
    for day in range(51, 91):
        race_no += 1
        race = _add_race(db, track, day, race_no)
        db.commit()
        _add_result(db, race, dogs, winner_trap=2)
    db.commit()

    # Probe B on day 100
    race_no += 1
    probe_b = _add_race(db, track, 100, race_no)
    db.commit()
    _add_result(db, probe_b, dogs, winner_trap=2)
    db.commit()

    return track, dogs, probe_a, probe_b


def _trap1_entry(db, race):
    return (
        db.query(RaceEntry)
        .filter(RaceEntry.race_id == race.id, RaceEntry.trap == 1)
        .one()
    )


def test_trap_rate_is_as_of_race_date(db):
    _track, _dogs, probe_a, probe_b = _seed(db)
    ea = _trap1_entry(db, probe_a)
    eb = _trap1_entry(db, probe_b)

    df = compute_builtin_features_batch(db, [ea.id, eb.id])

    rate_a = df.loc[ea.id, "trap_win_rate_at_track"]
    rate_b = df.loc[eb.id, "trap_win_rate_at_track"]

    # Probe A must see only phase 1: 40 wins from 40 runs.
    assert rate_a == pytest.approx(1.0)
    # Probe B sees phase 1 + probe A + phase 2: 41 wins from 81 runs.
    assert rate_b == pytest.approx(41.0 / 81.0)
    # The leaky all-time computation gave both probes the same value.
    assert rate_a != pytest.approx(rate_b)


def test_trainer_rate_is_as_of_race_date(db):
    _track, _dogs, probe_a, probe_b = _seed(db)
    ea = _trap1_entry(db, probe_a)
    eb = _trap1_entry(db, probe_b)

    df = compute_builtin_features_batch(db, [ea.id, eb.id])

    # Trainer1 trains the trap-1 dog exclusively, so the numbers mirror
    # the trap pattern exactly.
    assert df.loc[ea.id, "trainer_win_rate"] == pytest.approx(1.0)
    assert df.loc[eb.id, "trainer_win_rate"] == pytest.approx(41.0 / 81.0)


def test_no_prior_data_yields_none(db):
    """The very first race has no history at all — every as-of aggregate
    must be missing rather than borrowing from the future."""
    _track, _dogs, _probe_a, _probe_b = _seed(db)
    first_race = (
        db.query(Race).order_by(Race.race_date).first()
    )
    e = _trap1_entry(db, first_race)

    df = compute_builtin_features_batch(db, [e.id])

    import math
    for col in ("trap_win_rate_at_track", "trainer_win_rate",
                "sire_progeny_win_rate", "trainer_win_rate_at_track"):
        v = df.loc[e.id, col]
        assert v is None or (isinstance(v, float) and math.isnan(v)), (
            f"{col} should be missing for the first-ever race, got {v!r}"
        )


def test_dam_and_trainer_window_features_are_as_of(db):
    """Tier 12: dam progeny rate mirrors the trap flip (Dam1 = trap-1 dog's
    dam); the 90-day trainer window sees ONLY phase 2 by probe B."""
    _track, dogs, probe_a, probe_b = _seed(db)
    # Give each dog a distinct dam (the seed leaves sire/dam per trap)
    ea = _trap1_entry(db, probe_a)
    eb = _trap1_entry(db, probe_b)

    df = compute_builtin_features_batch(db, [ea.id, eb.id])

    # Dam1's progeny = the trap-1 dog only: 40/40 by probe A, 41/81 by B.
    assert df.loc[ea.id, "dam_progeny_win_rate"] == pytest.approx(1.0)
    assert df.loc[eb.id, "dam_progeny_win_rate"] == pytest.approx(41.0 / 81.0)

    # Probe B is day 100; the trailing 90 days cover day 10 onward: runs on
    # days 10..40 (31 wins), day 50 (win), days 51..90 (40 losses) = 32/72.
    assert df.loc[eb.id, "trainer_win_rate_90d"] == pytest.approx(32.0 / 72.0)
    # And the career rate at the same moment is 41/81 — the delta is the
    # cold-streak signal.
    assert df.loc[eb.id, "trainer_form_delta_90d"] == pytest.approx(
        32.0 / 72.0 - 41.0 / 81.0
    )

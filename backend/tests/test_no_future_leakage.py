"""Leakage regression tests for batch built-in features (audit tasks C1/C15).

The contract under test: every feature computed for an entry may depend only
on data strictly before that entry's race date. Two probes:

  1. Adding races AFTER the target date (same trainer, sire, traps, track,
     distance — feeding every aggregate) must not change the target's
     features.
  2. Changing the target entry's OWN result must not change its features
     (the old all-time aggregates baked each row's own outcome into its
     trap/trainer/sire win rates — direct target leakage).

The seed sizes are chosen so the aggregate features clear their minimum
sample thresholds (trap 30, trainer 20, sire 50, track baseline 50, speed
figure bucket 30) and are therefore non-None — otherwise these tests would
pass vacuously.
"""

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
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
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()


def _add_race(db, track_id, race_date, dogs, rng, race_number=1):
    race = Race(
        track_id=track_id,
        race_date=race_date,
        race_time=time(19, 30),
        race_number=race_number,
        distance_m=525,
        grade="A3",
        race_type="flat",
        going="standard",
        num_runners=len(dogs),
        status="resulted",
    )
    db.add(race)
    db.flush()
    order = list(dogs)
    rng.shuffle(order)
    entries = []
    for pos, d in enumerate(order, start=1):
        ft = 29.0 + (pos - 1) * 0.12 + rng.normal(0, 0.03)
        e = RaceEntry(
            race_id=race.id,
            dog_id=d.id,
            trap=pos,
            finish_position=pos,
            finish_time=ft,
            adjusted_time=ft,
            sectional_time=ft * 0.18,
            weight_kg=32.0,
            sp_decimal=2.0 + pos,
            comment="led" if pos == 1 else "crd 1",
        )
        db.add(e)
        entries.append(e)
    db.flush()
    return race, entries


def _seed(db):
    track = Track(name="Testville", code="TST", active=True)
    db.add(track)
    db.commit()

    dogs = []
    for name in ("Alpha", "Bravo", "Charlie", "Delta"):
        d = Dog(
            name=name,
            trainer_name="T. Trainer",
            sire="Top Sire",
            birth_date=date(2023, 1, 1),
        )
        db.add(d)
        dogs.append(d)
    db.commit()

    rng = np.random.default_rng(7)
    start = date(2025, 1, 1)
    target_entry_id = None
    for i in range(45):
        _, entries = _add_race(db, track.id, start + timedelta(days=i), dogs, rng, i + 1)
        if i == 40:
            target_entry_id = entries[0].id
    db.commit()
    return track, dogs, target_entry_id, start, rng


def _features_for(db, entry_id) -> pd.Series:
    df = compute_builtin_features_batch(db, [entry_id])
    assert len(df) == 1
    return df.iloc[0]


def _assert_identical(before: pd.Series, after: pd.Series):
    diffs = []
    for col in before.index:
        b, a = before[col], after[col]
        b_nan = pd.isna(b)
        a_nan = pd.isna(a)
        if b_nan and a_nan:
            continue
        if b_nan != a_nan or (
            isinstance(b, float) and isinstance(a, float) and abs(b - a) > 1e-12
        ) or (not isinstance(b, float) and b != a):
            diffs.append(f"{col}: {b!r} -> {a!r}")
    assert not diffs, "features changed:\n" + "\n".join(diffs)


SHARP_FEATURES = [
    # one representative per aggregate family — must be non-None or the
    # invariance assertion is vacuous
    "trap_win_rate_at_track",
    "trainer_win_rate",
    "trainer_win_rate_at_track",
    "sire_progeny_win_rate",
    "sire_progeny_mean_time_at_dist",
    "track_speed_rating",
    "speed_figure_mean_last5",
    "trap_bias_deviation_going",
]


def test_features_have_teeth(db):
    _, _, target_entry_id, _, _ = _seed(db)
    feats = _features_for(db, target_entry_id)
    missing = [f for f in SHARP_FEATURES if pd.isna(feats.get(f))]
    assert not missing, f"expected non-None (raise seed sizes?): {missing}"


def test_invariant_to_future_races(db):
    track, dogs, target_entry_id, start, rng = _seed(db)
    before = _features_for(db, target_entry_id)

    # Five new resulted races AFTER the target date, same trainer/sire/track/
    # distance/traps — under the old all-time aggregates these shifted every
    # rate the target entry sees.
    for j in range(5):
        _add_race(db, track.id, start + timedelta(days=50 + j), dogs, rng, 100 + j)
    db.commit()

    after = _features_for(db, target_entry_id)
    _assert_identical(before, after)


def test_invariant_to_own_race_result(db):
    _, _, target_entry_id, _, _ = _seed(db)
    before = _features_for(db, target_entry_id)

    target = db.query(RaceEntry).get(target_entry_id)
    race_entries = db.query(RaceEntry).filter(RaceEntry.race_id == target.race_id).all()
    # Invert the finish order of the target's own race
    n = len(race_entries)
    for e in race_entries:
        e.finish_position = n + 1 - e.finish_position
    db.commit()

    after = _features_for(db, target_entry_id)
    _assert_identical(before, after)


def test_builtin_features_computable_pre_race(db):
    """Audit C4: every builtin feature NOT classified post-race-only must be
    computable on a scheduled race (current-entry result fields all NULL),
    given the dog has full history. A feature that is non-null for a resulted
    entry but null for an identical scheduled entry depends on current-race
    result data and must be added to POST_RACE_FEATURE_NAMES.
    """
    from app.models.race import Race
    from ml.feature_availability import POST_RACE_FEATURE_NAMES

    track, dogs, target_entry_id, start, _ = _seed(db)
    resulted_feats = _features_for(db, target_entry_id)
    target = db.query(RaceEntry).get(target_entry_id)
    target_race = db.query(Race).get(target.race_id)

    # Scheduled race the day after the target's race: same card data (grade,
    # distance, traps, dogs) but no results, no weigh-in, no going.
    sched = Race(
        track_id=track.id,
        race_date=target_race.race_date + timedelta(days=1),
        race_time=time(20, 0),
        race_number=99,
        distance_m=525,
        grade="A3",
        race_type="flat",
        going=None,
        num_runners=len(dogs),
        status="scheduled",
    )
    db.add(sched)
    db.flush()
    # Give the target dog the same trap it had in the resulted race so the
    # trap-keyed features are directly comparable; distribute the rest.
    other_traps = [t for t in range(1, len(dogs) + 1) if t != target.trap]
    sched_entry = None
    for d in dogs:
        if d.id == target.dog_id:
            trap = target.trap
        else:
            trap = other_traps.pop(0)
        e = RaceEntry(race_id=sched.id, dog_id=d.id, trap=trap)
        db.add(e)
        if d.id == target.dog_id:
            sched_entry = e
    db.commit()
    assert sched_entry is not None

    sched_feats = _features_for(db, sched_entry.id)

    offenders = []
    for col in resulted_feats.index:
        if col in POST_RACE_FEATURE_NAMES:
            continue
        if not pd.isna(resulted_feats[col]) and pd.isna(sched_feats.get(col)):
            offenders.append(col)
    assert not offenders, (
        "features computable on resulted but not scheduled races — add to "
        f"POST_RACE_FEATURE_NAMES or fix: {offenders}"
    )

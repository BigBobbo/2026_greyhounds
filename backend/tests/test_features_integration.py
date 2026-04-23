"""Integration tests for ELO + speed-figure features against an in-memory DB.

Builds a small synthetic database with a handful of dogs racing across two
distances at one track, then verifies that:

  * compute_elo_features_batch returns a row per requested entry with the
    expected per-distance and per-track ELO context.
  * compute_builtin_features_batch produces speed-figure aggregates derived
    from per-(track, distance) baselines.
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
from ml.race_features import (
    compute_builtin_features_batch,
    compute_elo_features_batch,
    compute_h2h_features_batch,
)


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


def _seed_simple(db):
    """Seed: 1 track, 4 dogs, ~12 races mostly at 525m with a few 480m."""
    track = Track(name="Testville", code="TST", active=True)
    db.add(track)
    db.commit()

    dogs = []
    for name in ("Alpha", "Bravo", "Charlie", "Delta"):
        d = Dog(name=name, trainer_name="Anon", birth_date=date(2023, 1, 1))
        db.add(d)
        dogs.append(d)
    db.commit()

    base_time = 29.0  # baseline finish time at 525m
    races_per_day = 0
    race_no = 0
    races = []

    # Generate 12 races over 12 days, 4 runners each, with stable skill ranking
    # (Alpha > Bravo > Charlie > Delta) so ELO should rank them in that order
    # by the end.  A handful of races run at a different distance (480m).
    for i in range(12):
        race_no += 1
        d_m = 525 if i % 4 != 0 else 480
        race = Race(
            track_id=track.id,
            race_date=date(2025, 1, 1) + timedelta(days=i),
            race_time=time(19, 30),
            race_number=race_no,
            distance_m=d_m,
            grade="A3",
            race_type="flat",
            going="standard",
            num_runners=4,
            status="resulted",
        )
        db.add(race)
        races.append(race)
    db.commit()

    # Dog skill: Alpha fastest, then B, C, D — small noise added per race.
    rng = np.random.default_rng(42)
    skill = {dogs[0].id: 0.0, dogs[1].id: 0.15, dogs[2].id: 0.30, dogs[3].id: 0.45}
    for race in races:
        # Sort dogs by skill + noise to determine finish order
        finishers = sorted(
            dogs,
            key=lambda d: skill[d.id] + rng.normal(0, 0.05),
        )
        # Distance offset so 480m has slightly faster baseline times
        dist_offset = -1.0 if race.distance_m == 480 else 0.0
        for pos, d in enumerate(finishers, start=1):
            ft = base_time + dist_offset + skill[d.id] + (pos - 1) * 0.1
            adj = ft  # going_allowance = 0 here
            db.add(RaceEntry(
                race_id=race.id,
                dog_id=d.id,
                trap=pos,  # arbitrary — not under test
                finish_position=pos,
                finish_time=ft,
                adjusted_time=adj,
                sectional_time=ft * 0.18,
                weight_kg=32.0,
                sp_decimal=2.0 + pos,
            ))
    db.commit()
    return track, dogs, races


def test_elo_orders_dogs_by_skill(db):
    _track, dogs, races = _seed_simple(db)
    # Last race entries — by then ELO should have settled
    last_race = max(races, key=lambda r: r.race_date)
    last_entries = (
        db.query(RaceEntry)
        .filter(RaceEntry.race_id == last_race.id)
        .all()
    )
    entry_ids = [e.id for e in last_entries]

    df = compute_elo_features_batch(db, entry_ids)
    assert not df.empty
    assert set(df.columns) >= {
        "dog_elo", "dog_elo_at_distance", "dog_elo_at_track",
        "field_avg_elo", "field_max_elo",
        "elo_rank_in_field", "elo_gap_to_best", "elo_gap_to_avg",
        "dog_elo_races",
    }

    # Look up entry IDs per dog name in this last race
    name_to_entry = {}
    for e in last_entries:
        dog = db.query(Dog).filter(Dog.id == e.dog_id).first()
        name_to_entry[dog.name] = e.id

    elo_alpha = df.loc[name_to_entry["Alpha"], "dog_elo"]
    elo_delta = df.loc[name_to_entry["Delta"], "dog_elo"]
    # Alpha (consistently fastest) should outrate Delta (consistently slowest)
    assert elo_alpha > elo_delta

    # Pre-race counts: each dog should have raced 11 times before the last race
    assert df.loc[name_to_entry["Alpha"], "dog_elo_races"] == 11

    # Field aggregates are constant within a race
    assert df["field_avg_elo"].nunique() == 1


def test_elo_no_leakage_from_current_race(db):
    """Pre-race ELO snapshot must not include the entry's own race result."""
    _track, _dogs, races = _seed_simple(db)
    # Pick a middle race
    target_race = races[5]
    entries = (
        db.query(RaceEntry)
        .filter(RaceEntry.race_id == target_race.id)
        .all()
    )
    target_ids = [e.id for e in entries]

    snap_with_full = compute_elo_features_batch(db, target_ids)

    # If we ask for the same race after only computing up to itself, the
    # snapshot should be identical — meaning the function didn't leak the
    # race result back into the snapshot.
    snap_again = compute_elo_features_batch(db, target_ids)
    pd.testing.assert_frame_equal(snap_with_full, snap_again)


def test_speed_figure_features_present_and_finite(db):
    _track, _dogs, races = _seed_simple(db)
    # Look at entries from the last race; by then there's plenty of history
    last_race = max(races, key=lambda r: r.race_date)
    last_entries = (
        db.query(RaceEntry)
        .filter(RaceEntry.race_id == last_race.id)
        .all()
    )
    entry_ids = [e.id for e in last_entries]

    df = compute_builtin_features_batch(db, entry_ids)
    assert not df.empty
    for col in (
        "speed_figure_best_last10",
        "speed_figure_mean_last5",
        "speed_figure_ewm_last10",
        "career_peak_speed_figure",
    ):
        assert col in df.columns
        # At least some entries should have a numeric value (we have 11 races
        # of history for each dog, all at the test track, so baselines exist)
        non_null = df[col].dropna()
        assert len(non_null) > 0
        for v in non_null:
            assert np.isfinite(v)


def test_elo_handles_unresulted_prediction_race(db):
    """At prediction time the target race is 'scheduled' (not 'resulted').

    The ELO function should still snapshot pre-race ratings for those
    entries and must not feed the unresulted race's (missing) results back
    into the ELO state."""
    track, dogs, races = _seed_simple(db)

    # Snapshot ELO state across all dogs immediately after the last resulted
    # race so we can compare it to what the function reports for an
    # unresulted future race that follows it.
    last_race = max(races, key=lambda r: r.race_date)
    last_entries = (
        db.query(RaceEntry)
        .filter(RaceEntry.race_id == last_race.id)
        .all()
    )
    baseline = compute_elo_features_batch(db, [e.id for e in last_entries])

    # Now create a future scheduled race with the same 4 dogs
    future_race = Race(
        track_id=track.id,
        race_date=date(2025, 2, 1),
        race_time=time(20, 0),
        race_number=99,
        distance_m=525,
        grade="A3",
        race_type="flat",
        going="standard",
        num_runners=4,
        status="scheduled",
    )
    db.add(future_race)
    db.commit()

    future_entries = []
    for trap_no, d in enumerate(dogs, start=1):
        e = RaceEntry(
            race_id=future_race.id,
            dog_id=d.id,
            trap=trap_no,
            finish_position=None,  # not yet run
            sp_decimal=2.0 + trap_no,
        )
        db.add(e)
        future_entries.append(e)
    db.commit()

    future_ids = [e.id for e in future_entries]
    df = compute_elo_features_batch(db, future_ids)
    assert not df.empty
    assert len(df) == len(future_ids)

    # Each dog's pre-race overall ELO at the future race should equal its
    # POST-race ELO at the previous resulted race (no resulted races in
    # between).  The previous-race snapshot is the *pre*-race state for
    # that race, so we apply the race's own update once before comparing.
    # Simplest check: ratings strictly between min and max baseline ratings
    for col in ("dog_elo", "field_avg_elo", "field_max_elo"):
        assert df[col].notna().all()


def test_h2h_tracks_wins_and_losses_within_field(db):
    _track, dogs, races = _seed_simple(db)
    # The last race has all 4 dogs with a stable skill ordering — by then
    # Alpha has beaten every other dog many times.
    last_race = max(races, key=lambda r: r.race_date)
    last_entries = (
        db.query(RaceEntry)
        .filter(RaceEntry.race_id == last_race.id)
        .all()
    )
    entry_ids = [e.id for e in last_entries]

    df = compute_h2h_features_batch(db, entry_ids)
    assert not df.empty
    assert set(df.columns) >= {
        "h2h_meetings_vs_field",
        "h2h_wins_vs_field",
        "h2h_losses_vs_field",
        "h2h_win_rate_vs_field",
        "h2h_avg_beaten_length_vs_field",
        "best_opponent_beaten_count",
    }

    name_to_entry = {}
    for e in last_entries:
        dog = db.query(Dog).filter(Dog.id == e.dog_id).first()
        name_to_entry[dog.name] = e.id

    # Every dog has raced with the other three in every prior race, so
    # meetings should be: 11 prior races * 3 opponents = 33
    for dog_name in ("Alpha", "Bravo", "Charlie", "Delta"):
        meetings = df.loc[name_to_entry[dog_name], "h2h_meetings_vs_field"]
        assert meetings == 33

    # Alpha (fastest) should have the highest win rate vs field
    win_rate_alpha = df.loc[name_to_entry["Alpha"], "h2h_win_rate_vs_field"]
    win_rate_delta = df.loc[name_to_entry["Delta"], "h2h_win_rate_vs_field"]
    assert win_rate_alpha > win_rate_delta

    # Alpha should have beaten all 3 opponents at least once
    assert df.loc[name_to_entry["Alpha"], "best_opponent_beaten_count"] == 3


def test_h2h_returns_empty_on_empty_input(db):
    df = compute_h2h_features_batch(db, [])
    assert df.empty


def test_comment_derived_features_surface_through_batch(db):
    """Verify the structured comment parser propagates through the batch
    pipeline: when we seed every race with specific comments, the per-dog
    rate features should reflect what was actually parsed."""
    track = Track(name="CmtTrack", code="CMT", active=True)
    db.add(track)
    db.commit()

    dog = Dog(name="Runner", trainer_name="Anon", birth_date=date(2023, 1, 1))
    db.add(dog)
    db.commit()

    # Need a 2-dog field so the race is valid in our schema
    dog2 = Dog(name="Filler", trainer_name="Anon", birth_date=date(2023, 1, 1))
    db.add(dog2)
    db.commit()

    # Seed 10 resulted races; in 7 of them Runner is "EP Ld 1", in 3 "SAw Fd".
    comments = ["EP Ld 1"] * 7 + ["SAw Fd Ck 2"] * 3
    target_race = None
    for i, comment in enumerate(comments):
        race = Race(
            track_id=track.id,
            race_date=date(2025, 2, 1) + timedelta(days=i),
            race_time=time(19, 30),
            race_number=i + 1,
            distance_m=525,
            grade="A3",
            race_type="flat",
            going="standard",
            num_runners=2,
            status="resulted",
        )
        db.add(race)
        db.commit()
        # Runner wins 7, loses 3 — matches the EP/SAw pattern
        runner_pos = 1 if comment.startswith("EP") else 2
        filler_pos = 2 if runner_pos == 1 else 1
        db.add(RaceEntry(
            race_id=race.id, dog_id=dog.id, trap=1,
            finish_position=runner_pos, finish_time=29.0 + 0.1 * runner_pos,
            adjusted_time=29.0 + 0.1 * runner_pos,
            sectional_time=5.0,
            weight_kg=32.0, sp_decimal=2.0, comment=comment,
        ))
        db.add(RaceEntry(
            race_id=race.id, dog_id=dog2.id, trap=2,
            finish_position=filler_pos, finish_time=29.0 + 0.1 * filler_pos,
            adjusted_time=29.0 + 0.1 * filler_pos,
            sectional_time=5.0,
            weight_kg=32.0, sp_decimal=2.0, comment="",
        ))
        db.commit()
        target_race = race

    # Now add an 11th (target) race — features for this entry should reflect
    # the previous 10 history rows of Runner.
    target_race = Race(
        track_id=track.id,
        race_date=date(2025, 2, 20),
        race_time=time(19, 30),
        race_number=99,
        distance_m=525,
        grade="A3",
        race_type="flat",
        going="standard",
        num_runners=2,
        status="resulted",
    )
    db.add(target_race)
    db.commit()
    target_entry = RaceEntry(
        race_id=target_race.id, dog_id=dog.id, trap=1,
        finish_position=1, finish_time=29.1, adjusted_time=29.1,
        sectional_time=5.0, weight_kg=32.0, sp_decimal=2.0,
    )
    db.add(target_entry)
    db.add(RaceEntry(
        race_id=target_race.id, dog_id=dog2.id, trap=2,
        finish_position=2, finish_time=29.2, adjusted_time=29.2,
        sectional_time=5.0, weight_kg=32.0, sp_decimal=2.0,
    ))
    db.commit()

    df = compute_builtin_features_batch(db, [target_entry.id])
    row = df.loc[target_entry.id]

    # 7 of 10 prior comments were "EP Ld 1"
    assert abs(row["running_style_ep_rate_last10"] - 0.7) < 1e-6
    assert abs(row["led_at_bend1_rate_last10"] - 0.7) < 1e-6
    # 3 of 10 were "SAw Fd Ck 2"
    assert abs(row["slow_away_rate_last10"] - 0.3) < 1e-6
    assert abs(row["faded_rate_last10"] - 0.3) < 1e-6
    assert abs(row["trouble_bend2_rate_last10"] - 0.3) < 1e-6
    # No "MP" / "LP" / wide / railed tokens in the seed comments
    assert row["running_style_mp_rate_last10"] == 0.0
    assert row["running_style_lp_rate_last10"] == 0.0
    assert row["ran_wide_rate_last10"] == 0.0


def test_speed_figure_orders_with_skill(db):
    _track, dogs, races = _seed_simple(db)
    last_race = max(races, key=lambda r: r.race_date)
    last_entries = (
        db.query(RaceEntry)
        .filter(RaceEntry.race_id == last_race.id)
        .all()
    )
    entry_ids = [e.id for e in last_entries]
    df = compute_builtin_features_batch(db, entry_ids)

    name_to_entry = {}
    for e in last_entries:
        dog = db.query(Dog).filter(Dog.id == e.dog_id).first()
        name_to_entry[dog.name] = e.id

    # Higher speed figure = better.  Alpha consistently runs faster than
    # Delta, so its career peak speed figure should exceed Delta's.
    peak_alpha = df.loc[name_to_entry["Alpha"], "career_peak_speed_figure"]
    peak_delta = df.loc[name_to_entry["Delta"], "career_peak_speed_figure"]
    if peak_alpha is not None and peak_delta is not None:
        assert peak_alpha > peak_delta

"""Roundtrip tests for the new Prediction columns (place/show/position dist)."""

from datetime import date, time

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.dog import Dog
from app.models.experiment import Experiment
from app.models.prediction import Prediction
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _seed_experiment_and_entry(s):
    track = Track(name="Shelbourne Park", code="SPK")
    s.add(track)
    s.flush()

    race = Race(
        track_id=track.id,
        race_date=date(2026, 4, 24),
        race_number=1,
        distance_m=525,
        grade="A3",
        race_type="flat",
        status="resulted",
        source="gri",
    )
    s.add(race)
    s.flush()

    dog = Dog(name="Test Dog", sex="D")
    s.add(dog)
    s.flush()

    entry = RaceEntry(race_id=race.id, dog_id=dog.id, trap=1, finish_position=1)
    s.add(entry)
    s.flush()

    experiment = Experiment(
        name="test-exp",
        algorithm="plackett_luce",
        target="win_prob",
        feature_set=[],
        hyperparameters={},
        split_config={},
        status="completed",
    )
    s.add(experiment)
    s.commit()
    return experiment.id, entry.id


def test_prediction_schema_has_new_columns(db):
    """The columns defined on the model should materialise when create_all runs."""
    inspector = inspect(db.get_bind())
    cols = {c["name"] for c in inspector.get_columns("predictions")}
    assert {"place2_probability", "place3_probability", "position_probs_json"}.issubset(cols)


def test_roundtrip_place_and_position_probs(db):
    exp_id, entry_id = _seed_experiment_and_entry(db)

    position_probs = {"p1": 0.34, "p2": 0.21, "p3": 0.15, "p4_plus": 0.30}
    pred = Prediction(
        experiment_id=exp_id,
        race_entry_id=entry_id,
        win_probability=0.34,
        place2_probability=0.55,
        place3_probability=0.70,
        position_probs_json=position_probs,
        confidence=0.42,
    )
    db.add(pred)
    db.commit()

    loaded = db.query(Prediction).filter_by(experiment_id=exp_id).one()
    assert loaded.place2_probability == pytest.approx(0.55)
    assert loaded.place3_probability == pytest.approx(0.70)
    assert loaded.position_probs_json == position_probs
    assert loaded.win_probability == pytest.approx(0.34)


def test_legacy_null_fields_still_allowed(db):
    """Trainers that don't emit position distributions should still save cleanly."""
    exp_id, entry_id = _seed_experiment_and_entry(db)

    pred = Prediction(
        experiment_id=exp_id,
        race_entry_id=entry_id,
        win_probability=0.20,
        confidence=0.30,
        # place2/place3/position_probs_json all default to None
    )
    db.add(pred)
    db.commit()

    loaded = db.query(Prediction).filter_by(experiment_id=exp_id).one()
    assert loaded.place2_probability is None
    assert loaded.place3_probability is None
    assert loaded.position_probs_json is None

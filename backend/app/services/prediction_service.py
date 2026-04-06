"""
Prediction service: generate predictions for races using a trained model.

1. Load a trained model from an experiment
2. Compute features for the target race entries
3. Generate predictions (win probability, position, time)
4. Normalize win probabilities within each race to sum to ~1.0
5. Save predictions to DB
"""

import logging
import os
from datetime import date, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models.experiment import Experiment
from app.models.prediction import Prediction
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.dog import Dog
from app.models.track import Track
from app.services.feature_engine import get_dog_history, get_race_context, compute_visual_feature
from app.services.feature_sandbox import execute_feature_code
from app.models.feature_definition import FeatureDefinition

logger = logging.getLogger(__name__)


def load_trained_model(experiment: Experiment):
    """Load a trained model from disk."""
    if not experiment.model_path or not os.path.exists(experiment.model_path):
        raise FileNotFoundError(f"Model file not found: {experiment.model_path}")
    return joblib.load(experiment.model_path)


def compute_features_for_entries(
    db: Session,
    entry_ids: list[int],
    feature_defs: list[FeatureDefinition],
) -> pd.DataFrame:
    """Compute features for specific race entries on-the-fly (no caching)."""
    rows = {}

    for entry_id in entry_ids:
        ctx = get_race_context(db, entry_id)
        if not ctx:
            continue

        dog_id = ctx["dog_id"]
        race_date = ctx["race_date"]
        history = get_dog_history(db, dog_id, race_date)

        feature_values = {}
        for feat in feature_defs:
            value = None
            if feat.feature_type == "visual":
                config = feat.config_json or {}
                value = compute_visual_feature(history, config, ctx)
            elif feat.feature_type == "code" and feat.code:
                value, _ = execute_feature_code(feat.code, history, ctx)

            feature_values[feat.name] = value

        rows[entry_id] = feature_values

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "race_entry_id"
    return df


def _get_train_cutoff(experiment: Experiment) -> date | None:
    """Extract the training data cutoff date from an experiment's split config."""
    split_config = experiment.split_config or {}
    cutoff_str = split_config.get("train_cutoff_date")
    if cutoff_str:
        return date.fromisoformat(cutoff_str)
    return None


def predict_race(
    db: Session,
    experiment_id: int,
    race_id: int,
) -> list[dict[str, Any]]:
    """
    Generate predictions for all entries in a race.

    Returns list of prediction dicts with dog info.
    Raises ValueError if the race falls within the training data period.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment or experiment.status != "completed":
        raise ValueError(f"Experiment {experiment_id} not found or not completed")

    # Load model
    trainer = load_trained_model(experiment)

    # Get race entries
    entries = (
        db.query(RaceEntry, Dog.name.label("dog_name"), Race.race_date)
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.race_id == race_id)
        .order_by(RaceEntry.trap)
        .all()
    )

    # Guard against predicting on training data
    if entries:
        race_date = entries[0].race_date
        if isinstance(race_date, datetime):
            race_date = race_date.date()
        train_cutoff = _get_train_cutoff(experiment)
        if train_cutoff and race_date < train_cutoff:
            raise ValueError(
                f"Race date {race_date} falls within the training data period "
                f"(before {train_cutoff}). Predicting on training data would give "
                f"misleadingly optimistic results. Use a race dated after {train_cutoff}, "
                f"or retrain with an earlier cutoff."
            )

    if not entries:
        return []

    entry_ids = [e.RaceEntry.id for e in entries]

    # Get feature definitions
    feature_defs = (
        db.query(FeatureDefinition)
        .filter(FeatureDefinition.id.in_(experiment.feature_set))
        .all()
    )

    # Compute features
    X = compute_features_for_entries(db, entry_ids, feature_defs)

    if X.empty:
        return []

    # Fill NaN with 0 (same as training)
    X = X.fillna(X.median() if len(X) > 1 else 0)

    # Ensure columns match training
    feature_names = [f.name for f in feature_defs]
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    # Generate predictions
    predictions = []
    raw_proba = trainer.predict_proba(X) if hasattr(trainer, "predict_proba") else None
    raw_pred = trainer.predict(X)

    # Normalize win probabilities to sum to 1.0 within the race
    if raw_proba is not None:
        proba_sum = raw_proba.sum()
        if proba_sum > 0:
            normalized_proba = raw_proba / proba_sum
        else:
            normalized_proba = raw_proba
    else:
        normalized_proba = None

    for i, (entry_row, entry_id) in enumerate(zip(entries, entry_ids)):
        entry = entry_row.RaceEntry
        dog_name = entry_row.dog_name

        pred_data = {
            "race_entry_id": entry_id,
            "dog_name": dog_name,
            "trap": entry.trap,
            "experiment_id": experiment_id,
        }

        if experiment.target == "win_prob":
            pred_data["win_probability"] = float(normalized_proba[i]) if normalized_proba is not None else None
            pred_data["predicted_position"] = None
            pred_data["predicted_time"] = None
        elif experiment.target == "finish_position":
            pred_data["win_probability"] = None
            pred_data["predicted_position"] = float(raw_pred[i])
            pred_data["predicted_time"] = None
        elif experiment.target == "finish_time":
            pred_data["win_probability"] = None
            pred_data["predicted_position"] = None
            pred_data["predicted_time"] = float(raw_pred[i])

        predictions.append(pred_data)

    # Sort by win probability (highest first) or predicted position/time
    if experiment.target == "win_prob":
        predictions.sort(key=lambda p: p.get("win_probability") or 0, reverse=True)
    elif experiment.target == "finish_position":
        predictions.sort(key=lambda p: p.get("predicted_position") or 999)
    elif experiment.target == "finish_time":
        predictions.sort(key=lambda p: p.get("predicted_time") or 999)

    return predictions


def save_predictions(db: Session, predictions: list[dict[str, Any]]) -> int:
    """Save predictions to DB, upserting on (experiment_id, race_entry_id)."""
    saved = 0
    for pred in predictions:
        existing = (
            db.query(Prediction)
            .filter(
                Prediction.experiment_id == pred["experiment_id"],
                Prediction.race_entry_id == pred["race_entry_id"],
            )
            .first()
        )

        if existing:
            existing.win_probability = pred.get("win_probability")
            existing.predicted_position = pred.get("predicted_position")
            existing.predicted_time = pred.get("predicted_time")
            existing.created_at = datetime.utcnow()
        else:
            db.add(Prediction(
                experiment_id=pred["experiment_id"],
                race_entry_id=pred["race_entry_id"],
                win_probability=pred.get("win_probability"),
                predicted_position=pred.get("predicted_position"),
                predicted_time=pred.get("predicted_time"),
            ))
        saved += 1

    db.commit()
    return saved


def predict_upcoming_races(
    db: Session,
    experiment_id: int,
) -> list[dict[str, Any]]:
    """
    Generate predictions for all scheduled (upcoming) races.
    Returns grouped predictions by race.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment or experiment.status != "completed":
        raise ValueError(f"Experiment {experiment_id} not ready")

    # Find scheduled races
    scheduled_races = (
        db.query(Race, Track.name.label("track_name"), Track.code.label("track_code"))
        .join(Track)
        .filter(Race.status == "scheduled")
        .order_by(Race.race_date, Race.race_number)
        .all()
    )

    results = []
    for race_row in scheduled_races:
        race = race_row.Race
        try:
            preds = predict_race(db, experiment_id, race.id)
            if preds:
                save_predictions(db, preds)
                results.append({
                    "race_id": race.id,
                    "race_date": str(race.race_date),
                    "race_number": race.race_number,
                    "track_name": race_row.track_name,
                    "track_code": race_row.track_code,
                    "distance_m": race.distance_m,
                    "grade": race.grade,
                    "predictions": preds,
                })
        except Exception as e:
            logger.error("Failed to predict race %d: %s", race.id, e)

    return results

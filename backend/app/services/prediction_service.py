"""
Prediction service: generate predictions for races using a trained model.

1. Load a trained model (with preprocessing artifacts) from an experiment
2. Compute features for the target race entries
3. Generate predictions (win probability, position, time)
4. Compute confidence metrics (entropy, margin, edge)
5. Normalize win probabilities within each race via softmax
6. Compute bankroll-aware staking recommendations (Kelly criterion)
7. Save predictions to DB
"""

import logging
import math
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


def load_trained_model(experiment: Experiment) -> dict[str, Any]:
    """Load a trained model artifact from disk.

    Returns a dict with keys: trainer, feature_medians, is_ranking.
    Backwards-compatible with old models saved as raw trainer objects.
    """
    if not experiment.model_path or not os.path.exists(experiment.model_path):
        raise FileNotFoundError(f"Model file not found: {experiment.model_path}")

    artifact = joblib.load(experiment.model_path)

    # Backwards compatibility: old models were saved as the trainer directly
    if not isinstance(artifact, dict):
        return {
            "trainer": artifact,
            "feature_medians": {},
            "is_ranking": False,
        }

    return artifact


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


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def _compute_confidence(probs: np.ndarray) -> dict[str, float]:
    """Compute confidence metrics for a race's probability distribution.

    Returns:
        entropy: Normalized entropy (0 = totally confident, 1 = uniform/no opinion)
        margin: Probability gap between 1st and 2nd pick
        confidence_score: Combined score (0 to 1, higher = more confident)
        confidence_tier: "strong", "moderate", "weak", or "avoid"
    """
    n = len(probs)
    if n <= 1:
        return {
            "entropy": 0.0,
            "margin": 1.0,
            "confidence_score": 1.0,
            "confidence_tier": "strong",
        }

    # Normalized entropy: 0 = one dog certain, 1 = uniform distribution
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    max_entropy = math.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Margin between top two picks
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]

    # Combined confidence score: weight margin and inverse entropy
    # Margin is more important for betting decisions
    confidence_score = 0.6 * (margin * n) + 0.4 * (1 - normalized_entropy)
    confidence_score = max(0.0, min(1.0, confidence_score))

    # Tier assignment
    if confidence_score >= 0.65 and margin >= 0.08:
        tier = "strong"
    elif confidence_score >= 0.40 and margin >= 0.04:
        tier = "moderate"
    elif confidence_score >= 0.20:
        tier = "weak"
    else:
        tier = "avoid"

    return {
        "entropy": round(normalized_entropy, 4),
        "margin": round(margin, 4),
        "confidence_score": round(confidence_score, 4),
        "confidence_tier": tier,
    }


def _compute_kelly_stake(
    win_prob: float,
    odds_decimal: float | None,
    bankroll: float = 100.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
) -> dict[str, Any]:
    """Compute Kelly criterion stake for a bet.

    Args:
        win_prob: Model's estimated win probability
        odds_decimal: Decimal odds (e.g. 3.5 means +250)
        bankroll: Current bankroll
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly, safer)
        min_edge: Minimum edge required to place a bet (default 5%)

    Returns dict with stake info or None if no bet recommended.
    """
    if odds_decimal is None or odds_decimal <= 1.0:
        return {"bet": False, "reason": "no_odds"}

    implied_prob = 1.0 / odds_decimal
    edge = win_prob - implied_prob

    if edge < min_edge:
        return {
            "bet": False,
            "reason": "insufficient_edge",
            "edge": round(edge, 4),
            "implied_prob": round(implied_prob, 4),
        }

    # Kelly formula: f* = (bp - q) / b
    # where b = odds - 1, p = win_prob, q = 1 - win_prob
    b = odds_decimal - 1
    f_star = (b * win_prob - (1 - win_prob)) / b

    # Cap at kelly_fraction of full Kelly for safety
    fractional_kelly = max(0, f_star * kelly_fraction)

    # Also cap at 5% of bankroll as absolute max
    max_stake_pct = 0.05
    stake_pct = min(fractional_kelly, max_stake_pct)
    stake = round(bankroll * stake_pct, 2)

    return {
        "bet": True,
        "stake": stake,
        "stake_pct": round(stake_pct * 100, 2),
        "full_kelly_pct": round(f_star * 100, 2),
        "edge": round(edge, 4),
        "implied_prob": round(implied_prob, 4),
        "expected_value": round(win_prob * (odds_decimal - 1) - (1 - win_prob), 4),
    }


def predict_race(
    db: Session,
    experiment_id: int,
    race_id: int,
    bankroll: float = 100.0,
) -> list[dict[str, Any]]:
    """
    Generate predictions for all entries in a race.

    Returns list of prediction dicts with dog info, confidence metrics,
    and Kelly staking recommendations.
    Raises ValueError if the race falls within the training data period.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment or experiment.status != "completed":
        raise ValueError(f"Experiment {experiment_id} not found or not completed")

    # Load model artifact (trainer + preprocessing info)
    artifact = load_trained_model(experiment)
    trainer = artifact["trainer"]
    feature_medians = artifact.get("feature_medians", {})
    is_ranking = artifact.get("is_ranking", False)

    # Get race entries with SP odds
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

    # Fill NaN using training set medians (consistent with training)
    if feature_medians:
        X = X.fillna(feature_medians)
    # Any remaining NaN (new features, etc.) fill with 0
    X = X.fillna(0)

    # Ensure columns match training
    feature_names = [f.name for f in feature_defs]
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    # Add race-relative features (same as training)
    # All entries belong to one race, so create a constant race_id series
    from ml.dataset_builder import add_race_relative_features
    race_id_series = pd.Series(race_id, index=X.index, name="race_id")
    X = add_race_relative_features(X, race_id_series)

    # Fill any NaN in new relative features
    X = X.fillna(0)

    # Generate predictions
    raw_scores = trainer.predict(X)

    # Compute probabilities (calibration is built into the trainer)
    if is_ranking:
        # LambdaRank: softmax + isotonic calibration (built into scores_to_proba)
        win_probs = trainer.scores_to_proba(raw_scores, group_sizes=[len(X)])
    elif hasattr(trainer, "predict_proba"):
        # Pointwise classifiers: predict_proba now applies calibration internally
        raw_proba = trainer.predict_proba(X)
        if raw_proba is not None:
            # Normalize within race via softmax for multi-dog comparison
            win_probs = _softmax(np.log(np.clip(raw_proba, 1e-10, 1.0)))
        else:
            win_probs = None
    else:
        win_probs = None

    # Compute race-level confidence metrics
    race_confidence = _compute_confidence(win_probs) if win_probs is not None else None

    # Build prediction list
    predictions = []
    for i, (entry_row, entry_id) in enumerate(zip(entries, entry_ids)):
        entry = entry_row.RaceEntry
        dog_name = entry_row.dog_name

        pred_data = {
            "race_entry_id": entry_id,
            "dog_name": dog_name,
            "trap": entry.trap,
            "experiment_id": experiment_id,
        }

        if is_ranking or experiment.target == "win_prob":
            win_prob = float(win_probs[i]) if win_probs is not None else None
            pred_data["win_probability"] = win_prob
            pred_data["predicted_position"] = None
            pred_data["predicted_time"] = None

            # Confidence score for this dog (race-level confidence * individual probability)
            if race_confidence and win_prob is not None:
                pred_data["confidence"] = round(race_confidence["confidence_score"], 4)
                pred_data["confidence_tier"] = race_confidence["confidence_tier"]
                pred_data["margin"] = race_confidence["margin"]
                pred_data["entropy"] = race_confidence["entropy"]

            # Kelly staking recommendation
            if win_prob is not None:
                kelly = _compute_kelly_stake(
                    win_prob, entry.sp_decimal, bankroll=bankroll,
                )
                pred_data["kelly"] = kelly
            else:
                pred_data["kelly"] = {"bet": False, "reason": "no_probability"}

            # Edge vs market
            if win_prob is not None and entry.sp_decimal and entry.sp_decimal > 1:
                implied = 1.0 / entry.sp_decimal
                pred_data["edge"] = round(win_prob - implied, 4)
                pred_data["is_value"] = win_prob > implied * 1.05  # 5% min edge
            else:
                pred_data["edge"] = None
                pred_data["is_value"] = None

        elif experiment.target == "finish_position":
            pred_data["win_probability"] = None
            pred_data["predicted_position"] = float(raw_scores[i])
            pred_data["predicted_time"] = None
            pred_data["confidence"] = None
        elif experiment.target == "finish_time":
            pred_data["win_probability"] = None
            pred_data["predicted_position"] = None
            pred_data["predicted_time"] = float(raw_scores[i])
            pred_data["confidence"] = None

        predictions.append(pred_data)

    # Sort by win probability (highest first) or predicted position/time
    if is_ranking or experiment.target == "win_prob":
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
            existing.confidence = pred.get("confidence")
            existing.created_at = datetime.utcnow()
        else:
            db.add(Prediction(
                experiment_id=pred["experiment_id"],
                race_entry_id=pred["race_entry_id"],
                win_probability=pred.get("win_probability"),
                predicted_position=pred.get("predicted_position"),
                predicted_time=pred.get("predicted_time"),
                confidence=pred.get("confidence"),
            ))
        saved += 1

    db.commit()
    return saved


def predict_upcoming_races(
    db: Session,
    experiment_id: int,
    bankroll: float = 100.0,
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
            preds = predict_race(db, experiment_id, race.id, bankroll=bankroll)
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

"""Predictions API: generate, view, and manage race predictions."""

import logging
from datetime import date
from threading import Thread
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.database import get_db, SessionLocal
from app.models.dog import Dog
from app.models.prediction import Prediction
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track
from app.schemas.experiment import PredictionResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])


class PredictRaceRequest(BaseModel):
    experiment_id: int
    race_id: int


class PredictRaceResponse(BaseModel):
    race_id: int
    predictions: list[dict[str, Any]]
    saved: int


@router.get("/", response_model=list[PredictionResponse])
def list_predictions(
    experiment_id: int | None = None,
    race_id: int | None = None,
    date_from: date | None = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db),
):
    query = (
        db.query(Prediction, Dog.name.label("dog_name"), RaceEntry.trap)
        .join(RaceEntry, Prediction.race_entry_id == RaceEntry.id)
        .join(Dog, RaceEntry.dog_id == Dog.id)
    )
    if experiment_id:
        query = query.filter(Prediction.experiment_id == experiment_id)
    if race_id:
        query = query.join(Race, RaceEntry.race_id == Race.id).filter(Race.id == race_id)
    if date_from:
        if not race_id:
            query = query.join(Race, RaceEntry.race_id == Race.id)
        query = query.filter(Race.race_date >= date_from)

    rows = query.order_by(Prediction.win_probability.desc()).limit(limit).all()
    results = []
    for pred, dog_name, trap in rows:
        resp = PredictionResponse.model_validate(pred)
        resp.dog_name = dog_name
        resp.trap = trap
        results.append(resp)
    return results


@router.get("/preflight/{race_id}")
def predict_race_preflight(
    race_id: int,
    experiment_id: int,
    db: Session = Depends(get_db),
):
    """Diagnostic: surface missing-feature gaps for a race **without** predicting.

    Computes the feature matrix the model would see at predict time and
    reports any NaN cells per (entry, feature) plus any post-race-only
    features the experiment was trained on. Lets the UI warn the user
    before kicking off a real predict (which now fails loudly on missing
    data instead of silently imputing).
    """
    import pandas as pd

    from app.services.prediction_service import (
        compute_features_for_entries,
        load_trained_model,
    )
    from app.models.experiment import Experiment
    from app.models.feature_definition import FeatureDefinition
    from ml.feature_availability import post_race_features_in_use

    race = db.query(Race).filter(Race.id == race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    experiment = (
        db.query(Experiment).filter(Experiment.id == experiment_id).first()
    )
    if not experiment or experiment.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment {experiment_id} not found or not completed",
        )

    artifact = load_trained_model(experiment)
    trained_feature_names = artifact.get("feature_names", []) or []

    entries = (
        db.query(RaceEntry, Dog.name.label("dog_name"))
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .filter(RaceEntry.race_id == race_id)
        .order_by(RaceEntry.trap)
        .all()
    )
    entry_ids = [e.RaceEntry.id for e in entries]
    entry_lookup = {e.RaceEntry.id: (e.RaceEntry.trap, e.dog_name) for e in entries}

    split_cfg = experiment.split_config or {}
    include_builtin = split_cfg.get("include_builtin_features", True)
    feature_defs = (
        db.query(FeatureDefinition)
        .filter(FeatureDefinition.id.in_(experiment.feature_set))
        .all()
    )

    X = compute_features_for_entries(
        db, entry_ids, feature_defs, include_builtin=include_builtin,
    )
    X = X.apply(pd.to_numeric, errors="coerce") if not X.empty else X

    missing_per_feature: list[dict[str, Any]] = []
    if not X.empty:
        for col in X.columns[X.isna().any()]:
            offenders = []
            for eid in X.index[X[col].isna()].tolist():
                trap, dog_name = entry_lookup.get(int(eid), (None, None))
                offenders.append({"entry_id": int(eid), "trap": trap, "dog_name": dog_name})
            missing_per_feature.append({"feature": str(col), "missing_for": offenders})

    post_race_used = post_race_features_in_use(trained_feature_names)

    return {
        "race_id": race_id,
        "race_status": race.status,
        "experiment_id": experiment_id,
        "n_entries": len(entries),
        "post_race_features_in_use": [
            {"feature": name, "reason": reason}
            for name, reason in post_race_used.items()
        ],
        "missing_features": missing_per_feature,
        "would_fail": bool(
            race.status == "scheduled"
            and (post_race_used or missing_per_feature)
        ),
    }


@router.get("/race/{race_id}")
def predict_single_race(
    race_id: int,
    experiment_id: int,
    bankroll: float = Query(default=100.0, ge=1),
    db: Session = Depends(get_db),
):
    """Generate predictions for a specific race using a trained model."""
    from app.services.prediction_service import predict_race, save_predictions

    race = db.query(Race).filter(Race.id == race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    try:
        preds = predict_race(db, experiment_id, race_id, bankroll=bankroll)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if preds:
        saved = save_predictions(db, preds)
    else:
        saved = 0

    # Add race info
    track = db.query(Track).filter(Track.id == race.track_id).first()

    return {
        "race_id": race_id,
        "race_date": str(race.race_date),
        "race_number": race.race_number,
        "track_name": track.name if track else None,
        "distance_m": race.distance_m,
        "grade": race.grade,
        "predictions": preds,
        "saved": saved,
    }


@router.get("/races-for-date")
def get_races_for_date(
    race_date: date,
    track_code: str | None = None,
    db: Session = Depends(get_db),
):
    """Get all races for a given date, optionally filtered by track.
    Used by the race picker in the predictions UI."""
    query = (
        db.query(
            Race.id,
            Race.race_number,
            Race.distance_m,
            Race.grade,
            Race.status,
            Race.race_date,
            Track.name.label("track_name"),
            Track.code.label("track_code"),
        )
        .join(Track, Race.track_id == Track.id)
        .filter(Race.race_date == race_date)
    )

    if track_code:
        query = query.filter(Track.code == track_code)

    query = query.order_by(Track.name, Race.race_number)
    rows = query.all()

    return [
        {
            "id": r.id,
            "race_number": r.race_number,
            "distance_m": r.distance_m,
            "grade": r.grade,
            "status": r.status,
            "race_date": str(r.race_date),
            "track_name": r.track_name,
            "track_code": r.track_code,
        }
        for r in rows
    ]


@router.get("/by-date")
def predict_races_by_date(
    race_date: date,
    experiment_id: int,
    track_code: str | None = None,
    bankroll: float = Query(default=100.0, ge=1),
    only_scheduled: bool = Query(
        default=True,
        description="If true, only predict races with status='scheduled' (skip resulted)",
    ),
    db: Session = Depends(get_db),
):
    """Generate predictions for every race on a given date.

    Default `only_scheduled=true` matches the future-races flow: scrape via
    `/scraping/scrape-date` first (step 1), then call this endpoint (step 2).
    Pass `only_scheduled=false` to also predict already-resulted races (e.g.
    for back-testing on a historical date).

    The expensive ELO chronological sweep runs once for *all* entries on the
    date, not once per race. With a typical Irish card (4-6 tracks × ~10
    races) this keeps the request well under the gateway timeout.
    """
    from app.services.prediction_service import (
        compute_features_for_entries,
        predict_race,
        save_predictions,
    )
    from app.models.experiment import Experiment
    from app.models.feature_definition import FeatureDefinition

    query = (
        db.query(Race, Track.name.label("track_name"), Track.code.label("track_code"))
        .join(Track, Race.track_id == Track.id)
        .filter(Race.race_date == race_date)
    )
    if only_scheduled:
        query = query.filter(Race.status == "scheduled")
    if track_code:
        query = query.filter(Track.code == track_code)
    rows = query.order_by(Track.name, Race.race_number).all()

    if not rows:
        return {
            "race_date": str(race_date),
            "experiment_id": experiment_id,
            "races_predicted": 0,
            "races_failed": 0,
            "races": [],
            "errors": [],
        }

    # Validate experiment up front so failures here surface as a structured
    # 4xx instead of dying inside the per-race loop with a generic 500.
    experiment = (
        db.query(Experiment).filter(Experiment.id == experiment_id).first()
    )
    if not experiment or experiment.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment {experiment_id} not found or not completed",
        )

    # Single batched feature computation across every entry on the card.
    # The ELO pass walks ~84k historical races regardless of target count,
    # so doing it once for N races is dramatically cheaper than N times.
    race_ids = [row.Race.id for row in rows]
    all_entry_ids = [
        eid for (eid,) in db.query(RaceEntry.id)
        .filter(RaceEntry.race_id.in_(race_ids))
        .all()
    ]

    precomputed = None
    if all_entry_ids:
        try:
            split_cfg = experiment.split_config or {}
            include_builtin = split_cfg.get("include_builtin_features", True)
            feature_defs = (
                db.query(FeatureDefinition)
                .filter(FeatureDefinition.id.in_(experiment.feature_set))
                .all()
            )
            logger.info(
                "by-date: batch-computing features for %d entries across %d races",
                len(all_entry_ids), len(race_ids),
            )
            precomputed = compute_features_for_entries(
                db, all_entry_ids, feature_defs, include_builtin=include_builtin,
            )
        except Exception as e:
            logger.exception("by-date: batched feature compute failed")
            raise HTTPException(
                status_code=500,
                detail=f"Feature batch computation failed: {e}",
            )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for row in rows:
        race = row.Race
        try:
            preds = predict_race(
                db, experiment_id, race.id, bankroll=bankroll,
                precomputed_features=precomputed,
            )
            if preds:
                save_predictions(db, preds)
                results.append({
                    "race_id": race.id,
                    "race_date": str(race.race_date),
                    "race_time": str(race.race_time) if race.race_time else None,
                    "race_number": race.race_number,
                    "track_name": row.track_name,
                    "track_code": row.track_code,
                    "distance_m": race.distance_m,
                    "grade": race.grade,
                    "status": race.status,
                    "predictions": preds,
                })
        except Exception as e:
            logger.warning("by-date: race %d (%s R%s) failed: %s",
                           race.id, row.track_code, race.race_number, e)
            errors.append({
                "race_id": race.id,
                "track_code": row.track_code,
                "race_number": race.race_number,
                "error": str(e),
            })

    return {
        "race_date": str(race_date),
        "experiment_id": experiment_id,
        "races_predicted": len(results),
        "races_failed": len(errors),
        "races": results,
        "errors": errors,
    }


@router.get("/upcoming")
def get_upcoming_predictions(
    experiment_id: int,
    bankroll: float = Query(default=100.0, ge=1),
    db: Session = Depends(get_db),
):
    """Get or generate predictions for all scheduled (upcoming) races."""
    from app.services.prediction_service import predict_upcoming_races

    try:
        results = predict_upcoming_races(db, experiment_id, bankroll=bankroll)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "experiment_id": experiment_id,
        "races_predicted": len(results),
        "races": results,
    }


@router.get("/race/{race_id}/ensemble")
def predict_race_ensemble(
    race_id: int,
    experiment_ids: str = Query(..., description="Comma-separated experiment IDs"),
    weights: str | None = Query(default=None, description="Comma-separated weights (must match experiment_ids)"),
    bankroll: float = Query(default=100.0, ge=1),
    db: Session = Depends(get_db),
):
    """Generate ensemble predictions by combining multiple trained models."""
    from app.services.prediction_service import predict_race_ensemble as ensemble_predict

    race = db.query(Race).filter(Race.id == race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    exp_ids = [int(x.strip()) for x in experiment_ids.split(",")]
    w = [float(x.strip()) for x in weights.split(",")] if weights else None

    try:
        preds = ensemble_predict(db, exp_ids, race_id, weights=w, bankroll=bankroll)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    track = db.query(Track).filter(Track.id == race.track_id).first()

    return {
        "race_id": race_id,
        "race_date": str(race.race_date),
        "race_number": race.race_number,
        "track_name": track.name if track else None,
        "distance_m": race.distance_m,
        "grade": race.grade,
        "ensemble_experiment_ids": exp_ids,
        "predictions": preds,
    }


@router.get("/best-bets")
def get_best_bets(
    experiment_id: int,
    race_date: date,
    bankroll: float = Query(default=100.0, ge=1),
    min_confidence: str = Query(default="moderate", description="Minimum confidence tier: strong, moderate, weak"),
    min_edge: float = Query(default=0.05, description="Minimum edge over market odds"),
    limit: int = Query(default=10, le=50),
    db: Session = Depends(get_db),
):
    """
    Get the best betting opportunities for a given date.

    Scans all races for the date, generates predictions, and returns
    the top value bets ranked by edge — filtered by confidence tier.
    """
    from app.services.prediction_service import predict_race, save_predictions

    tier_order = {"strong": 3, "moderate": 2, "weak": 1, "avoid": 0}
    min_tier_value = tier_order.get(min_confidence, 2)

    # Get all races for the date
    races = (
        db.query(Race, Track.name.label("track_name"), Track.code.label("track_code"))
        .join(Track, Race.track_id == Track.id)
        .filter(Race.race_date == race_date)
        .order_by(Track.name, Race.race_number)
        .all()
    )

    if not races:
        return {"race_date": str(race_date), "best_bets": [], "races_scanned": 0}

    all_value_bets = []
    races_scanned = 0

    for race_row in races:
        race = race_row.Race
        try:
            preds = predict_race(db, experiment_id, race.id, bankroll=bankroll)
            races_scanned += 1

            if preds:
                save_predictions(db, preds)

            for pred in preds:
                edge = pred.get("edge")
                kelly = pred.get("kelly", {})
                confidence_tier = pred.get("confidence_tier", "avoid")
                tier_value = tier_order.get(confidence_tier, 0)

                if (
                    edge is not None
                    and edge >= min_edge
                    and kelly.get("bet", False)
                    and tier_value >= min_tier_value
                ):
                    all_value_bets.append({
                        "race_id": race.id,
                        "race_number": race.race_number,
                        "track_name": race_row.track_name,
                        "track_code": race_row.track_code,
                        "distance_m": race.distance_m,
                        "grade": race.grade,
                        "dog_name": pred["dog_name"],
                        "trap": pred["trap"],
                        "win_probability": pred.get("win_probability"),
                        "edge": edge,
                        "confidence_tier": confidence_tier,
                        "confidence_score": pred.get("confidence"),
                        "kelly_stake": kelly.get("stake"),
                        "kelly_stake_pct": kelly.get("stake_pct"),
                        "expected_value": kelly.get("expected_value"),
                        "implied_prob": kelly.get("implied_prob"),
                    })
        except Exception as e:
            logger.warning("Best bets: race %d failed: %s", race.id, e)

    # Sort by edge (highest first) and limit
    all_value_bets.sort(key=lambda b: b["edge"], reverse=True)
    best_bets = all_value_bets[:limit]

    return {
        "race_date": str(race_date),
        "experiment_id": experiment_id,
        "races_scanned": races_scanned,
        "total_value_bets": len(all_value_bets),
        "best_bets": best_bets,
    }


@router.get("/results-comparison")
def results_comparison(
    experiment_id: int,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db),
):
    """
    Compare predictions vs actual results for resulted races.
    Shows how the model performed on past races.
    """
    rows = (
        db.query(
            Prediction,
            RaceEntry.finish_position,
            RaceEntry.finish_time,
            RaceEntry.trap,
            RaceEntry.sp_decimal,
            Dog.name.label("dog_name"),
            Race.race_date,
            Race.race_number,
            Race.grade,
            Track.name.label("track_name"),
        )
        .join(RaceEntry, Prediction.race_entry_id == RaceEntry.id)
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Track, Race.track_id == Track.id)
        .filter(
            Prediction.experiment_id == experiment_id,
            Race.status == "resulted",
            RaceEntry.finish_position.isnot(None),
        )
        .order_by(Race.race_date.desc(), Race.race_number, Prediction.win_probability.desc())
        .limit(limit)
        .all()
    )

    results = []
    for row in rows:
        pred = row.Prediction
        results.append({
            "race_date": str(row.race_date),
            "race_number": row.race_number,
            "track_name": row.track_name,
            "grade": row.grade,
            "dog_name": row.dog_name,
            "trap": row.trap,
            "win_probability": pred.win_probability,
            "predicted_position": pred.predicted_position,
            "predicted_time": pred.predicted_time,
            "confidence": pred.confidence,
            "actual_position": row.finish_position,
            "actual_time": row.finish_time,
            "sp_decimal": row.sp_decimal,
            "won": row.finish_position == 1,
            "edge": round(pred.win_probability - 1.0 / row.sp_decimal, 4)
                if pred.win_probability and row.sp_decimal and row.sp_decimal > 1 else None,
            "value": (pred.win_probability or 0) > (1 / (row.sp_decimal or 999)) * 1.05
                if pred.win_probability and row.sp_decimal else None,
        })

    return {
        "experiment_id": experiment_id,
        "total_comparisons": len(results),
        "results": results,
    }

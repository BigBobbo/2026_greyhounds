"""Predictions API: generate, view, and manage race predictions."""

from datetime import date
from threading import Thread
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

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

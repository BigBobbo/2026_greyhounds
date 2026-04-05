from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dog import Dog
from app.models.prediction import Prediction
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.schemas.experiment import PredictionResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])


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
        query = query.join(Race, RaceEntry.race_id == Race.id).filter(Race.race_date >= date_from)

    rows = query.limit(limit).all()
    results = []
    for pred, dog_name, trap in rows:
        resp = PredictionResponse.model_validate(pred)
        resp.dog_name = dog_name
        resp.trap = trap
        results.append(resp)
    return results

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentResponse

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/experiments", response_model=list[ExperimentResponse])
def list_experiments(
    status: str | None = None,
    target: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(Experiment)
    if status:
        query = query.filter(Experiment.status == status)
    if target:
        query = query.filter(Experiment.target == target)
    return query.order_by(Experiment.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
def create_experiment(experiment: ExperimentCreate, db: Session = Depends(get_db)):
    db_experiment = Experiment(
        name=experiment.name,
        description=experiment.description,
        algorithm=experiment.algorithm,
        target=experiment.target,
        hyperparameters=experiment.hyperparameters,
        feature_set=experiment.feature_set,
        split_config=experiment.split_config,
        status="pending",
    )
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    # TODO: trigger training background task
    return db_experiment

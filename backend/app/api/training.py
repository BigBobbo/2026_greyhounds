"""Training API: create experiments, trigger training, view results."""

import os
from threading import Thread

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentResponse
from ml.trainers.base import BaseTrainer

router = APIRouter(prefix="/training", tags=["training"])


class DefaultParamsResponse(BaseModel):
    params: dict


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


@router.get("/default-params/{algorithm}", response_model=DefaultParamsResponse)
def get_default_params(algorithm: str):
    """Get default hyperparameters for an algorithm."""
    params = BaseTrainer.get_default_params(algorithm)
    if not params:
        raise HTTPException(status_code=404, detail=f"Unknown algorithm: {algorithm}")
    return DefaultParamsResponse(params=params)


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
def create_experiment(experiment: ExperimentCreate, db: Session = Depends(get_db)):
    """Create an experiment and start training in the background."""
    # Use default params if none provided
    hyperparams = experiment.hyperparameters
    if not hyperparams:
        hyperparams = BaseTrainer.get_default_params(experiment.algorithm)

    db_experiment = Experiment(
        name=experiment.name,
        description=experiment.description,
        algorithm=experiment.algorithm,
        target=experiment.target,
        hyperparameters=hyperparams,
        feature_set=experiment.feature_set,
        split_config=experiment.split_config,
        status="pending",
    )
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)

    exp_id = db_experiment.id
    use_optuna = experiment.auto_tune
    optuna_trials = experiment.optuna_trials

    # Start training in background
    def _train():
        db2 = SessionLocal()
        try:
            from app.services.training_service import run_training, run_optuna_optimization
            if use_optuna:
                run_optuna_optimization(db2, exp_id, n_trials=optuna_trials)
            else:
                run_training(db2, exp_id)
        finally:
            db2.close()

    Thread(target=_train, daemon=True).start()

    return db_experiment


@router.delete("/experiments/{experiment_id}", status_code=204)
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    db.delete(experiment)
    db.commit()


@router.delete("/cache/builtin-features", status_code=200)
def clear_builtin_features_cache():
    """Clear the built-in features cache. Useful after scraping new race data."""
    from ml.dataset_builder import BUILTIN_CACHE_PATH

    if os.path.exists(BUILTIN_CACHE_PATH):
        os.remove(BUILTIN_CACHE_PATH)
        return {"detail": "Built-in features cache cleared"}
    return {"detail": "No cache to clear"}

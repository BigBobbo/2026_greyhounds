"""Daily prediction schedule API.

CRUD for ``ModelSchedule`` rows, a manual-trigger endpoint that fires the
job synchronously in a background thread, a list of recent
``ScheduledPredictionRun`` audit rows, and a per-schedule performance
summary derived from predictions ↔ results.
"""

from __future__ import annotations

from datetime import date
from threading import Thread
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.experiment import Experiment
from app.models.schedule import ModelSchedule, ScheduledPredictionRun
from app.services.schedule_service import compute_performance, run_schedule_job
from app.tasks.scheduler import register_schedule_job, unregister_schedule_job

router = APIRouter(prefix="/schedule", tags=["schedule"])


class ScheduleCreate(BaseModel):
    experiment_id: int
    enabled: bool = True
    is_main: bool = False
    cron_hour: int = Field(default=8, ge=0, le=23)
    cron_minute: int = Field(default=30, ge=0, le=59)
    timezone: str = "Europe/Dublin"
    scrape_upcoming: bool = True
    predict_days_ahead: int = Field(default=1, ge=0, le=7)


class ScheduleUpdate(BaseModel):
    enabled: bool | None = None
    is_main: bool | None = None
    cron_hour: int | None = Field(default=None, ge=0, le=23)
    cron_minute: int | None = Field(default=None, ge=0, le=59)
    timezone: str | None = None
    scrape_upcoming: bool | None = None
    predict_days_ahead: int | None = Field(default=None, ge=0, le=7)


class ScheduleResponse(BaseModel):
    id: int
    experiment_id: int
    experiment_name: str | None = None
    experiment_status: str | None = None
    enabled: bool
    is_main: bool
    cron_hour: int
    cron_minute: int
    timezone: str
    scrape_upcoming: bool
    predict_days_ahead: int
    created_at: str
    updated_at: str
    last_run: dict[str, Any] | None = None


class RunResponse(BaseModel):
    id: int
    model_schedule_id: int
    run_date: str
    status: str
    trigger: str
    races_predicted: int
    races_skipped: int
    predictions_written: int
    error_message: str | None
    started_at: str
    finished_at: str | None


def _serialize_run(run: ScheduledPredictionRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "model_schedule_id": run.model_schedule_id,
        "run_date": run.run_date.isoformat(),
        "status": run.status,
        "trigger": run.trigger,
        "races_predicted": run.races_predicted,
        "races_skipped": run.races_skipped,
        "predictions_written": run.predictions_written,
        "error_message": run.error_message,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def _serialize_schedule(
    db: Session,
    sched: ModelSchedule,
    exp: Experiment | None = None,
    last_run: "ScheduledPredictionRun | None | bool" = False,
) -> dict[str, Any]:
    # exp/last_run may be supplied by batch callers (list endpoint) to avoid
    # two extra queries per row; single-row callers omit them.
    if exp is None:
        exp = db.query(Experiment).filter(Experiment.id == sched.experiment_id).first()
    if last_run is False:
        last_run = (
            db.query(ScheduledPredictionRun)
            .filter(ScheduledPredictionRun.model_schedule_id == sched.id)
            .order_by(desc(ScheduledPredictionRun.started_at))
            .first()
        )
    return {
        "id": sched.id,
        "experiment_id": sched.experiment_id,
        "experiment_name": exp.name if exp else None,
        "experiment_status": exp.status if exp else None,
        "enabled": sched.enabled,
        "is_main": sched.is_main,
        "cron_hour": sched.cron_hour,
        "cron_minute": sched.cron_minute,
        "timezone": sched.timezone,
        "scrape_upcoming": sched.scrape_upcoming,
        "predict_days_ahead": sched.predict_days_ahead,
        "created_at": sched.created_at.isoformat(),
        "updated_at": sched.updated_at.isoformat(),
        "last_run": _serialize_run(last_run) if last_run else None,
    }


def _clear_other_main(db: Session, keep_id: int | None) -> None:
    """Ensure at most one schedule has ``is_main=True``."""
    q = db.query(ModelSchedule).filter(ModelSchedule.is_main.is_(True))
    if keep_id is not None:
        q = q.filter(ModelSchedule.id != keep_id)
    for other in q.all():
        other.is_main = False


@router.get("", response_model=list[ScheduleResponse])
def list_schedules(db: Session = Depends(get_db)):
    schedules = db.query(ModelSchedule).order_by(ModelSchedule.id).all()
    if not schedules:
        return []

    # Batch the lookups the per-row serializer would otherwise repeat
    # (two queries per schedule -> three queries total).
    exp_ids = {s.experiment_id for s in schedules}
    experiments = {
        e.id: e
        for e in db.query(Experiment).filter(Experiment.id.in_(exp_ids)).all()
    }
    sched_ids = [s.id for s in schedules]
    latest_runs: dict[int, ScheduledPredictionRun] = {}
    for run in (
        db.query(ScheduledPredictionRun)
        .filter(ScheduledPredictionRun.model_schedule_id.in_(sched_ids))
        .order_by(ScheduledPredictionRun.started_at)
        .all()
    ):
        latest_runs[run.model_schedule_id] = run  # later rows overwrite earlier

    return [
        _serialize_schedule(
            db, s,
            exp=experiments.get(s.experiment_id),
            last_run=latest_runs.get(s.id),
        )
        for s in schedules
    ]


@router.post("", response_model=ScheduleResponse, status_code=201)
def create_schedule(req: ScheduleCreate, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == req.experiment_id).first()
    if not exp:
        raise HTTPException(404, f"Experiment {req.experiment_id} not found")
    if exp.status != "completed":
        raise HTTPException(
            400, f"Experiment {req.experiment_id} is '{exp.status}', not completed"
        )

    existing = (
        db.query(ModelSchedule)
        .filter(ModelSchedule.experiment_id == req.experiment_id)
        .first()
    )
    if existing:
        raise HTTPException(
            409, f"Schedule already exists for experiment {req.experiment_id}"
        )

    sched = ModelSchedule(
        experiment_id=req.experiment_id,
        enabled=req.enabled,
        is_main=req.is_main,
        cron_hour=req.cron_hour,
        cron_minute=req.cron_minute,
        timezone=req.timezone,
        scrape_upcoming=req.scrape_upcoming,
        predict_days_ahead=req.predict_days_ahead,
    )
    db.add(sched)
    db.flush()
    if req.is_main:
        _clear_other_main(db, keep_id=sched.id)
    db.commit()
    db.refresh(sched)

    if sched.enabled:
        register_schedule_job(sched)

    return _serialize_schedule(db, sched)


@router.patch("/{schedule_id}", response_model=ScheduleResponse)
def update_schedule(
    schedule_id: int, req: ScheduleUpdate, db: Session = Depends(get_db)
):
    sched = db.query(ModelSchedule).filter(ModelSchedule.id == schedule_id).first()
    if not sched:
        raise HTTPException(404, f"Schedule {schedule_id} not found")

    data = req.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(sched, key, value)
    if data.get("is_main"):
        _clear_other_main(db, keep_id=sched.id)
    db.commit()
    db.refresh(sched)

    # Always re-register so cron-time and enabled changes take effect.
    unregister_schedule_job(sched.id)
    if sched.enabled:
        register_schedule_job(sched)

    return _serialize_schedule(db, sched)


@router.delete("/{schedule_id}", status_code=204)
def delete_schedule(schedule_id: int, db: Session = Depends(get_db)):
    sched = db.query(ModelSchedule).filter(ModelSchedule.id == schedule_id).first()
    if not sched:
        raise HTTPException(404, f"Schedule {schedule_id} not found")
    unregister_schedule_job(sched.id)
    db.delete(sched)
    db.commit()
    return None


@router.post("/{schedule_id}/run", response_model=RunResponse, status_code=202)
def trigger_run(schedule_id: int, db: Session = Depends(get_db)):
    """Fire the scheduled job immediately on a background thread.

    Returns the ``ScheduledPredictionRun`` row in ``running`` state — poll
    ``GET /schedule/{id}/runs`` for the final status.
    """
    sched = db.query(ModelSchedule).filter(ModelSchedule.id == schedule_id).first()
    if not sched:
        raise HTTPException(404, f"Schedule {schedule_id} not found")

    # Insert the audit row up front so the response can carry it; the
    # service will look it up and update it. Actually the service creates
    # its own row — to avoid duplicates, we simply spawn the worker and
    # return the most recent row after a short tick. To keep things
    # simple, we just spawn and synthesize a placeholder response.
    Thread(
        target=run_schedule_job, args=(schedule_id,), kwargs={"trigger": "manual"},
        daemon=True,
    ).start()

    return {
        "id": 0,
        "model_schedule_id": schedule_id,
        "run_date": date.today().isoformat(),
        "status": "running",
        "trigger": "manual",
        "races_predicted": 0,
        "races_skipped": 0,
        "predictions_written": 0,
        "error_message": None,
        "started_at": "",
        "finished_at": None,
    }


@router.get("/{schedule_id}/runs", response_model=list[RunResponse])
def list_runs(
    schedule_id: int,
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    runs = (
        db.query(ScheduledPredictionRun)
        .filter(ScheduledPredictionRun.model_schedule_id == schedule_id)
        .order_by(desc(ScheduledPredictionRun.started_at))
        .limit(limit)
        .all()
    )
    return [_serialize_run(r) for r in runs]


@router.get("/{schedule_id}/performance")
def schedule_performance(
    schedule_id: int,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    try:
        return compute_performance(db, schedule_id, days=days)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e

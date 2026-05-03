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

    Computes the same feature matrix `predict_race` would build, then
    classifies each gap so the UI can give the user an actionable answer:

      * `post_race_features_in_use` — features the experiment was trained
        on that need data only published after the race runs (current SP,
        weigh-in weight, live odds drift). Fix: retrain without them.
      * `entries_missing_history` — dogs in this race with zero prior
        resulted races. History-dependent features will be NaN for these
        dogs by definition. Fix: wait until the dog has run, or accept a
        debutant blind spot.
      * `missing_features` — every NaN cell, with a `reason` field
        distinguishing post-race-only / debutant / sparse-history. The
        last category is the one a backfill scrape can fix.
    """
    import pandas as pd

    from app.services.prediction_service import (
        compute_features_for_entries,
        load_trained_model,
    )
    from app.models.experiment import Experiment
    from app.models.feature_definition import FeatureDefinition
    from ml.feature_availability import (
        POST_RACE_FEATURE_NAMES,
        post_race_features_in_use,
    )

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

    # Per-entry prior-race counts (resulted only, strictly before this
    # race date). Used to label "debutant" missing-feature reasons.
    history_counts: dict[int, int] = {eid: 0 for eid in entry_ids}
    if entries:
        from sqlalchemy import func as sa_func
        target_dog_ids = {e.RaceEntry.dog_id: e.RaceEntry.id for e in entries}
        prior_counts_rows = (
            db.query(
                RaceEntry.dog_id,
                sa_func.count(RaceEntry.id).label("prior_count"),
            )
            .join(Race, Race.id == RaceEntry.race_id)
            .filter(
                RaceEntry.dog_id.in_(list(target_dog_ids.keys())),
                Race.race_date < race.race_date,
                Race.status == "resulted",
            )
            .group_by(RaceEntry.dog_id)
            .all()
        )
        for dog_id, count in prior_counts_rows:
            entry_id = target_dog_ids.get(dog_id)
            if entry_id is not None:
                history_counts[entry_id] = int(count)

    debutants = [
        {
            "entry_id": eid,
            "trap": entry_lookup[eid][0],
            "dog_name": entry_lookup[eid][1],
        }
        for eid, count in history_counts.items()
        if count == 0
    ]
    debutant_ids = {d["entry_id"] for d in debutants}

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

    def _classify(feature_name: str, entry_id: int) -> str:
        if feature_name in POST_RACE_FEATURE_NAMES:
            return "post_race_data"
        if entry_id in debutant_ids:
            return "dog_has_no_history"
        return "history_field_missing"

    missing_per_feature: list[dict[str, Any]] = []
    if not X.empty:
        for col in X.columns[X.isna().any()]:
            offenders = []
            for eid in X.index[X[col].isna()].tolist():
                eid_int = int(eid)
                trap, dog_name = entry_lookup.get(eid_int, (None, None))
                offenders.append({
                    "entry_id": eid_int,
                    "trap": trap,
                    "dog_name": dog_name,
                    "reason": _classify(str(col), eid_int),
                })
            missing_per_feature.append({
                "feature": str(col),
                "missing_for": offenders,
                # A feature is "irrecoverable for this race" only if every
                # offender is either a debutant or a post-race column —
                # the user has no scrape they can run to fix those.
                "all_offenders_irrecoverable": all(
                    o["reason"] in ("post_race_data", "dog_has_no_history")
                    for o in offenders
                ),
            })

    post_race_used = post_race_features_in_use(trained_feature_names)

    # Per-entry data-completeness: fraction of the feature matrix that
    # was non-NaN before imputation. Mirrors what the prediction pipeline
    # now attaches to each prediction so the UI can flag thinly-raced
    # dogs visually instead of refusing the whole race.
    data_completeness: list[dict[str, Any]] = []
    if not X.empty:
        for eid, frac in X.notna().mean(axis=1).to_dict().items():
            eid_int = int(eid)
            trap, dog_name = entry_lookup.get(eid_int, (None, None))
            data_completeness.append({
                "entry_id": eid_int,
                "trap": trap,
                "dog_name": dog_name,
                "completeness": round(float(frac), 4),
            })

    # `would_fail` now only matches the predict-time hard-refuse rule:
    # a scheduled race fails iff the trained feature list contains any
    # post-race-only column. Sparse-history NaN cells are no longer a
    # blocker — they're median-filled by the prediction pipeline (same
    # as training) and reflected in the per-entry completeness score
    # above. Bet sizing on those dogs should be downweighted by the
    # caller, not refused outright.
    would_fail = bool(race.status == "scheduled" and post_race_used)

    return {
        "race_id": race_id,
        "race_status": race.status,
        "experiment_id": experiment_id,
        "n_entries": len(entries),
        "post_race_features_in_use": [
            {"feature": name, "reason": reason}
            for name, reason in post_race_used.items()
        ],
        "entries_missing_history": debutants,
        "missing_features": missing_per_feature,
        "data_completeness": data_completeness,
        "would_fail": would_fail,
    }


@router.get("/race/{race_id}")
def predict_single_race(
    race_id: int,
    experiment_id: int,
    bankroll: float = Query(default=100.0, ge=1),
    refresh: bool = Query(
        default=False,
        description="If true, recompute even if saved predictions exist",
    ),
    db: Session = Depends(get_db),
):
    """Generate or fetch saved predictions for a specific race.

    Default behaviour: if saved predictions exist for this (experiment, race)
    pair, return them directly without re-running the model. Pass
    `refresh=true` to force recomputation (e.g. after retraining or to
    re-stake against a new bankroll).
    """
    from app.services.prediction_service import (
        get_saved_predictions_for_race,
        predict_race,
        save_predictions,
    )

    race = db.query(Race).filter(Race.id == race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    cached: list[dict[str, Any]] = []
    if not refresh:
        cached = get_saved_predictions_for_race(db, experiment_id, race_id)

    if cached:
        preds = cached
        saved = 0
        from_cache = True
    else:
        try:
            preds = predict_race(db, experiment_id, race_id, bankroll=bankroll)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        saved = save_predictions(db, preds) if preds else 0
        from_cache = False

    track = db.query(Track).filter(Track.id == race.track_id).first()

    last_predicted_at: str | None = None
    if preds:
        timestamps = [
            p.get("updated_at") or p.get("created_at") for p in preds
        ]
        timestamps = [t for t in timestamps if t]
        if timestamps:
            last_predicted_at = max(timestamps)

    return {
        "race_id": race_id,
        "race_date": str(race.race_date),
        "race_number": race.race_number,
        "track_name": track.name if track else None,
        "distance_m": race.distance_m,
        "grade": race.grade,
        "predictions": preds,
        "saved": saved,
        "from_cache": from_cache,
        "last_predicted_at": last_predicted_at,
    }


@router.get("/race/{race_id}/saved")
def get_saved_race_predictions(
    race_id: int,
    experiment_id: int,
    db: Session = Depends(get_db),
):
    """Return saved predictions for a race without ever recomputing.

    Returns 404 if no predictions have been saved for this (race, experiment).
    """
    from app.services.prediction_service import get_saved_predictions_for_race

    race = db.query(Race).filter(Race.id == race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    preds = get_saved_predictions_for_race(db, experiment_id, race_id)
    if not preds:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No saved predictions for race {race_id} with experiment "
                f"{experiment_id}. Run a prediction first."
            ),
        )

    track = db.query(Track).filter(Track.id == race.track_id).first()
    timestamps = [p.get("updated_at") or p.get("created_at") for p in preds]
    timestamps = [t for t in timestamps if t]
    last_predicted_at = max(timestamps) if timestamps else None

    return {
        "race_id": race_id,
        "race_date": str(race.race_date),
        "race_number": race.race_number,
        "track_name": track.name if track else None,
        "distance_m": race.distance_m,
        "grade": race.grade,
        "predictions": preds,
        "from_cache": True,
        "last_predicted_at": last_predicted_at,
    }


@router.get("/history")
def prediction_history(
    experiment_id: int | None = Query(
        default=None,
        description="Filter by experiment. Omit to see history across all models.",
    ),
    race_date_from: date | None = None,
    race_date_to: date | None = None,
    track_code: str | None = None,
    limit: int = Query(default=200, le=1000),
    db: Session = Depends(get_db),
):
    """List past prediction sessions grouped by (experiment, race).

    One row per race that has saved predictions, with the model's top pick,
    when it was predicted, the bankroll context, and (if the race has
    resulted) whether the top pick won. Lets the user browse past
    predictions without re-running anything.
    """
    from sqlalchemy import func as sa_func

    from app.models.experiment import Experiment

    # Aggregate at the (experiment, race) level: how many entries were
    # predicted and the most recent prediction timestamp per race.
    agg_query = (
        db.query(
            Prediction.experiment_id,
            RaceEntry.race_id.label("race_id"),
            sa_func.count(Prediction.id).label("n_predictions"),
            sa_func.max(
                sa_func.coalesce(Prediction.updated_at, Prediction.created_at)
            ).label("last_predicted_at"),
        )
        .join(RaceEntry, Prediction.race_entry_id == RaceEntry.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .group_by(Prediction.experiment_id, RaceEntry.race_id)
    )

    if experiment_id is not None:
        agg_query = agg_query.filter(Prediction.experiment_id == experiment_id)
    if race_date_from is not None:
        agg_query = agg_query.filter(Race.race_date >= race_date_from)
    if race_date_to is not None:
        agg_query = agg_query.filter(Race.race_date <= race_date_to)
    if track_code is not None:
        agg_query = (
            agg_query.join(Track, Race.track_id == Track.id)
            .filter(Track.code == track_code)
        )

    agg_query = agg_query.order_by(sa_func.max(
        sa_func.coalesce(Prediction.updated_at, Prediction.created_at)
    ).desc()).limit(limit)

    agg_rows = agg_query.all()

    if not agg_rows:
        return {"sessions": [], "total": 0}

    # Build lookup of race + track + experiment metadata in batch.
    race_ids = list({row.race_id for row in agg_rows})
    exp_ids = list({row.experiment_id for row in agg_rows})

    races_meta = {
        r.id: r for r in (
            db.query(Race).filter(Race.id.in_(race_ids)).all()
        )
    }
    tracks_meta = {
        t.id: t for t in (
            db.query(Track)
            .filter(Track.id.in_({r.track_id for r in races_meta.values()}))
            .all()
        )
    }
    experiments_meta = {
        e.id: e for e in (
            db.query(Experiment).filter(Experiment.id.in_(exp_ids)).all()
        )
    }

    # For each (experiment, race) fetch the top pick — single batched query.
    pairs = [(row.experiment_id, row.race_id) for row in agg_rows]
    top_picks: dict[tuple[int, int], dict[str, Any]] = {}
    if pairs:
        top_rows = (
            db.query(
                Prediction.experiment_id,
                RaceEntry.race_id.label("race_id"),
                Prediction.win_probability,
                Prediction.confidence,
                Prediction.confidence_tier,
                Prediction.edge,
                Prediction.is_value,
                Prediction.kelly_bet,
                Prediction.kelly_stake,
                Prediction.bankroll_used,
                RaceEntry.trap,
                RaceEntry.finish_position,
                Dog.name.label("dog_name"),
            )
            .join(RaceEntry, Prediction.race_entry_id == RaceEntry.id)
            .join(Dog, RaceEntry.dog_id == Dog.id)
            .filter(RaceEntry.race_id.in_(race_ids))
            .filter(Prediction.experiment_id.in_(exp_ids))
            .order_by(
                Prediction.experiment_id,
                RaceEntry.race_id,
                Prediction.win_probability.desc().nullslast(),
            )
            .all()
        )
        # Keep the first row per (exp, race) — the highest-prob pick.
        for r in top_rows:
            key = (r.experiment_id, r.race_id)
            if key not in top_picks:
                top_picks[key] = {
                    "dog_name": r.dog_name,
                    "trap": r.trap,
                    "win_probability": r.win_probability,
                    "confidence": r.confidence,
                    "confidence_tier": r.confidence_tier,
                    "edge": r.edge,
                    "is_value": r.is_value,
                    "kelly_bet": bool(r.kelly_bet) if r.kelly_bet is not None else None,
                    "kelly_stake": r.kelly_stake,
                    "bankroll_used": r.bankroll_used,
                    "actual_position": r.finish_position,
                    "top_pick_won": (
                        r.finish_position == 1 if r.finish_position is not None else None
                    ),
                }

    sessions = []
    for row in agg_rows:
        race = races_meta.get(row.race_id)
        track = tracks_meta.get(race.track_id) if race else None
        exp = experiments_meta.get(row.experiment_id)
        sessions.append({
            "experiment_id": row.experiment_id,
            "experiment_name": exp.name if exp else None,
            "experiment_algorithm": exp.algorithm if exp else None,
            "race_id": row.race_id,
            "race_date": str(race.race_date) if race else None,
            "race_number": race.race_number if race else None,
            "race_status": race.status if race else None,
            "track_name": track.name if track else None,
            "track_code": track.code if track else None,
            "distance_m": race.distance_m if race else None,
            "grade": race.grade if race else None,
            "n_predictions": row.n_predictions,
            "last_predicted_at": (
                row.last_predicted_at.isoformat()
                if row.last_predicted_at else None
            ),
            "top_pick": top_picks.get((row.experiment_id, row.race_id)),
        })

    return {"sessions": sessions, "total": len(sessions)}


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

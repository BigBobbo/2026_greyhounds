"""Feature definition CRUD + preview + materialization endpoints."""

from threading import Thread
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.feature_definition import FeatureDefinition
from app.schemas.feature import (
    FeatureDefinitionCreate,
    FeatureDefinitionResponse,
    FeatureDefinitionUpdate,
    FeaturePreviewRequest,
    FeaturePreviewResponse,
)
from app.services.feature_engine import get_dog_history, compute_visual_feature
from app.services.feature_sandbox import execute_feature_code, validate_feature_code

router = APIRouter(prefix="/features", tags=["features"])


class MaterializeRequest(BaseModel):
    feature_ids: list[int] | None = None  # None = all enabled
    force: bool = False


class MaterializeResponse(BaseModel):
    message: str
    results: dict[str, Any] | None = None


class FeatureCoverageItem(BaseModel):
    feature_id: int
    name: str
    display_name: str | None
    feature_type: str
    enabled: bool
    computed_count: int
    total_entries: int
    coverage_pct: float


@router.get("/", response_model=list[FeatureDefinitionResponse])
def list_features(enabled_only: bool = False, db: Session = Depends(get_db)):
    query = db.query(FeatureDefinition)
    if enabled_only:
        query = query.filter(FeatureDefinition.enabled.is_(True))
    return query.order_by(FeatureDefinition.name).all()


@router.get("/coverage", response_model=list[FeatureCoverageItem])
def get_coverage(db: Session = Depends(get_db)):
    """Get computation coverage stats for all features."""
    from ml.feature_store import get_feature_coverage
    return get_feature_coverage(db)


@router.get("/start-materialize")
def start_materialize_get(force: bool = False, db: Session = Depends(get_db)):
    """GET endpoint to trigger materialization from browser URL bar."""
    req = MaterializeRequest(force=force)
    return trigger_materialization(req, db)


@router.get("/{feature_id}", response_model=FeatureDefinitionResponse)
def get_feature(feature_id: int, db: Session = Depends(get_db)):
    feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    return feature


@router.post("/", response_model=FeatureDefinitionResponse, status_code=201)
def create_feature(feature: FeatureDefinitionCreate, db: Session = Depends(get_db)):
    # Validate code features
    if feature.feature_type == "code" and feature.code:
        error = validate_feature_code(feature.code)
        if error:
            raise HTTPException(status_code=400, detail=f"Invalid code: {error}")

    db_feature = FeatureDefinition(**feature.model_dump())
    db.add(db_feature)
    db.commit()
    db.refresh(db_feature)
    return db_feature


@router.patch("/{feature_id}", response_model=FeatureDefinitionResponse)
def update_feature(feature_id: int, update: FeatureDefinitionUpdate, db: Session = Depends(get_db)):
    db_feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not db_feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    for key, value in update.model_dump(exclude_unset=True).items():
        setattr(db_feature, key, value)
    db.commit()
    db.refresh(db_feature)
    return db_feature


@router.delete("/{feature_id}", status_code=204)
def delete_feature(feature_id: int, db: Session = Depends(get_db)):
    db_feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not db_feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    db.delete(db_feature)
    db.commit()


@router.post("/preview", response_model=FeaturePreviewResponse)
def preview_feature(req: FeaturePreviewRequest, db: Session = Depends(get_db)):
    """
    Preview a feature value for a specific dog without saving.
    Uses the dog's most recent race as the race context.
    """
    from app.models.race import Race
    from app.models.race_entry import RaceEntry

    # Get the dog's most recent race entry for context
    latest_entry = (
        db.query(RaceEntry)
        .join(Race)
        .filter(RaceEntry.dog_id == req.dog_id, Race.status == "resulted")
        .order_by(Race.race_date.desc())
        .first()
    )

    if not latest_entry:
        return FeaturePreviewResponse(error="No race history found for this dog")

    from app.services.feature_engine import get_race_context
    ctx = get_race_context(db, latest_entry.id)
    if not ctx:
        return FeaturePreviewResponse(error="Could not build race context")

    history = get_dog_history(db, req.dog_id, ctx["race_date"])
    if history.empty:
        return FeaturePreviewResponse(error="No prior race history for this dog")

    if req.feature_type == "visual":
        config = req.config_json or {}
        value = compute_visual_feature(history, config, ctx)
        return FeaturePreviewResponse(value=value)

    elif req.feature_type == "code":
        if not req.code:
            return FeaturePreviewResponse(error="No code provided")

        error = validate_feature_code(req.code)
        if error:
            return FeaturePreviewResponse(error=error)

        value, error = execute_feature_code(req.code, history, ctx)
        return FeaturePreviewResponse(value=value, error=error)

    return FeaturePreviewResponse(error=f"Unknown feature_type: {req.feature_type}")


@router.post("/materialize", response_model=MaterializeResponse)
def trigger_materialization(req: MaterializeRequest, db: Session = Depends(get_db)):
    """Trigger feature materialization in the background."""
    from ml.feature_store import materialize_feature, materialize_all_features

    if req.feature_ids:
        features = db.query(FeatureDefinition).filter(FeatureDefinition.id.in_(req.feature_ids)).all()
        if not features:
            raise HTTPException(status_code=404, detail="No features found")
    else:
        features = db.query(FeatureDefinition).filter(FeatureDefinition.enabled.is_(True)).all()

    if not features:
        return MaterializeResponse(message="No enabled features to materialize")

    def _run():
        db2 = SessionLocal()
        try:
            for f in features:
                # Re-fetch in new session
                feat = db2.query(FeatureDefinition).filter(FeatureDefinition.id == f.id).first()
                if feat:
                    materialize_feature(db2, feat, force=req.force)
        finally:
            db2.close()

    thread = Thread(target=_run, daemon=True)
    thread.start()

    return MaterializeResponse(
        message=f"Materialization started for {len(features)} features in background",
    )


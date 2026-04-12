"""Feature definition CRUD + preview + materialization endpoints."""

import logging
import os
import glob as globmod
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/features", tags=["features"])


class MaterializeRequest(BaseModel):
    feature_ids: list[int] | None = None  # None = all enabled
    force: bool = False
    # When set, computed features are saved under a named version snapshot.
    # Create the version first via POST /features/versions, then pass its id.
    version_id: int | None = None


class MaterializeResponse(BaseModel):
    message: str
    results: dict[str, Any] | None = None


class CreateVersionRequest(BaseModel):
    name: str
    description: str | None = None


class VersionResponse(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: Any
    coverage_snapshot: Any | None
    feature_count: int = 0
    model_config = {"from_attributes": True}


class FeatureCoverageItem(BaseModel):
    feature_id: int
    name: str
    display_name: str | None
    feature_type: str
    enabled: bool
    computed_count: int
    incomplete_count: int = 0
    total_entries: int
    coverage_pct: float


class DataIntegrityRequest(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    max_gap_days: int = 14


@router.get("/", response_model=list[FeatureDefinitionResponse])
def list_features(enabled_only: bool = False, db: Session = Depends(get_db)):
    query = db.query(FeatureDefinition)
    if enabled_only:
        query = query.filter(FeatureDefinition.enabled.is_(True))
    return query.order_by(FeatureDefinition.name).all()


@router.get("/coverage", response_model=list[FeatureCoverageItem])
def get_coverage(version_id: int | None = None, db: Session = Depends(get_db)):
    """Get computation coverage stats for all features, optionally filtered by version."""
    from ml.feature_store import get_feature_coverage
    return get_feature_coverage(db, version_id=version_id)


@router.get("/data-integrity")
def get_data_integrity(
    start_date: str | None = None,
    end_date: str | None = None,
    max_gap_days: int = 14,
    db: Session = Depends(get_db),
):
    """
    Check data completeness before materializing features.

    Reports scrape coverage gaps that could cause features to be computed
    with incomplete dog histories — e.g. if a dog raced at Dublin and
    Limerick but only Limerick data has been scraped, rolling features
    like 'mean last 5 race times' would silently be wrong.

    Returns a recommendation of "safe", "warning", or "incomplete".
    """
    from datetime import date as date_type
    from ml.data_integrity import assess_materialization_readiness

    sd = date_type.fromisoformat(start_date) if start_date else None
    ed = date_type.fromisoformat(end_date) if end_date else None

    return assess_materialization_readiness(db, sd, ed, max_gap_days)


@router.post("/data-integrity")
def post_data_integrity(req: DataIntegrityRequest, db: Session = Depends(get_db)):
    """POST variant of data-integrity check."""
    from datetime import date as date_type
    from ml.data_integrity import assess_materialization_readiness

    sd = date_type.fromisoformat(req.start_date) if req.start_date else None
    ed = date_type.fromisoformat(req.end_date) if req.end_date else None

    return assess_materialization_readiness(db, sd, ed, req.max_gap_days)


@router.get("/versions")
def list_versions(db: Session = Depends(get_db)):
    """List all feature versions, newest first."""
    from sqlalchemy import func as sqlfunc
    from app.models.feature_version import FeatureVersion
    from app.models.computed_feature import ComputedFeature

    count_sub = (
        db.query(
            ComputedFeature.version_id,
            sqlfunc.count(ComputedFeature.id).label("feature_count"),
        )
        .group_by(ComputedFeature.version_id)
        .subquery()
    )

    rows = (
        db.query(FeatureVersion, sqlfunc.coalesce(count_sub.c.feature_count, 0))
        .outerjoin(count_sub, FeatureVersion.id == count_sub.c.version_id)
        .order_by(FeatureVersion.created_at.desc())
        .all()
    )

    return [
        {
            "id": v.id,
            "name": v.name,
            "description": v.description,
            "created_at": v.created_at,
            "coverage_snapshot": v.coverage_snapshot,
            "feature_count": count,
        }
        for v, count in rows
    ]


@router.post("/versions", status_code=201)
def create_version(req: CreateVersionRequest, db: Session = Depends(get_db)):
    """
    Create a named feature version (snapshot).

    This captures a data-integrity snapshot at creation time so you can
    see the scrape coverage that was in effect when features were computed.
    After creating, pass the returned id to POST /features/materialize
    to compute features into this version.
    """
    from app.models.feature_version import FeatureVersion
    from ml.data_integrity import assess_materialization_readiness

    existing = db.query(FeatureVersion).filter(FeatureVersion.name == req.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Version '{req.name}' already exists")

    snapshot = assess_materialization_readiness(db)

    version = FeatureVersion(
        name=req.name,
        description=req.description,
        coverage_snapshot=snapshot,
    )
    db.add(version)
    db.commit()
    db.refresh(version)

    return {
        "id": version.id,
        "name": version.name,
        "description": version.description,
        "created_at": version.created_at,
        "coverage_snapshot": version.coverage_snapshot,
        "feature_count": 0,
    }


@router.get("/versions/{version_id}")
def get_version(version_id: int, db: Session = Depends(get_db)):
    """Get details for a specific feature version."""
    from sqlalchemy import func as sqlfunc
    from app.models.feature_version import FeatureVersion
    from app.models.computed_feature import ComputedFeature

    version = db.query(FeatureVersion).filter(FeatureVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    try:
        count = (
            db.query(sqlfunc.count(ComputedFeature.id))
            .filter(ComputedFeature.version_id == version.id)
            .scalar() or 0
        )
        incomplete = (
            db.query(sqlfunc.count(ComputedFeature.id))
            .filter(
                ComputedFeature.version_id == version.id,
                ComputedFeature.data_complete.is_(False),
            )
            .scalar() or 0
        )
    except Exception:
        db.rollback()
        count = 0
        incomplete = 0

    return {
        "id": version.id,
        "name": version.name,
        "description": version.description,
        "created_at": version.created_at,
        "coverage_snapshot": version.coverage_snapshot,
        "feature_count": count,
        "incomplete_count": incomplete,
    }


@router.delete("/versions/{version_id}", status_code=204)
def delete_version(version_id: int, db: Session = Depends(get_db)):
    """Delete a feature version and all its computed features."""
    from app.models.feature_version import FeatureVersion
    from app.models.computed_feature import ComputedFeature

    version = db.query(FeatureVersion).filter(FeatureVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    db.query(ComputedFeature).filter(ComputedFeature.version_id == version.id).delete()
    db.delete(version)
    db.commit()


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
    """
    Trigger feature materialization in the background.

    If version_id is provided, features are saved into that named snapshot.
    Otherwise they are saved unversioned (upserted in place).
    """
    from ml.feature_store import materialize_feature

    if req.version_id is not None:
        from app.models.feature_version import FeatureVersion
        version = db.query(FeatureVersion).filter(FeatureVersion.id == req.version_id).first()
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")

    if req.feature_ids:
        features = db.query(FeatureDefinition).filter(FeatureDefinition.id.in_(req.feature_ids)).all()
        if not features:
            raise HTTPException(status_code=404, detail="No features found")
    else:
        features = db.query(FeatureDefinition).filter(FeatureDefinition.enabled.is_(True)).all()

    if not features:
        return MaterializeResponse(message="No enabled features to materialize")

    version_id = req.version_id

    def _run():
        db2 = SessionLocal()
        try:
            for f in features:
                feat = db2.query(FeatureDefinition).filter(FeatureDefinition.id == f.id).first()
                if feat:
                    try:
                        materialize_feature(
                            db2, feat, force=req.force, version_id=version_id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to materialize feature '%s' (id=%d), "
                            "continuing with remaining features",
                            f.name, f.id,
                        )
                        db2.rollback()
        finally:
            db2.close()

    thread = Thread(target=_run, daemon=True)
    thread.start()

    version_msg = f" into version {version_id}" if version_id else ""
    return MaterializeResponse(
        message=f"Materialization started for {len(features)} features{version_msg} in background",
    )


@router.delete("/computed/all", status_code=200)
def clear_all_computed_features():
    """
    Delete ALL computed features (across all versions) and all feature versions.
    Also removes model artifacts from disk.

    Strategy:
    1. Delete model .joblib files to free disk space
    2. Dispose SQLAlchemy engine, then checkpoint + shrink the WAL
    3. DROP + recreate the feature tables
    4. VACUUM to reclaim freed pages

    This does NOT delete:
    - Scraped data (tracks, dogs, races, race_entries)
    - Feature definitions (so they can be re-materialized)
    - Experiment metadata (but model files are removed from disk)
    """
    import sqlite3
    from app.config import settings

    # Parse DB path from SQLAlchemy URL (sqlite:///./data/greyhound.db)
    db_url = settings.database_url
    if db_url.startswith("sqlite:///"):
        db_path = db_url[len("sqlite:///"):]
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported DB URL: {db_url}")

    # ---- PHASE 1: Free disk space by deleting model files ----

    models_deleted = 0
    model_dir = settings.model_artifacts_dir
    if os.path.isdir(model_dir):
        for f in globmod.glob(os.path.join(model_dir, "*.joblib")):
            try:
                os.remove(f)
                models_deleted += 1
            except OSError as e:
                logger.warning("Failed to delete model artifact %s: %s", f, e)

    logger.info("Phase 1 done: freed %d model artifacts", models_deleted)

    # ---- PHASE 2: Dispose engine and checkpoint WAL properly ----

    from app.database import engine
    engine.dispose()

    wal_size = 0
    wal_path = db_path + "-wal"
    if os.path.isfile(wal_path):
        wal_size = os.path.getsize(wal_path)

    try:
        conn = sqlite3.connect(db_path)

        # Try to checkpoint the WAL into the main DB first (preserves data)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.info("WAL checkpoint succeeded")
        except Exception as e:
            logger.warning("WAL checkpoint failed: %s", e)

        # Check if DB is healthy
        try:
            result = conn.execute("PRAGMA quick_check").fetchone()
            db_ok = result and result[0] == "ok"
        except Exception:
            db_ok = False

        if not db_ok:
            logger.warning("Database integrity check failed, attempting recovery")
            conn.close()
            # Try to recover by removing WAL/SHM and reopening
            for path in (db_path + "-wal", db_path + "-shm"):
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            conn = sqlite3.connect(db_path)
            # Re-check
            try:
                result = conn.execute("PRAGMA quick_check").fetchone()
                db_ok = result and result[0] == "ok"
            except Exception:
                db_ok = False

            if not db_ok:
                logger.warning(
                    "DB still not clean after WAL removal — proceeding anyway "
                    "since we're dropping the affected tables"
                )

        conn.execute("PRAGMA foreign_keys=OFF")
        cursor = conn.cursor()

        # Count before dropping (may fail if tables are corrupted)
        try:
            computed_count = cursor.execute(
                "SELECT COUNT(*) FROM computed_features"
            ).fetchone()[0]
        except Exception:
            computed_count = -1  # unknown
        try:
            version_count = cursor.execute(
                "SELECT COUNT(*) FROM feature_versions"
            ).fetchone()[0]
        except Exception:
            version_count = -1

        # Drop and recreate computed_features
        cursor.execute("DROP TABLE IF EXISTS computed_features")
        cursor.execute("""
            CREATE TABLE computed_features (
                id INTEGER PRIMARY KEY,
                race_entry_id INTEGER NOT NULL REFERENCES race_entries(id),
                feature_def_id INTEGER NOT NULL REFERENCES feature_definitions(id),
                value FLOAT,
                computed_at DATETIME,
                data_complete BOOLEAN DEFAULT 1,
                version_id INTEGER REFERENCES feature_versions(id),
                CONSTRAINT uq_computed_entry_feature_version
                    UNIQUE (race_entry_id, feature_def_id, version_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX ix_computed_features_race_entry_id "
            "ON computed_features (race_entry_id)"
        )
        cursor.execute(
            "CREATE INDEX ix_computed_features_version_id "
            "ON computed_features (version_id)"
        )
        conn.commit()

        # Drop and recreate feature_versions
        cursor.execute("DROP TABLE IF EXISTS feature_versions")
        cursor.execute("""
            CREATE TABLE feature_versions (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL UNIQUE,
                description TEXT,
                created_at DATETIME,
                coverage_snapshot JSON
            )
        """)
        conn.commit()

        # VACUUM to reclaim freed pages
        vacuumed = False
        try:
            conn.execute("VACUUM")
            vacuumed = True
        except Exception as e:
            logger.warning("VACUUM failed: %s", e)

        # Restore WAL mode for normal operation
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        # Final integrity check
        try:
            result = conn.execute("PRAGMA quick_check").fetchone()
            final_ok = result and result[0] == "ok"
        except Exception:
            final_ok = False

        cursor.close()
        conn.close()

    except sqlite3.Error as e:
        logger.error("Cleanup phase 2 failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed at DB phase: {e}. "
            f"Freed {models_deleted} model artifacts. "
            f"WAL was {wal_size // (1024*1024)} MB.",
        )
    except Exception as e:
        logger.error("Cleanup failed unexpectedly: %s", e)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")

    return {
        "computed_features_deleted": computed_count,
        "feature_versions_deleted": version_count,
        "model_artifacts_deleted": models_deleted,
        "wal_file_mb": wal_size // (1024 * 1024),
        "vacuum_completed": vacuumed,
        "database_healthy": final_ok,
        "message": (
            f"Cleared {computed_count:,} computed features across "
            f"{version_count} versions. "
            f"Removed {models_deleted} model artifacts. "
            f"WAL was {wal_size // (1024*1024)} MB. "
            f"DB healthy: {final_ok}. "
            f"{'Disk space reclaimed via VACUUM.' if vacuumed else 'VACUUM skipped.'}"
        ),
    }


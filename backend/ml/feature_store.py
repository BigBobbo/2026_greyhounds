"""
Feature store: batch materialization of features into the computed_features table.

Handles both visual (JSON config) and code (Python) features.
Supports incremental computation — only computes for entries that don't already have values.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.models.computed_feature import ComputedFeature
from app.models.feature_definition import FeatureDefinition
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.services.feature_engine import compute_visual_feature, get_dog_history, get_race_context
from app.services.feature_sandbox import execute_feature_code

logger = logging.getLogger(__name__)


def materialize_feature(
    db: Session,
    feature_def: FeatureDefinition,
    race_entry_ids: list[int] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """
    Materialize a single feature for specified race entries (or all resulted entries).

    Returns {"computed": N, "skipped": N, "errors": N}
    """
    stats = {"computed": 0, "skipped": 0, "errors": 0}

    # Get entries to compute for
    if race_entry_ids:
        entries_query = db.query(RaceEntry.id).filter(RaceEntry.id.in_(race_entry_ids))
    else:
        entries_query = (
            db.query(RaceEntry.id)
            .join(Race)
            .filter(Race.status == "resulted")
        )

    if not force:
        # Exclude entries that already have this feature computed
        already_computed = (
            db.query(ComputedFeature.race_entry_id)
            .filter(ComputedFeature.feature_def_id == feature_def.id)
        )
        entries_query = entries_query.filter(~RaceEntry.id.in_(already_computed))

    entry_ids = [row[0] for row in entries_query.all()]

    if not entry_ids:
        logger.info("No entries to compute for feature %s", feature_def.name)
        return stats

    logger.info(
        "Materializing feature '%s' for %d entries",
        feature_def.name, len(entry_ids),
    )

    # Process in batches
    batch_size = 100
    for i in range(0, len(entry_ids), batch_size):
        batch = entry_ids[i:i + batch_size]

        for entry_id in batch:
            ctx = get_race_context(db, entry_id)
            if not ctx:
                stats["errors"] += 1
                continue

            dog_id = ctx["dog_id"]
            race_date = ctx["race_date"]

            # Get dog history (before this race)
            history = get_dog_history(db, dog_id, race_date)

            # Compute the feature
            value = None
            error = None

            if feature_def.feature_type == "visual":
                config = feature_def.config_json or {}
                value = compute_visual_feature(history, config, ctx)
            elif feature_def.feature_type == "code":
                code = feature_def.code or ""
                value, error = execute_feature_code(code, history, ctx)
                if error:
                    logger.debug("Error computing '%s' for entry %d: %s", feature_def.name, entry_id, error)
                    stats["errors"] += 1
                    continue

            # Upsert computed value
            existing = (
                db.query(ComputedFeature)
                .filter(
                    ComputedFeature.race_entry_id == entry_id,
                    ComputedFeature.feature_def_id == feature_def.id,
                )
                .first()
            )

            if existing:
                existing.value = value
                existing.computed_at = datetime.utcnow()
            else:
                db.add(ComputedFeature(
                    race_entry_id=entry_id,
                    feature_def_id=feature_def.id,
                    value=value,
                    computed_at=datetime.utcnow(),
                ))

            stats["computed"] += 1

        db.commit()

        if (i + batch_size) % 500 == 0:
            logger.info(
                "Feature '%s': %d/%d entries computed",
                feature_def.name, min(i + batch_size, len(entry_ids)), len(entry_ids),
            )

    logger.info(
        "Feature '%s' done: %d computed, %d skipped, %d errors",
        feature_def.name, stats["computed"], stats["skipped"], stats["errors"],
    )
    return stats


def materialize_all_features(
    db: Session,
    race_entry_ids: list[int] | None = None,
    force: bool = False,
) -> dict[str, dict[str, int]]:
    """Materialize all enabled features."""
    features = db.query(FeatureDefinition).filter(FeatureDefinition.enabled.is_(True)).all()
    results = {}

    for feature_def in features:
        stats = materialize_feature(db, feature_def, race_entry_ids, force)
        results[feature_def.name] = stats

    return results


def build_feature_matrix(
    db: Session,
    feature_ids: list[int],
    race_entry_ids: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix (DataFrame) from computed features.

    Returns a DataFrame with race_entry_id as index and feature names as columns.
    """
    features = db.query(FeatureDefinition).filter(FeatureDefinition.id.in_(feature_ids)).all()
    feature_map = {f.id: f.name for f in features}

    query = (
        db.query(
            ComputedFeature.race_entry_id,
            ComputedFeature.feature_def_id,
            ComputedFeature.value,
        )
        .filter(ComputedFeature.feature_def_id.in_(feature_ids))
    )

    if race_entry_ids:
        query = query.filter(ComputedFeature.race_entry_id.in_(race_entry_ids))

    rows = query.all()

    if not rows:
        return pd.DataFrame()

    data: dict[int, dict[str, float | None]] = {}
    for entry_id, feat_id, value in rows:
        if entry_id not in data:
            data[entry_id] = {}
        data[entry_id][feature_map[feat_id]] = value

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "race_entry_id"
    return df


def get_feature_coverage(db: Session) -> list[dict[str, Any]]:
    """Get feature coverage stats (how many entries have each feature computed)."""
    features = db.query(FeatureDefinition).all()
    total_entries = db.query(func.count(RaceEntry.id)).scalar() or 0

    result = []
    for f in features:
        computed = (
            db.query(func.count(ComputedFeature.id))
            .filter(ComputedFeature.feature_def_id == f.id)
            .scalar() or 0
        )
        result.append({
            "feature_id": f.id,
            "name": f.name,
            "display_name": f.display_name,
            "feature_type": f.feature_type,
            "enabled": f.enabled,
            "computed_count": computed,
            "total_entries": total_entries,
            "coverage_pct": round(computed / total_entries * 100, 1) if total_entries > 0 else 0,
        })

    return result

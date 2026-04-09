"""
Feature store: batch materialization of features into the computed_features table.

Handles both visual (JSON config) and code (Python) features.
Supports incremental computation — only computes for entries that don't already have values.

Data completeness:
Each computed feature is tagged with `data_complete`.  When the dog's race
history may be incomplete (e.g. some tracks not yet scraped for the feature's
date window), the flag is set to False so downstream consumers can filter
these out during training.
"""

import logging
from datetime import datetime, timedelta
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
from ml.data_integrity import check_dog_history_complete

logger = logging.getLogger(__name__)

# Default lookback window (days) used when checking whether a dog's history
# is complete.  90 days covers most "last N" features comfortably.
_DEFAULT_COMPLETENESS_WINDOW_DAYS = 90


def _feature_window_days(feature_def: FeatureDefinition) -> int:
    """Estimate the lookback window (in days) a feature needs.

    For "last_n" features we can't know the exact date span, so we use a
    generous default.  For "days" features we use the configured value.
    """
    if feature_def.feature_type == "visual":
        config = feature_def.config_json or {}
        window = config.get("window", {})
        if window.get("type") == "days":
            return int(window.get("n", _DEFAULT_COMPLETENESS_WINDOW_DAYS))
    return _DEFAULT_COMPLETENESS_WINDOW_DAYS


def materialize_feature(
    db: Session,
    feature_def: FeatureDefinition,
    race_entry_ids: list[int] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """
    Materialize a single feature for specified race entries (or all resulted entries).

    Returns {"computed": N, "skipped": N, "errors": N, "incomplete": N}

    Each computed feature is tagged with ``data_complete`` — False when the
    dog's history may be incomplete because some tracks have scrape gaps in
    the feature's lookback window.
    """
    stats = {"computed": 0, "skipped": 0, "errors": 0, "incomplete": 0}

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

    window_days = _feature_window_days(feature_def)

    # Cache completeness checks per (dog_id, race_date) to avoid redundant
    # DB queries — many entries share the same dog.
    completeness_cache: dict[tuple[int, str], bool] = {}

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

            # Check data completeness (cached per dog+date)
            cache_key = (dog_id, str(race_date))
            if cache_key not in completeness_cache:
                completeness_cache[cache_key] = check_dog_history_complete(
                    db, dog_id, race_date, window_days=window_days,
                )
            data_complete = completeness_cache[cache_key]

            if not data_complete:
                stats["incomplete"] += 1

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
                existing.data_complete = data_complete
            else:
                db.add(ComputedFeature(
                    race_entry_id=entry_id,
                    feature_def_id=feature_def.id,
                    value=value,
                    computed_at=datetime.utcnow(),
                    data_complete=data_complete,
                ))

            stats["computed"] += 1

        db.commit()

        if (i + batch_size) % 500 == 0:
            logger.info(
                "Feature '%s': %d/%d entries computed (%d incomplete)",
                feature_def.name, min(i + batch_size, len(entry_ids)),
                len(entry_ids), stats["incomplete"],
            )

    if stats["incomplete"]:
        logger.warning(
            "Feature '%s': %d/%d entries flagged as INCOMPLETE (dog history "
            "may be missing races from tracks with scrape gaps)",
            feature_def.name, stats["incomplete"], stats["computed"],
        )

    logger.info(
        "Feature '%s' done: %d computed, %d skipped, %d errors, %d incomplete",
        feature_def.name, stats["computed"], stats["skipped"],
        stats["errors"], stats["incomplete"],
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
    only_complete: bool = False,
) -> pd.DataFrame:
    """
    Build a feature matrix (DataFrame) from computed features.
    Memory-efficient: queries directly into pandas via raw SQL.

    Args:
        only_complete: If True, exclude computed features flagged as
            data_complete=False.  This drops rows where *any* feature
            was computed with incomplete dog history.

    Returns a DataFrame with race_entry_id as index and feature names as columns.
    """
    features = db.query(FeatureDefinition).filter(FeatureDefinition.id.in_(feature_ids)).all()
    feature_map = {f.id: f.name for f in features}

    if not feature_map:
        return pd.DataFrame()

    complete_filter = "AND data_complete = 1" if only_complete else ""

    # Use raw SQL with pandas for memory efficiency — avoids building
    # millions of Python objects via SQLAlchemy ORM
    from sqlalchemy import text
    import sqlite3

    # Get the raw connection
    connection = db.get_bind().raw_connection()

    if race_entry_ids:
        # Build the pivot query in batches and concatenate
        all_dfs = []
        batch_size = 400

        for i in range(0, len(race_entry_ids), batch_size):
            batch = race_entry_ids[i:i + batch_size]
            placeholders_entries = ",".join(str(int(eid)) for eid in batch)
            placeholders_features = ",".join(str(int(fid)) for fid in feature_ids)

            sql = f"""
                SELECT race_entry_id, feature_def_id, value
                FROM computed_features
                WHERE feature_def_id IN ({placeholders_features})
                AND race_entry_id IN ({placeholders_entries})
                {complete_filter}
            """
            batch_df = pd.read_sql_query(sql, connection)
            if not batch_df.empty:
                all_dfs.append(batch_df)

        if not all_dfs:
            return pd.DataFrame()

        long_df = pd.concat(all_dfs, ignore_index=True)
    else:
        placeholders_features = ",".join(str(int(fid)) for fid in feature_ids)
        sql = f"""
            SELECT race_entry_id, feature_def_id, value
            FROM computed_features
            WHERE feature_def_id IN ({placeholders_features})
            {complete_filter}
        """
        long_df = pd.read_sql_query(sql, connection)

    if long_df.empty:
        return pd.DataFrame()

    # Map feature IDs to names
    long_df["feature_name"] = long_df["feature_def_id"].map(feature_map)

    # Pivot from long to wide format
    df = long_df.pivot(index="race_entry_id", columns="feature_name", values="value")
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
        incomplete = (
            db.query(func.count(ComputedFeature.id))
            .filter(
                ComputedFeature.feature_def_id == f.id,
                ComputedFeature.data_complete.is_(False),
            )
            .scalar() or 0
        )
        result.append({
            "feature_id": f.id,
            "name": f.name,
            "display_name": f.display_name,
            "feature_type": f.feature_type,
            "enabled": f.enabled,
            "computed_count": computed,
            "incomplete_count": incomplete,
            "total_entries": total_entries,
            "coverage_pct": round(computed / total_entries * 100, 1) if total_entries > 0 else 0,
        })

    return result

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
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import and_, case, func, text
from sqlalchemy.orm import Session

from app.models.computed_feature import ComputedFeature
from app.models.feature_definition import FeatureDefinition
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track
from app.services.feature_engine import compute_visual_feature
from app.services.feature_sandbox import execute_feature_code
from ml.data_integrity import find_coverage_gaps

logger = logging.getLogger(__name__)

# Default lookback window (days) used when checking whether a dog's history
# is complete.  90 days covers most "last N" features comfortably.
_DEFAULT_COMPLETENESS_WINDOW_DAYS = 90


def _wal_checkpoint(db: Session) -> None:
    """Run a passive WAL checkpoint to reclaim disk space.

    PASSIVE mode checkpoints as much as possible without blocking concurrent
    readers/writers, which is safe to run between batches.
    """
    try:
        raw_conn = db.get_bind().raw_connection()
        raw_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    except Exception as e:
        logger.debug("WAL checkpoint skipped: %s", e)


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


def _batch_load_contexts(db: Session, entry_ids: list[int]) -> dict[int, dict]:
    """Load race contexts for many entries in one query.

    Returns {entry_id: {trap, dog_id, track_id, distance_m, ...}}
    """
    from app.models.track import Track

    rows = (
        db.query(
            RaceEntry.id,
            RaceEntry.trap,
            RaceEntry.dog_id,
            RaceEntry.sp_decimal,
            Race.track_id,
            Race.distance_m,
            Race.grade,
            Race.race_date,
            Race.race_type,
            Track.code.label("track_code"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Track, Race.track_id == Track.id)
        .filter(RaceEntry.id.in_(entry_ids))
        .all()
    )
    return {
        row.id: {
            "trap": row.trap,
            "dog_id": row.dog_id,
            "sp_decimal": row.sp_decimal,
            "track_id": row.track_id,
            "distance_m": row.distance_m,
            "grade": row.grade,
            "race_date": row.race_date,
            "race_type": row.race_type,
            "track_code": row.track_code,
        }
        for row in rows
    }


def _batch_load_dog_histories(db: Session, dog_ids: set[int]) -> dict[int, pd.DataFrame]:
    """Load full race history for multiple dogs in one query.

    Returns {dog_id: DataFrame} where each DataFrame contains the dog's
    complete history sorted chronologically.  The caller filters by date.
    """
    from app.models.track import Track

    rows = (
        db.query(
            RaceEntry.dog_id,
            RaceEntry.trap,
            RaceEntry.finish_position,
            RaceEntry.finish_time,
            RaceEntry.sectional_time,
            RaceEntry.beaten_distance,
            RaceEntry.weight_kg,
            RaceEntry.sp_decimal,
            RaceEntry.starting_price,
            RaceEntry.comment,
            Race.race_date,
            Race.track_id,
            Race.distance_m,
            Race.grade,
            Race.race_type,
            Race.going,
            Race.num_runners,
            Track.name.label("track_name"),
            Track.code.label("track_code"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Track, Race.track_id == Track.id)
        .filter(
            RaceEntry.dog_id.in_(dog_ids),
            Race.status == "resulted",
        )
        .order_by(RaceEntry.dog_id, Race.race_date)
        .all()
    )

    if not rows:
        return {}

    columns = [
        "dog_id", "trap", "finish_position", "finish_time", "sectional_time",
        "beaten_distance", "weight_kg", "sp_decimal", "starting_price",
        "comment", "race_date", "track_id", "distance_m", "grade",
        "race_type", "going", "num_runners", "track_name", "track_code",
    ]
    all_df = pd.DataFrame(rows, columns=columns)

    result: dict[int, pd.DataFrame] = {}
    for dog_id, group in all_df.groupby("dog_id"):
        result[int(dog_id)] = group.drop(columns=["dog_id"]).reset_index(drop=True)

    return result


def _precompute_completeness(
    db: Session,
    dog_histories: dict[int, pd.DataFrame],
    gap_track_ids: set[int],
) -> dict[int, bool]:
    """Pre-compute data completeness for each dog.

    A dog's data is incomplete if they have raced at any track that has
    coverage gaps.  This is a conservative check — it doesn't vary by
    date window (which would be too expensive per-entry).

    Returns {dog_id: True/False}
    """
    result: dict[int, bool] = {}
    for dog_id, history in dog_histories.items():
        if history.empty:
            result[dog_id] = True
            continue
        dog_track_ids = set(history["track_id"].unique())
        result[dog_id] = not bool(dog_track_ids & gap_track_ids)
    return result


def materialize_feature(
    db: Session,
    feature_def: FeatureDefinition,
    race_entry_ids: list[int] | None = None,
    force: bool = False,
    version_id: int | None = None,
) -> dict[str, int]:
    """
    Materialize a single feature for specified race entries (or all resulted entries).

    Args:
        version_id: If provided, computed features are stored under this
            FeatureVersion.  Each version is an independent snapshot — the
            same (entry, feature) pair can exist once per version.  When
            None, features are stored unversioned and upserted in place
            (legacy behaviour).

    Returns {"computed": N, "skipped": N, "errors": N, "incomplete": N}
    """
    stats = {"computed": 0, "skipped": 0, "errors": 0, "incomplete": 0}

    # Cache attributes before any expunge_all() detaches the ORM object
    feature_name = feature_def.name
    feature_id = feature_def.id
    feature_type = feature_def.feature_type
    feature_config = feature_def.config_json
    feature_code = feature_def.code

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
        # Exclude entries that already have this feature computed (for this version)
        already_computed = (
            db.query(ComputedFeature.race_entry_id)
            .filter(ComputedFeature.feature_def_id == feature_id)
        )
        if version_id is not None:
            already_computed = already_computed.filter(
                ComputedFeature.version_id == version_id,
            )
        else:
            already_computed = already_computed.filter(
                ComputedFeature.version_id.is_(None),
            )
        entries_query = entries_query.filter(~RaceEntry.id.in_(already_computed))

    entry_ids = [row[0] for row in entries_query.all()]

    if not entry_ids:
        logger.info("No entries to compute for feature %s", feature_name)
        return stats

    logger.info(
        "Materializing feature '%s' for %d entries (version_id=%s)",
        feature_name, len(entry_ids), version_id,
    )

    # --- Pre-compute coverage gaps ONCE for the full date range ---
    date_range = (
        db.query(func.min(Race.race_date), func.max(Race.race_date))
        .filter(Race.status == "resulted")
        .first()
    )
    gap_track_ids: set[int] = set()
    if date_range and date_range[0]:
        active_tracks = db.query(Track.id, Track.code).filter(Track.active.is_(True)).all()
        code_to_id = {code: tid for tid, code in active_tracks}
        gaps = find_coverage_gaps(db, date_range[0], date_range[1])
        gap_track_ids = {code_to_id[g["track_code"]] for g in gaps if g["track_code"] in code_to_id}

    # --- Process in large batches to amortize DB round-trips ---
    batch_size = 2000
    for i in range(0, len(entry_ids), batch_size):
        batch = entry_ids[i:i + batch_size]

        # 1. Batch-load all race contexts for this batch (1 query)
        contexts = _batch_load_contexts(db, batch)

        # 2. Collect unique dog IDs and batch-load their histories (1 query)
        dog_ids = {ctx["dog_id"] for ctx in contexts.values()}
        histories = _batch_load_dog_histories(db, dog_ids)

        # 3. Pre-compute completeness per dog (no queries — uses cached gaps)
        completeness = _precompute_completeness(db, histories, gap_track_ids)

        # 4. Compute features and build insert list
        new_features: list[ComputedFeature] = []

        for entry_id in batch:
            ctx = contexts.get(entry_id)
            if not ctx:
                stats["errors"] += 1
                continue

            dog_id = ctx["dog_id"]
            race_date = ctx["race_date"]

            # Filter dog's full history to only races before this date
            full_history = histories.get(dog_id)
            if full_history is not None and not full_history.empty:
                history = full_history[full_history["race_date"] < race_date].copy()
                # Keep only most recent 100 (same as original limit)
                history = history.tail(100).reset_index(drop=True)
            else:
                history = pd.DataFrame()

            # Compute the feature
            value = None

            if feature_type == "visual":
                config = feature_config or {}
                value = compute_visual_feature(history, config, ctx)
            elif feature_type == "code":
                code = feature_code or ""
                value, error = execute_feature_code(code, history, ctx)
                if error:
                    stats["errors"] += 1
                    continue

            data_complete = completeness.get(dog_id, True)
            if not data_complete:
                stats["incomplete"] += 1

            # For versioned: always insert.  For unversioned: also just
            # insert since we already filtered out existing entries above.
            new_features.append(ComputedFeature(
                race_entry_id=entry_id,
                feature_def_id=feature_id,
                value=value,
                computed_at=datetime.utcnow(),
                data_complete=data_complete,
                version_id=version_id,
            ))

            stats["computed"] += 1

        # Bulk insert the batch
        if new_features:
            try:
                db.bulk_save_objects(new_features)
                db.commit()
            except Exception as e:
                db.rollback()
                if "disk is full" in str(e) or "database or disk is full" in str(e):
                    # WAL file likely bloated — checkpoint and retry once
                    logger.warning(
                        "Disk full during batch commit for '%s', "
                        "running WAL checkpoint and retrying...",
                        feature_name,
                    )
                    _wal_checkpoint(db)
                    try:
                        db.bulk_save_objects(new_features)
                        db.commit()
                    except Exception as retry_err:
                        db.rollback()
                        logger.error(
                            "Retry failed for feature '%s': %s. "
                            "Returning partial results.",
                            feature_name, retry_err,
                        )
                        return stats
                else:
                    raise

            # Free session memory — these objects are persisted and no
            # longer needed in the identity map.
            db.expunge_all()

            # Reclaim WAL disk space between batches to prevent the WAL
            # file from growing unbounded during large materializations.
            _wal_checkpoint(db)

        logger.info(
            "Feature '%s': %d/%d entries computed (%d incomplete)",
            feature_name, min(i + len(batch), len(entry_ids)),
            len(entry_ids), stats["incomplete"],
        )

    if stats["incomplete"]:
        logger.warning(
            "Feature '%s': %d/%d entries flagged as INCOMPLETE (dog history "
            "may be missing races from tracks with scrape gaps)",
            feature_name, stats["incomplete"], stats["computed"],
        )

    logger.info(
        "Feature '%s' done: %d computed, %d skipped, %d errors, %d incomplete",
        feature_name, stats["computed"], stats["skipped"],
        stats["errors"], stats["incomplete"],
    )
    return stats


def _wal_checkpoint_truncate(db: Session) -> None:
    """Run a TRUNCATE WAL checkpoint to reclaim disk space.

    TRUNCATE mode waits for all readers to finish, checkpoints the entire
    WAL into the main database, then truncates the WAL file to zero bytes.
    Use this after large operations (full materialization, bulk deletes) to
    reclaim the WAL file space on disk.
    """
    try:
        raw_conn = db.get_bind().raw_connection()
        raw_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        logger.info("WAL TRUNCATE checkpoint completed — WAL file reclaimed")
    except Exception as e:
        logger.warning("WAL TRUNCATE checkpoint failed: %s", e)


def materialize_all_features(
    db: Session,
    race_entry_ids: list[int] | None = None,
    force: bool = False,
    version_id: int | None = None,
) -> dict[str, dict[str, int]]:
    """Materialize all enabled features, optionally into a named version."""
    features = db.query(FeatureDefinition).filter(FeatureDefinition.enabled.is_(True)).all()
    results = {}

    for feature_def in features:
        stats = materialize_feature(
            db, feature_def, race_entry_ids, force, version_id=version_id,
        )
        results[feature_def.name] = stats

    # After all features are materialized, do a full WAL checkpoint to
    # reclaim the WAL file space on disk.
    _wal_checkpoint_truncate(db)

    return results


def build_feature_matrix(
    db: Session,
    feature_ids: list[int],
    race_entry_ids: list[int] | None = None,
    only_complete: bool = False,
    version_id: int | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix (DataFrame) from computed features.
    Memory-efficient: queries directly into pandas via raw SQL.

    Args:
        only_complete: If True, exclude computed features flagged as
            data_complete=False.  This drops rows where *any* feature
            was computed with incomplete dog history.
        version_id: If provided, only use features from this version
            snapshot.  If None, uses unversioned (version_id IS NULL)
            features.

    Returns a DataFrame with race_entry_id as index and feature names as columns.
    """
    features = db.query(FeatureDefinition).filter(FeatureDefinition.id.in_(feature_ids)).all()
    feature_map = {f.id: f.name for f in features}

    if not feature_map:
        return pd.DataFrame()

    # Build optional SQL filters
    extra_filters = ""
    if only_complete:
        extra_filters += " AND data_complete = 1"
    if version_id is not None:
        extra_filters += f" AND version_id = {int(version_id)}"
    else:
        extra_filters += " AND version_id IS NULL"

    # Use raw SQL with pandas for memory efficiency — avoids building
    # millions of Python objects via SQLAlchemy ORM
    engine = db.get_bind()

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
                {extra_filters}
            """
            batch_df = pd.read_sql_query(sql, engine)
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
            {extra_filters}
        """
        long_df = pd.read_sql_query(sql, engine)

    if long_df.empty:
        return pd.DataFrame()

    # Map feature IDs to names
    long_df["feature_name"] = long_df["feature_def_id"].map(feature_map)

    # Pivot from long to wide format
    df = long_df.pivot(index="race_entry_id", columns="feature_name", values="value")
    df.index.name = "race_entry_id"

    return df


def get_feature_coverage(
    db: Session,
    version_id: int | None = None,
) -> list[dict[str, Any]]:
    """Get feature coverage stats (how many entries have each feature computed).

    Args:
        version_id: If provided, count only features for this version.
            If None, counts unversioned (version_id IS NULL) features.
    """
    features = db.query(FeatureDefinition).all()
    total_entries = db.query(func.count(RaceEntry.id)).scalar() or 0

    # Single query to get both computed and incomplete counts per feature
    version_filter = (
        ComputedFeature.version_id == version_id
        if version_id is not None
        else ComputedFeature.version_id.is_(None)
    )
    counts = (
        db.query(
            ComputedFeature.feature_def_id,
            func.count(ComputedFeature.id).label("computed"),
            func.sum(
                case((ComputedFeature.data_complete == False, 1), else_=0)
            ).label("incomplete"),
        )
        .filter(version_filter)
        .group_by(ComputedFeature.feature_def_id)
        .all()
    )
    counts_map = {row.feature_def_id: (row.computed, row.incomplete or 0) for row in counts}

    return [
        {
            "feature_id": f.id,
            "name": f.name,
            "display_name": f.display_name,
            "feature_type": f.feature_type,
            "enabled": f.enabled,
            "computed_count": counts_map.get(f.id, (0, 0))[0],
            "incomplete_count": counts_map.get(f.id, (0, 0))[1],
            "total_entries": total_entries,
            "coverage_pct": round(counts_map.get(f.id, (0, 0))[0] / total_entries * 100, 1) if total_entries > 0 else 0,
        }
        for f in features
    ]

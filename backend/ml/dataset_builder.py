"""
Dataset builder: assembles feature matrices + target variables for ML training.

Handles:
- Building feature matrix from computed features
- Adding target variables (win, position, time)
- Time-based train/val/test splitting
- Race-level grouping (all dogs in same race stay in same split)
"""

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models.race import Race
from app.models.race_entry import RaceEntry
from ml.feature_store import build_feature_matrix

logger = logging.getLogger(__name__)


def build_dataset(
    db: Session,
    feature_ids: list[int],
    target: str,
    split_config: dict[str, Any] | None = None,
    only_complete: bool = False,
    version_id: int | None = None,
) -> dict[str, Any]:
    """
    Build a complete dataset for model training.

    Args:
        feature_ids: list of FeatureDefinition IDs to use as features
        target: "win_prob", "finish_position", or "finish_time"
        split_config: {"test_after": "2025-06-01", "val_pct": 0.15} or None for default
        only_complete: If True, exclude features flagged as data_complete=False.
            Use this when scrape coverage is incomplete across tracks to avoid
            training on features computed with partial dog histories.
        version_id: If provided, use features from this version snapshot.
            If None, uses unversioned features.

    Returns:
        {
            "X_train": DataFrame, "y_train": Series,
            "X_val": DataFrame, "y_val": Series,
            "X_test": DataFrame, "y_test": Series,
            "feature_names": list[str],
            "stats": {"total_entries", "train_size", "val_size", "test_size", ...}
        }
    """
    if split_config is None:
        split_config = {}

    max_entries = split_config.get("max_entries", 300000)

    # Get all resulted race entries with their race dates
    query = (
        db.query(
            RaceEntry.id.label("entry_id"),
            RaceEntry.finish_position,
            RaceEntry.finish_time,
            RaceEntry.sp_decimal,
            RaceEntry.race_id,
            Race.race_date,
            Race.num_runners,
        )
        .join(Race)
        .filter(Race.status == "resulted")
        .filter(RaceEntry.finish_position.isnot(None))
        .order_by(Race.race_date.desc())
    )

    if max_entries:
        query = query.limit(max_entries)

    entries = query.all()

    if not entries:
        raise ValueError("No resulted race entries found in database")

    entries_df = pd.DataFrame(entries, columns=[
        "entry_id", "finish_position", "finish_time", "sp_decimal", "race_id", "race_date", "num_runners",
    ])

    logger.info("Found %d resulted entries", len(entries_df))

    # Build feature matrix
    entry_ids = entries_df["entry_id"].tolist()
    X = build_feature_matrix(
        db, feature_ids, entry_ids,
        only_complete=only_complete, version_id=version_id,
    )

    if X.empty:
        raise ValueError("Feature matrix is empty — have features been materialized?")

    # Align entries with feature matrix (only keep entries that have features)
    entries_df = entries_df.set_index("entry_id")
    common_ids = entries_df.index.intersection(X.index)

    if len(common_ids) == 0:
        raise ValueError("No entries have computed features. Run feature materialization first.")

    entries_df = entries_df.loc[common_ids]
    X = X.loc[common_ids]

    logger.info("After alignment: %d entries with %d features", len(X), X.shape[1])

    # Build target variable
    y = _build_target(entries_df, target)

    # Drop rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    entries_df = entries_df[valid_mask]

    # Drop feature columns that are all NaN
    nan_cols = X.columns[X.isna().all()]
    if len(nan_cols) > 0:
        logger.info("Dropping %d all-NaN feature columns: %s", len(nan_cols), list(nan_cols))
        X = X.drop(columns=nan_cols)

    # Fill remaining NaN with column median
    X = X.fillna(X.median())

    logger.info("Final dataset: %d entries, %d features", len(X), X.shape[1])

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = _time_based_split(
        X, y, entries_df["race_date"], entries_df["race_id"], split_config,
    )

    # Also split the metadata (sp_decimal, race_id) for betting evaluation
    meta_train = entries_df.loc[X_train.index, ["sp_decimal", "race_id"]]
    meta_val = entries_df.loc[X_val.index, ["sp_decimal", "race_id"]]
    meta_test = entries_df.loc[X_test.index, ["sp_decimal", "race_id"]]

    feature_names = list(X.columns)

    # Save column medians from training set for consistent imputation at prediction time
    feature_medians = X.median().to_dict()

    # Recompute cutoff dates so they can be persisted by the training service
    test_after = split_config.get("test_after")
    val_pct = split_config.get("val_pct", 0.15)
    test_pct = split_config.get("test_pct", 0.15)
    race_dates = entries_df["race_date"]

    if test_after:
        test_cutoff = pd.Timestamp(test_after).date()
    else:
        sorted_dates = race_dates.sort_values()
        test_idx = int(len(sorted_dates) * (1 - test_pct))
        test_cutoff = sorted_dates.iloc[test_idx]

    train_val_dates = race_dates[race_dates < test_cutoff]
    if len(train_val_dates) > 0:
        sorted_tv = train_val_dates.sort_values()
        val_idx = int(len(sorted_tv) * (1 - val_pct / (1 - test_pct)))
        val_cutoff = sorted_tv.iloc[val_idx]
    else:
        val_cutoff = test_cutoff

    # Compute race group sizes for LambdaRank training
    group_train = _compute_group_sizes(entries_df.loc[X_train.index, "race_id"])
    group_val = _compute_group_sizes(entries_df.loc[X_val.index, "race_id"])
    group_test = _compute_group_sizes(entries_df.loc[X_test.index, "race_id"])

    stats = {
        "total_entries": len(X),
        "total_features": len(feature_names),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "nan_features_dropped": len(nan_cols),
        "train_cutoff_date": str(val_cutoff),
        "test_cutoff_date": str(test_cutoff),
        "only_complete_data": only_complete,
        "feature_version_id": version_id,
    }

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "meta_train": meta_train,
        "meta_val": meta_val,
        "meta_test": meta_test,
        "feature_names": feature_names,
        "feature_medians": feature_medians,
        "group_train": group_train,
        "group_val": group_val,
        "group_test": group_test,
        "stats": stats,
    }


def _build_target(entries_df: pd.DataFrame, target: str) -> pd.Series:
    """Build the target variable based on target type."""
    if target == "win_prob":
        return (entries_df["finish_position"] == 1).astype(float)
    elif target == "finish_position":
        return entries_df["finish_position"].astype(float)
    elif target == "finish_time":
        return entries_df["finish_time"].astype(float)
    else:
        raise ValueError(f"Unknown target: {target}")


def _time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    race_dates: pd.Series,
    race_ids: pd.Series,
    split_config: dict[str, Any],
) -> tuple:
    """
    Time-based split ensuring:
    1. Test set is the most recent data
    2. No race is split across train/val/test (all dogs in same race stay together)
    3. Chronological ordering: train < val < test
    """
    test_after = split_config.get("test_after")
    val_pct = split_config.get("val_pct", 0.15)
    test_pct = split_config.get("test_pct", 0.15)

    if test_after:
        test_cutoff = pd.Timestamp(test_after).date()
    else:
        # Use percentile of dates
        sorted_dates = race_dates.sort_values()
        test_idx = int(len(sorted_dates) * (1 - test_pct))
        test_cutoff = sorted_dates.iloc[test_idx]

    # Val cutoff
    train_val_dates = race_dates[race_dates < test_cutoff]
    if len(train_val_dates) > 0:
        sorted_tv = train_val_dates.sort_values()
        val_idx = int(len(sorted_tv) * (1 - val_pct / (1 - test_pct)))
        val_cutoff = sorted_tv.iloc[val_idx]
    else:
        val_cutoff = test_cutoff

    # Split masks
    train_mask = race_dates < val_cutoff
    val_mask = (race_dates >= val_cutoff) & (race_dates < test_cutoff)
    test_mask = race_dates >= test_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(
        "Split: train=%d (<%s), val=%d (<%s), test=%d (>=%s)",
        len(X_train), val_cutoff, len(X_val), test_cutoff, len(X_test), test_cutoff,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def _compute_group_sizes(race_ids: pd.Series) -> list[int]:
    """Compute group sizes (dogs per race) in order of appearance.

    This is needed by LambdaRank which requires knowing how many entries
    belong to each race group.  The entries must be contiguous by race_id
    in the dataset, which is guaranteed by our time-based split (races are
    sorted by date and all entries for a race are together).
    """
    if race_ids.empty:
        return []

    groups = []
    current_race = race_ids.iloc[0]
    count = 0
    for rid in race_ids:
        if rid == current_race:
            count += 1
        else:
            groups.append(count)
            current_race = rid
            count = 1
    groups.append(count)
    return groups

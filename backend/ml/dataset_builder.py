"""
Dataset builder: assembles feature matrices + target variables for ML training.

Handles:
- Building feature matrix from computed features
- Computing built-in race-context features (trap bias, grade movement, etc.)
- Adding target variables (win, position, time)
- Time-based train/val/test splitting
- Race-level grouping (all dogs in same race stay in same split)
"""

import gc
import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import and_, func
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
    include_builtin_features: bool = True,
    include_sp_features: bool = True,
    include_pace_shape_features: bool = True,
    include_race_relative_features: bool = True,
    include_elo_features: bool = True,
    include_odds_snapshot_features: bool = True,
    include_h2h_features: bool = True,
    heartbeat_fn: Any | None = None,
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
        include_builtin_features: If True, add built-in race-context features
            (trap bias, grade movement, days since last, weight change, etc.)
        include_sp_features: If True, add SP-derived features (current_sp_decimal,
            current_sp_implied_prob, sp_rank_in_field, market_overround).
        include_pace_shape_features: If True, add pace-shape features derived from
            is_front_runner / early_speed_ratio (num_front_runners_in_race,
            is_sole_front_runner, pace_pressure, early_speed_rank, is_predicted_leader).
        include_race_relative_features: If True, add race-relative features
            (per-feature vs_field and rank-in-field columns, plus num_runners).

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

    max_entries = split_config.get("max_entries", 50000)

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

    if X.empty and not include_builtin_features:
        raise ValueError("Feature matrix is empty — have features been materialized?")

    # Compute built-in race-context features
    if include_builtin_features:
        logger.info("Computing built-in race-context features...")
        builtin_X = _compute_builtin_features(db, entry_ids, heartbeat_fn=heartbeat_fn)
        if not builtin_X.empty:
            if X.empty:
                X = builtin_X
            else:
                # Drop overlapping columns from X so builtin versions take precedence
                overlap = X.columns.intersection(builtin_X.columns)
                if len(overlap) > 0:
                    logger.info("Dropping %d overlapping columns from computed features: %s", len(overlap), list(overlap))
                    X = X.drop(columns=overlap)
                X = X.join(builtin_X, how="outer")
            logger.info("Added %d built-in features, matrix now %d columns", builtin_X.shape[1], X.shape[1])

    # Compute ELO ratings (overall, per-distance, per-track) and field-relative
    # ELO features in one chronological pass over all resulted races.
    if include_elo_features:
        logger.info("Computing ELO rating features...")
        from ml.race_features import compute_elo_features_batch
        elo_X = compute_elo_features_batch(db, entry_ids, heartbeat_fn=heartbeat_fn)
        if not elo_X.empty:
            if X.empty:
                X = elo_X
            else:
                overlap = X.columns.intersection(elo_X.columns)
                if len(overlap) > 0:
                    X = X.drop(columns=overlap)
                X = X.join(elo_X, how="outer")
            logger.info("Added %d ELO features, matrix now %d columns",
                        elo_X.shape[1], X.shape[1])

    # Head-to-head features against the specific opponents in today's race
    if include_h2h_features:
        logger.info("Computing head-to-head features...")
        from ml.race_features import compute_h2h_features_batch
        h2h_X = compute_h2h_features_batch(db, entry_ids, heartbeat_fn=heartbeat_fn)
        if not h2h_X.empty:
            if X.empty:
                X = h2h_X
            else:
                overlap = X.columns.intersection(h2h_X.columns)
                if len(overlap) > 0:
                    X = X.drop(columns=overlap)
                X = X.join(h2h_X, how="outer")
            logger.info("Added %d H2H features, matrix now %d columns",
                        h2h_X.shape[1], X.shape[1])

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

    # Add SP-derived features from the current race
    if include_sp_features:
        X = _add_sp_features(X, entries_df)

    # Market-drift features from odds_snapshots (no-op if table is empty)
    if include_odds_snapshot_features:
        X = _add_odds_snapshot_features(db, X, entries_df)

    # Add race-relative features (compare each dog to its race field)
    if include_race_relative_features:
        X = add_race_relative_features(X, entries_df["race_id"])

    # Add pace shape features (front-runner count, early speed rank, etc.)
    if include_pace_shape_features:
        X = _add_pace_shape_features(X, entries_df["race_id"])

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

    # Fill remaining NaN with column median; columns where ALL values are NaN
    # get median=NaN, so fill those with 0.0 as a safe default
    X = X.fillna(X.median()).fillna(0.0)

    logger.info("Final dataset: %d entries, %d features", len(X), X.shape[1])

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = _time_based_split(
        X, y, entries_df["race_date"], entries_df["race_id"], split_config,
    )

    # Also split the metadata (sp_decimal, race_id) for betting evaluation
    meta_train = entries_df.loc[X_train.index, ["sp_decimal", "race_id", "race_date"]]
    meta_val = entries_df.loc[X_val.index, ["sp_decimal", "race_id", "race_date"]]
    meta_test = entries_df.loc[X_test.index, ["sp_decimal", "race_id", "race_date"]]

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
        test_idx = min(int(len(sorted_dates) * (1 - test_pct)), len(sorted_dates) - 1)
        test_cutoff = sorted_dates.iloc[test_idx]

    train_val_dates = race_dates[race_dates < test_cutoff]
    if len(train_val_dates) > 0:
        sorted_tv = train_val_dates.sort_values()
        val_idx = min(int(len(sorted_tv) * (1 - val_pct / (1 - test_pct))), len(sorted_tv) - 1)
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
        test_idx = min(int(len(sorted_dates) * (1 - test_pct)), len(sorted_dates) - 1)
        test_cutoff = sorted_dates.iloc[test_idx]

    # Val cutoff
    train_val_dates = race_dates[race_dates < test_cutoff]
    if len(train_val_dates) > 0:
        sorted_tv = train_val_dates.sort_values()
        val_idx = min(int(len(sorted_tv) * (1 - val_pct / (1 - test_pct))), len(sorted_tv) - 1)
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

    if len(X_train) == 0:
        raise ValueError(
            f"Training split is empty (val_cutoff={val_cutoff}). "
            "Check split_config — the dataset may be too small or date range too narrow."
        )
    if len(X_val) == 0:
        logger.warning("Validation split is empty — calibration will be skipped")
    if len(X_test) == 0:
        logger.warning("Test split is empty — evaluation metrics will be limited")

    logger.info(
        "Split: train=%d (<%s), val=%d (<%s), test=%d (>=%s)",
        len(X_train), val_cutoff, len(X_val), test_cutoff, len(X_test), test_cutoff,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def _compute_builtin_features(db: Session, entry_ids: list[int],
                               heartbeat_fn=None) -> pd.DataFrame:
    """Compute built-in race-context features for a list of entries.

    Uses bulk queries to avoid the N+1 problem — instead of ~8 DB queries per
    entry, this runs ~10 aggregate queries total then assembles features in memory.

    Processes in batches to limit peak memory — each batch loads dog histories
    only for its subset of entries.
    """
    from ml.race_features import compute_builtin_features_batch

    BATCH_SIZE = 5000
    if len(entry_ids) <= BATCH_SIZE:
        return compute_builtin_features_batch(db, entry_ids, heartbeat_fn=heartbeat_fn)

    all_dfs = []
    for i in range(0, len(entry_ids), BATCH_SIZE):
        batch = entry_ids[i:i + BATCH_SIZE]
        logger.info("Computing builtin features batch %d-%d of %d",
                     i, min(i + BATCH_SIZE, len(entry_ids)), len(entry_ids))
        batch_df = compute_builtin_features_batch(db, batch, heartbeat_fn=heartbeat_fn)
        if not batch_df.empty:
            all_dfs.append(batch_df)
        gc.collect()

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs)


def generate_walk_forward_fold_indices(
    race_ids: pd.Series,
    race_dates: pd.Series,
    n_folds: int,
    embargo_days: int = 0,
    min_train_pct: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window walk-forward CV fold indices.

    Each fold keeps all dogs of the same race together, enforces strict
    chronological order (train races all precede val races), and applies
    a `embargo_days`-day gap between the last training race and the
    first val race to prevent leakage from features computed over rolling
    windows that straddle the cutoff.

    Args:
        race_ids: per-row race id, aligned positionally with race_dates.
                  Must be sorted chronologically (the caller has already
                  pre-sorted by race_date for the single-split pipeline,
                  so this typically holds).
        race_dates: per-row race date, same length as race_ids.
        n_folds: number of folds to generate.  n_folds=1 returns a single
                  fold whose val set is the last (1 - min_train_pct) of
                  races — equivalent to the current single-split behaviour.
        embargo_days: minimum number of days between the last train race
                  and the first val race; val races within this gap are
                  skipped.
        min_train_pct: minimum fraction of races that must be in the
                  training set for the first fold (expanding window starts
                  at this size and grows for each subsequent fold).

    Returns:
        List of (train_idx, val_idx) tuples where each is a numpy array of
        positional indices into `race_ids`/`race_dates`.
    """
    from datetime import timedelta as _td

    if n_folds < 1:
        return []

    # Race-aligned sequence (preserve appearance order — race_ids is sorted
    # chronologically so drop_duplicates yields chronological order).
    ord_idx = race_ids.index
    unique_races_series = race_ids.drop_duplicates()
    unique_race_ids = unique_races_series.values
    unique_race_dates = race_dates.loc[unique_races_series.index].values
    n_races = len(unique_race_ids)
    if n_races < n_folds + 1:
        return []

    val_size = int(n_races * (1.0 - min_train_pct) / n_folds)
    val_size = max(1, val_size)
    train_start_size = max(1, n_races - val_size * n_folds)

    # Build a lookup from row index -> positional index within input
    pos_by_index = {idx: i for i, idx in enumerate(ord_idx)}

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        train_end_race = train_start_size + fold * val_size
        val_start_race = train_end_race
        val_end_race = min(val_start_race + val_size, n_races)
        if val_end_race <= val_start_race:
            break
        if train_end_race < 1:
            continue

        train_last_date = unique_race_dates[train_end_race - 1]
        # race_date may be date or datetime64; handle both
        try:
            embargo_cutoff = train_last_date + _td(days=embargo_days)
        except TypeError:
            # numpy.datetime64 arithmetic
            embargo_cutoff = train_last_date + np.timedelta64(embargo_days, "D")

        # Skip val races inside the embargo window
        while val_start_race < val_end_race and unique_race_dates[val_start_race] <= embargo_cutoff:
            val_start_race += 1
        if val_start_race >= val_end_race:
            continue

        train_race_set = set(unique_race_ids[:train_end_race])
        val_race_set = set(unique_race_ids[val_start_race:val_end_race])

        train_mask = race_ids.isin(train_race_set).values
        val_mask = race_ids.isin(val_race_set).values

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))

    return folds


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


# Registries of feature names emitted by the helper functions below.  These
# are surfaced through the /features/auto-injected API so the UI can list
# everything the training pipeline injects alongside user-defined features.
SP_FEATURE_NAMES = [
    "current_sp_decimal",
    "current_sp_implied_prob",
    "current_sp_log_odds",
    "sp_rank_in_field",
    "market_overround",
    "current_sp_devigged_prob",
    "is_favorite",
    "is_second_favorite",
    "fav_gap",
    "second_fav_gap",
    "sp_vs_field_mean",
]

ODDS_SNAPSHOT_FEATURE_NAMES = [
    "opening_to_sp_drift",
    "odds_steam_rate",
    "cross_book_disagreement",
]

PACE_SHAPE_FEATURE_NAMES = [
    "num_front_runners_in_race",
    "is_sole_front_runner",
    "pace_pressure",
    "early_speed_rank",
    "is_predicted_leader",
    "pace_scenario_lone_speed",
    "pace_scenario_duel",
    "pace_scenario_contested",
    "pace_scenario_no_speed",
    "expected_lead_probability",
    "avg_opponent_early_speed",
    "early_speed_vs_field",
    "running_style_mismatch",
]

# Base columns for which add_race_relative_features emits 5 variants each
# (__vs_field, __rank, __z_in_field, __gap_to_best, __is_field_best).
# Exposed so the UI can describe what field-relative features exist without
# having to enumerate every suffix.
RACE_RELATIVE_BASE_COLUMNS = [
    # Time / pace
    "mean_finish_time_last5",
    "min_finish_time_last10",
    "mean_adjusted_time_last5",
    "best_adjusted_time_last10",
    "ewm_adjusted_time_last10",
    "mean_sectional_last5",
    "stdev_finish_time_last5",
    "mean_beaten_dist_last5",
    # Position-based
    "mean_position_last5",
    "ewm_position_last10",
    "win_rate_last10",
    "place_rate_last10",
    "bayesian_win_rate",
    "bayesian_place_rate",
    # Market
    "mean_sp_last5",
    # Experience / freshness
    "career_runs",
    "days_since_last_race",
    # Trainer / sire / track
    "trainer_win_rate",
    "trainer_place_rate",
    "trainer_win_rate_at_track",
    "sire_progeny_win_rate",
    "track_speed_rating",
    # Tier 3 — speed figure
    "speed_figure_best_last10",
    "speed_figure_mean_last5",
    "speed_figure_ewm_last10",
    "career_peak_speed_figure",
    # Tier 1 — ELO
    "dog_elo",
    "dog_elo_at_distance",
    "dog_elo_at_track",
    # Tier 7 — class gap vs field median
    "dog_median_career_grade_index",
    # Comment-derived
    "quick_away_rate_last10",
    "led_at_bend1_rate_last10",
    "finish_well_rate_last10",
    "clear_win_rate_last10",
]

RACE_RELATIVE_SUFFIXES = [
    "__vs_field",
    "__rank",
    "__z_in_field",
    "__gap_to_best",
    "__is_field_best",
]


def _add_sp_features(X: pd.DataFrame, entries_df: pd.DataFrame) -> pd.DataFrame:
    """Add starting-price-derived features from the current race.

    SP is the single strongest predictor of race outcomes (Benter 1994).
    These features use information known before the race starts.

    In addition to raw SP and the naive 1/odds implied probability, this
    function emits:

      * current_sp_log_odds           — log(sp_decimal), more linear for GBMs
      * current_sp_devigged_prob      — 1/sp divided by the race's overround,
                                        removing the bookmaker margin
      * sp_rank_in_field              — 1 = favourite
      * market_overround              — sum of 1/sp across the field
      * is_favorite / is_second_fav   — flags for market top picks
      * fav_gap                       — log-odds gap to the favourite; positive
                                        = this dog is longer than the favourite
      * second_fav_gap                — log-odds gap to the second favourite
      * sp_vs_field_mean              — log-odds minus race log-odds mean
    """
    sp = entries_df["sp_decimal"]
    race_ids = entries_df["race_id"]

    valid_sp = sp.notna() & (sp > 0)
    if valid_sp.sum() == 0:
        return X

    X = X.copy()
    X["current_sp_decimal"] = sp
    X.loc[valid_sp, "current_sp_implied_prob"] = 1.0 / sp[valid_sp]

    # Log odds: more linear in skill space than raw decimal odds
    X.loc[valid_sp, "current_sp_log_odds"] = np.log(sp[valid_sp])

    # SP rank within the race (1 = favourite / shortest price)
    X["sp_rank_in_field"] = sp.groupby(race_ids).rank(method="min")

    # Market overround per race (sum of implied probs)
    implied = 1.0 / sp.where(valid_sp)
    overround = implied.groupby(race_ids).transform("sum")
    X["market_overround"] = overround

    # De-vigged implied probability: rescale by overround so each race sums to 1.
    # This removes the bookmaker margin and is a sharper estimator than the raw 1/odds.
    devigged = implied / overround.replace(0, np.nan)
    X["current_sp_devigged_prob"] = devigged

    # Favourite / second-favourite flags
    X["is_favorite"] = (X["sp_rank_in_field"] == 1).astype(float)
    X["is_second_favorite"] = (X["sp_rank_in_field"] == 2).astype(float)

    # Favourite gap: log-odds gap to the shortest-priced dog in the race.
    # Clipping negatives to 0 — a dog tied with the favourite has gap 0.
    log_odds = X["current_sp_log_odds"]
    min_log_odds = log_odds.groupby(race_ids).transform("min")
    X["fav_gap"] = (log_odds - min_log_odds).clip(lower=0.0)

    # Second-favourite gap: distance in log-odds to the second-shortest price.
    # Uses groupby.transform with a nsmallest-style rank.
    second_smallest = (
        log_odds
        .groupby(race_ids)
        .transform(lambda s: s.nsmallest(2).iloc[-1] if s.notna().sum() >= 2 else np.nan)
    )
    X["second_fav_gap"] = log_odds - second_smallest

    # Field-relative log-odds (deviation from race mean log-odds)
    race_log_odds_mean = log_odds.groupby(race_ids).transform("mean")
    X["sp_vs_field_mean"] = log_odds - race_log_odds_mean

    logger.info("Added 10 SP-derived features")
    return X


def _add_odds_snapshot_features(
    db: Session,
    X: pd.DataFrame,
    entries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add pre-race market-drift features from the `odds_snapshots` table.

    When the table is empty (i.e. the scraper is not yet collecting live
    odds snapshots) every snapshot-derived feature is left as NaN and the
    function is effectively a no-op — the median imputation downstream
    fills them with 0 without harming the model.

    Emits per-entry:
      * opening_to_sp_drift  — log(sp) − log(earliest snapshot odds).
                                Positive = drift out (less confidence),
                                negative = steamed in (sharp money).
      * odds_steam_rate      — slope of log(odds) over the last hour of
                                snapshots before the race.
      * cross_book_disagreement — stdev of implied prob at latest scrape,
                                across bookmakers.
    """
    from app.models.odds import OddsSnapshot

    if X.empty:
        return X

    entry_index = entries_df.index
    # No dog_id available on the input frames — fetch from DB once
    lookup = (
        db.query(
            RaceEntry.id.label("entry_id"),
            RaceEntry.dog_id,
            RaceEntry.race_id,
            RaceEntry.sp_decimal,
        )
        .filter(RaceEntry.id.in_(list(entry_index)))
        .all()
    )
    if not lookup:
        return X

    meta = pd.DataFrame(lookup, columns=["entry_id", "dog_id", "race_id", "sp_decimal"]).set_index("entry_id")
    race_ids = meta["race_id"].dropna().unique().tolist()

    # Early exit: if the snapshots table has no rows for any of these races
    # we can skip the per-entry work entirely.
    snap_count = (
        db.query(func.count(OddsSnapshot.id))
        .filter(OddsSnapshot.race_id.in_(race_ids))
        .scalar()
    )
    if not snap_count:
        logger.info("Skipped odds-snapshot features — no snapshots for target races")
        return X

    snap_rows = (
        db.query(
            OddsSnapshot.race_id,
            OddsSnapshot.dog_id,
            OddsSnapshot.bookmaker,
            OddsSnapshot.odds_decimal,
            OddsSnapshot.scraped_at,
            OddsSnapshot.is_sp,
        )
        .filter(
            OddsSnapshot.race_id.in_(race_ids),
            OddsSnapshot.odds_decimal > 0,
        )
        .all()
    )
    if not snap_rows:
        return X

    snaps = pd.DataFrame(snap_rows, columns=[
        "race_id", "dog_id", "bookmaker", "odds_decimal", "scraped_at", "is_sp",
    ])
    snaps["log_odds"] = np.log(snaps["odds_decimal"])
    snaps = snaps.sort_values(["race_id", "dog_id", "scraped_at"])

    drift: dict[int, float] = {}
    steam: dict[int, float] = {}
    disagreement: dict[int, float] = {}

    # Iterate per (race_id, dog_id) — typically 6 dogs * a few races, cheap
    for (race_id, dog_id), group in snaps.groupby(["race_id", "dog_id"]):
        pre_sp = group[group["is_sp"] == False]  # noqa: E712 — SQL-style
        if pre_sp.empty:
            continue
        first_row = pre_sp.iloc[0]
        last_row = pre_sp.iloc[-1]

        # Match this (race, dog) back to an entry_id
        matching = meta[(meta["race_id"] == race_id) & (meta["dog_id"] == dog_id)]
        if matching.empty:
            continue
        entry_id = matching.index[0]
        sp_decimal = matching.iloc[0]["sp_decimal"]

        if sp_decimal is not None and sp_decimal > 0:
            drift[entry_id] = float(np.log(sp_decimal) - first_row["log_odds"])

        # Steam: slope of log_odds over last hour of snapshots.  Positive slope
        # = drifting out, negative = steaming in.
        one_hour = pre_sp[pre_sp["scraped_at"] >= (last_row["scraped_at"] - pd.Timedelta(hours=1))]
        if len(one_hour) >= 3:
            times = (one_hour["scraped_at"] - one_hour["scraped_at"].iloc[0]).dt.total_seconds().values
            times = times.astype(float)
            ys = one_hour["log_odds"].values.astype(float)
            if times.max() > 0:
                slope = float(np.polyfit(times / 3600.0, ys, 1)[0])
                steam[entry_id] = slope

        # Cross-book disagreement: stdev of implied prob across bookmakers at
        # the latest scrape time per book.
        latest_per_book = (
            group.sort_values("scraped_at")
            .groupby("bookmaker")
            .tail(1)
        )
        if len(latest_per_book) >= 2:
            implied = 1.0 / latest_per_book["odds_decimal"]
            disagreement[entry_id] = float(implied.std())

    X = X.copy()
    X["opening_to_sp_drift"] = pd.Series(drift)
    X["odds_steam_rate"] = pd.Series(steam)
    X["cross_book_disagreement"] = pd.Series(disagreement)

    # Align to the full index so downstream NaN handling applies
    X["opening_to_sp_drift"] = X["opening_to_sp_drift"].reindex(X.index)
    X["odds_steam_rate"] = X["odds_steam_rate"].reindex(X.index)
    X["cross_book_disagreement"] = X["cross_book_disagreement"].reindex(X.index)

    logger.info(
        "Added odds-snapshot features: drift=%d, steam=%d, disagreement=%d non-null values",
        X["opening_to_sp_drift"].notna().sum(),
        X["odds_steam_rate"].notna().sum(),
        X["cross_book_disagreement"].notna().sum(),
    )
    return X


def _add_pace_shape_features(X: pd.DataFrame, race_ids: pd.Series) -> pd.DataFrame:
    """Add race-level pace-shape features that require seeing all runners.

    Pace shape is the single biggest source of value in greyhound racing that
    the market chronically mis-prices — a lone front-runner has a hugely
    favourable scenario, a closer in a no-speed race is set up to fail.

    Features emitted:
      num_front_runners_in_race   count of dogs with is_front_runner > 0.5
      is_sole_front_runner        dog is the only likely leader
      pace_pressure               is_front_runner score * race front-runner count
      early_speed_rank            rank (1 = fastest breaker) by early-speed ratio
      is_predicted_leader         1.0 if this dog has the fastest early speed
      pace_scenario_lone_speed    1.0 if exactly one front-runner in the field
      pace_scenario_duel          1.0 if exactly two front-runners
      pace_scenario_contested     1.0 if three or more front-runners
      pace_scenario_no_speed      1.0 if no front-runners
      expected_lead_probability   softmax over -early_speed_ratio across the field
      avg_opponent_early_speed    mean early-speed ratio of OTHER dogs in the race
      early_speed_vs_field        dog's early-speed gap to the field median
      running_style_mismatch      penalty score: negative when style suits the pace
                                  scenario, positive when it doesn't
    """
    X = X.copy()

    has_fr = "is_front_runner" in X.columns
    has_es = "early_speed_ratio" in X.columns

    if has_fr:
        front_runner_count = X["is_front_runner"].groupby(race_ids).transform(
            lambda x: (x > 0.5).sum()
        )
        X["num_front_runners_in_race"] = front_runner_count

        X["is_sole_front_runner"] = (
            (X["is_front_runner"] > 0.5) & (front_runner_count == 1)
        ).astype(float)

        X["pace_pressure"] = X["is_front_runner"] * front_runner_count

        # One-hot pace scenario
        X["pace_scenario_lone_speed"] = (front_runner_count == 1).astype(float)
        X["pace_scenario_duel"] = (front_runner_count == 2).astype(float)
        X["pace_scenario_contested"] = (front_runner_count >= 3).astype(float)
        X["pace_scenario_no_speed"] = (front_runner_count == 0).astype(float)

    if has_es:
        # Lower early_speed_ratio = faster to first bend, so rank ascending
        X["early_speed_rank"] = X["early_speed_ratio"].groupby(race_ids).rank(method="min")
        X["is_predicted_leader"] = (X["early_speed_rank"] == 1).astype(float)

        # Softmax over -early_speed_ratio so faster dogs get higher lead prob
        def _lead_softmax(s: pd.Series) -> pd.Series:
            valid = s.notna()
            if valid.sum() == 0:
                return pd.Series(np.nan, index=s.index)
            # Negate: we want small early_speed_ratio to map to high probability
            neg = -s
            neg_max = neg[valid].max()
            exp = np.exp(neg - neg_max)
            exp = exp.where(valid, 0.0)
            total = exp.sum()
            if total <= 0:
                return pd.Series(np.nan, index=s.index)
            return exp / total

        X["expected_lead_probability"] = (
            X["early_speed_ratio"].groupby(race_ids).transform(_lead_softmax)
        )

        # Average of OTHER dogs' early speed in the field: (total - self) / (n - 1)
        race_total = X["early_speed_ratio"].groupby(race_ids).transform("sum")
        race_count = X["early_speed_ratio"].groupby(race_ids).transform("count")
        with np.errstate(divide="ignore", invalid="ignore"):
            X["avg_opponent_early_speed"] = np.where(
                race_count > 1,
                (race_total - X["early_speed_ratio"]) / (race_count - 1),
                np.nan,
            )

        # Early-speed gap vs field median (negative = this dog breaks faster)
        race_median_es = X["early_speed_ratio"].groupby(race_ids).transform("median")
        X["early_speed_vs_field"] = X["early_speed_ratio"] - race_median_es

    # Running-style mismatch: combines pace scenario + the dog's own style.
    # A closer (low is_front_runner) in a no_speed race gets penalised (no one
    # will set up the race).  A front-runner in a lone_speed race gets bonus.
    if has_fr:
        fr = X["is_front_runner"].fillna(0.0)
        # Bonus when dog is a front-runner and scenario favours it
        lone_speed_bonus = fr * X.get("pace_scenario_lone_speed", 0.0)
        contested_penalty = fr * X.get("pace_scenario_contested", 0.0)
        # Closers (1 - fr) do well when the race is contested (chaos + tiring leaders)
        closer_boost = (1.0 - fr) * X.get("pace_scenario_contested", 0.0)
        closer_penalty = (1.0 - fr) * X.get("pace_scenario_no_speed", 0.0)
        X["running_style_mismatch"] = (
            closer_penalty + contested_penalty - lone_speed_bonus - closer_boost
        )

    added = [
        c for c in (
            "num_front_runners_in_race", "is_sole_front_runner", "pace_pressure",
            "early_speed_rank", "is_predicted_leader",
            "pace_scenario_lone_speed", "pace_scenario_duel",
            "pace_scenario_contested", "pace_scenario_no_speed",
            "expected_lead_probability", "avg_opponent_early_speed",
            "early_speed_vs_field", "running_style_mismatch",
        ) if c in X.columns
    ]
    if added:
        logger.info("Added %d pace-shape features", len(added))

    return X


# Features for which a higher value indicates a better dog.  Used to flip
# the sign of gap-to-best / rank semantics so positive deltas always mean
# "this dog is closer to the best in the field".
_HIGHER_IS_BETTER = {
    "win_rate_last10",
    "place_rate_last10",
    "career_runs",
    "bayesian_win_rate",
    "bayesian_place_rate",
    "trainer_win_rate",
    "trainer_place_rate",
    "trainer_win_rate_at_track",
    "sire_progeny_win_rate",
    "speed_figure_best_last10",
    "speed_figure_mean_last5",
    "speed_figure_ewm_last10",
    "career_peak_speed_figure",
    "dog_elo",
    "dog_elo_at_distance",
    "dog_elo_at_track",
    # Comment-derived rates where higher = better
    "quick_away_rate_last10",
    "led_at_bend1_rate_last10",
    "finish_well_rate_last10",
    "clear_win_rate_last10",
    # "mean_sp_last5" stays in the lower-is-better camp: shorter price = better
    # market assessment, even though numerically smaller.
}


def add_race_relative_features(X: pd.DataFrame, race_ids: pd.Series) -> pd.DataFrame:
    """Add race-relative features that compare each dog to the race field.

    These features can't be computed as standalone per-dog features because
    they need to see all dogs in the same race.  They are derived from the
    existing per-dog features by computing within-race statistics.

    For each tracked feature column, adds:
      - {col}__vs_field        — dog's value minus race mean
      - {col}__rank            — 1-based rank within race (1 = best, with the
                                  direction flipped per `_HIGHER_IS_BETTER`)
      - {col}__z_in_field      — within-race z-score (signed so positive =
                                  better than the field on this metric)
      - {col}__gap_to_best     — distance from the best dog in the field
                                  (always >= 0 — large positive = far from
                                  best, 0 = is the best)
      - {col}__is_field_best   — 1.0 if dog has the best value in the field,
                                  else 0.0

    Also adds:
      - num_runners: how many dogs in this race
    """
    if X.empty or race_ids.empty:
        return X

    X = X.copy()
    race_ids_aligned = race_ids.loc[X.index]

    cols_to_process = [c for c in RACE_RELATIVE_BASE_COLUMNS if c in X.columns]

    new_cols: dict[str, pd.Series] = {}

    for col in cols_to_process:
        col_vals = X[col]
        higher_is_better = col in _HIGHER_IS_BETTER
        sign = 1.0 if higher_is_better else -1.0

        race_mean = col_vals.groupby(race_ids_aligned).transform("mean")
        race_std = col_vals.groupby(race_ids_aligned).transform("std")

        # vs field mean — kept in raw direction for backwards compatibility
        new_cols[f"{col}__vs_field"] = col_vals - race_mean

        # Rank within race (1 = best dog in the field on this metric)
        new_cols[f"{col}__rank"] = col_vals.groupby(race_ids_aligned).rank(
            ascending=not higher_is_better, method="min",
        )

        # z-score, signed so positive = better than the field
        z_raw = (col_vals - race_mean) / race_std.replace(0, np.nan)
        z_signed = sign * z_raw
        new_cols[f"{col}__z_in_field"] = z_signed.replace([np.inf, -np.inf], np.nan)

        # Gap to best in the field — always >= 0
        if higher_is_better:
            best = col_vals.groupby(race_ids_aligned).transform("max")
            new_cols[f"{col}__gap_to_best"] = best - col_vals
        else:
            best = col_vals.groupby(race_ids_aligned).transform("min")
            new_cols[f"{col}__gap_to_best"] = col_vals - best

        # Is field best — flag the (tied) leader on this metric
        new_cols[f"{col}__is_field_best"] = (
            new_cols[f"{col}__gap_to_best"] == 0
        ).astype(float)

    if new_cols:
        X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)

    # Number of runners in the race
    X["num_runners"] = race_ids_aligned.groupby(race_ids_aligned).transform("count").astype(float)

    logger.info(
        "Added %d race-relative features (%d base columns x 5 + num_runners)",
        len(cols_to_process) * 5 + 1, len(cols_to_process),
    )

    return X

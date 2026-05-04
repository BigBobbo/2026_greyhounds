"""
Prediction service: generate predictions for races using a trained model.

1. Load a trained model (with preprocessing artifacts) from an experiment
2. Compute features for the target race entries
3. Generate predictions (win probability, position, time)
4. Compute confidence metrics (entropy, margin, edge)
5. Normalize win probabilities within each race via softmax
6. Compute place/show probabilities and forecast/trio combos via the
   Henery-discounted Plackett-Luce ordering service
7. Compute bankroll-aware staking recommendations (Kelly criterion)
8. Save predictions to DB
"""

import json
import logging
import math
import os
from datetime import date, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models.experiment import Experiment
from app.models.prediction import Prediction
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.dog import Dog
from app.models.track import Track
from app.services.feature_engine import get_dog_history, get_race_context, compute_visual_feature
from app.services.feature_sandbox import execute_feature_code
from app.services.race_ordering import (
    OrderingResult,
    compute_combo_kelly,
    compute_ordering,
)
from app.models.feature_definition import FeatureDefinition
from ml.feature_availability import (
    PredictionDataError,
    post_race_features_in_use,
)
from ml.race_features import compute_race_context_features

logger = logging.getLogger(__name__)


def load_trained_model(experiment: Experiment) -> dict[str, Any]:
    """Load a trained model artifact from disk.

    Returns a dict with keys: trainer, feature_medians, feature_names, is_ranking.
    Backwards-compatible with old models saved as raw trainer objects or
    artifacts missing `feature_names` (in which case we try to recover the
    list from attributes the trainer exposes).
    """
    if not experiment.model_path or not os.path.exists(experiment.model_path):
        raise FileNotFoundError(f"Model file not found: {experiment.model_path}")

    artifact = joblib.load(experiment.model_path)

    # Backwards compatibility: old models were saved as the trainer directly
    if not isinstance(artifact, dict):
        artifact = {
            "trainer": artifact,
            "feature_medians": {},
            "is_ranking": False,
        }

    # Recover feature_names from the trainer if the artifact predates the
    # feature-list-persistence fix.  LambdaRank stashes them on the trainer
    # itself; LightGBM/XGBoost expose them on the underlying model.
    if not artifact.get("feature_names"):
        artifact["feature_names"] = _extract_trainer_feature_names(artifact.get("trainer"))

    return artifact


def _extract_trainer_feature_names(trainer: Any) -> list[str]:
    """Best-effort recovery of the training feature list from a trainer."""
    if trainer is None:
        return []
    names = getattr(trainer, "_feature_names", None)
    if names:
        return list(names)
    model = getattr(trainer, "model", None)
    if model is not None:
        for attr in ("feature_names_in_", "feature_name_"):
            names = getattr(model, attr, None)
            if names is not None and len(names) > 0:
                return list(names)
        booster = getattr(model, "booster_", None)
        if booster is not None:
            try:
                names = booster.feature_name()
                if names:
                    return list(names)
            except Exception:
                pass
    return []


def compute_features_for_entries(
    db: Session,
    entry_ids: list[int],
    feature_defs: list[FeatureDefinition],
    include_builtin: bool = True,
    include_elo: bool = True,
) -> pd.DataFrame:
    """Compute features for specific race entries on-the-fly (no caching).

    Args:
        include_builtin: If True, also compute built-in race-context features
            (trap bias, grade movement, days since last, speed figures, etc.)
        include_elo: If True, also compute ELO rating features in a single
            chronological pass.  Required to keep prediction parity with
            training when ELO features are part of the model.
    """
    rows = {}

    for entry_id in entry_ids:
        ctx = get_race_context(db, entry_id)
        if not ctx:
            continue

        dog_id = ctx["dog_id"]
        race_date = ctx["race_date"]
        history = get_dog_history(db, dog_id, race_date)

        feature_values = {}
        for feat in feature_defs:
            value = None
            if feat.feature_type == "visual":
                config = feat.config_json or {}
                value = compute_visual_feature(history, config, ctx)
            elif feat.feature_type == "code" and feat.code:
                value, _ = execute_feature_code(feat.code, history, ctx)

            feature_values[feat.name] = value

        rows[entry_id] = feature_values

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "race_entry_id"

    # Add built-in race-context features in one batched pass.  This shares
    # the exact code path used at training time so prediction sees the same
    # speed-figure, trap-bias and trainer/sire features the model was trained on.
    if include_builtin:
        from ml.race_features import compute_builtin_features_batch
        builtin_df = compute_builtin_features_batch(db, list(rows.keys()))
        if not builtin_df.empty:
            overlap = df.columns.intersection(builtin_df.columns)
            if len(overlap) > 0:
                df = df.drop(columns=overlap)
            df = df.join(builtin_df, how="left")

    # Add ELO features (single chronological pass — slightly heavier but
    # also shared with the training pipeline).
    if include_elo:
        from ml.race_features import compute_elo_features_batch
        elo_df = compute_elo_features_batch(db, list(rows.keys()))
        if not elo_df.empty:
            overlap = df.columns.intersection(elo_df.columns)
            if len(overlap) > 0:
                df = df.drop(columns=overlap)
            df = df.join(elo_df, how="left")

    # Head-to-head features against today's opponents
    from ml.race_features import compute_h2h_features_batch
    h2h_df = compute_h2h_features_batch(db, list(rows.keys()))
    if not h2h_df.empty:
        overlap = df.columns.intersection(h2h_df.columns)
        if len(overlap) > 0:
            df = df.drop(columns=overlap)
        df = df.join(h2h_df, how="left")

    return df


def _raise_for_missing_features(
    X: pd.DataFrame,
    race_id: int,
    entries: list[Any],
) -> None:
    """Raise PredictionDataError listing every (entry, feature) NaN cell.

    Used on scheduled (upcoming) races where silent imputation would
    produce a distribution shift between train and serve. We list up to a
    handful of (trap, dog_name) pairs per offending feature so the user
    can see immediately which dogs the scrape didn't cover.
    """
    if not X.isna().any().any():
        return

    # Map entry_id -> (trap, dog_name) for friendly error output.
    entry_lookup: dict[int, tuple[int | None, str | None]] = {}
    for row in entries:
        entry = row.RaceEntry
        entry_lookup[entry.id] = (entry.trap, getattr(row, "dog_name", None))

    missing_summary: dict[str, list[str]] = {}
    for col in X.columns[X.isna().any()]:
        offenders = []
        for entry_id in X.index[X[col].isna()].tolist()[:6]:
            trap, dog_name = entry_lookup.get(int(entry_id), (None, None))
            label = f"trap{trap} {dog_name}" if dog_name else f"entry_id={entry_id}"
            offenders.append(label)
        missing_summary[str(col)] = offenders

    pretty = "; ".join(
        f"{feat} -> {', '.join(rows)}" for feat, rows in missing_summary.items()
    )
    raise PredictionDataError(
        f"Refusing to silently impute missing features for scheduled race "
        f"{race_id}. Missing cells: {pretty}. Either backfill the upstream "
        f"data (dog history, sectional times, comments) or retrain without "
        f"these features."
    )


def _get_train_cutoff(experiment: Experiment) -> date | None:
    """Extract the training data cutoff date from an experiment's split config."""
    split_config = experiment.split_config or {}
    cutoff_str = split_config.get("train_cutoff_date")
    if cutoff_str:
        return date.fromisoformat(cutoff_str)
    return None


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def _compute_confidence(probs: np.ndarray) -> dict[str, float]:
    """Compute confidence metrics for a race's probability distribution.

    Returns:
        entropy: Normalized entropy (0 = totally confident, 1 = uniform/no opinion)
        margin: Probability gap between 1st and 2nd pick
        confidence_score: Combined score (0 to 1, higher = more confident)
        confidence_tier: "strong", "moderate", "weak", or "avoid"
    """
    n = len(probs)
    if n <= 1:
        return {
            "entropy": 0.0,
            "margin": 1.0,
            "confidence_score": 1.0,
            "confidence_tier": "strong",
        }

    # Normalized entropy: 0 = one dog certain, 1 = uniform distribution
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    max_entropy = math.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Margin between top two picks
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]

    # Combined confidence score: weight margin and inverse entropy
    # Margin is more important for betting decisions
    confidence_score = 0.6 * (margin * n) + 0.4 * (1 - normalized_entropy)
    confidence_score = max(0.0, min(1.0, confidence_score))

    # Tier assignment
    if confidence_score >= 0.65 and margin >= 0.08:
        tier = "strong"
    elif confidence_score >= 0.40 and margin >= 0.04:
        tier = "moderate"
    elif confidence_score >= 0.20:
        tier = "weak"
    else:
        tier = "avoid"

    return {
        "entropy": round(normalized_entropy, 4),
        "margin": round(margin, 4),
        "confidence_score": round(confidence_score, 4),
        "confidence_tier": tier,
    }


def _compute_kelly_stake(
    win_prob: float,
    odds_decimal: float | None,
    bankroll: float = 100.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
) -> dict[str, Any]:
    """Compute Kelly criterion stake for a bet.

    Args:
        win_prob: Model's estimated win probability
        odds_decimal: Decimal odds (e.g. 3.5 means +250)
        bankroll: Current bankroll
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly, safer)
        min_edge: Minimum edge required to place a bet (default 5%)

    Returns dict with stake info or None if no bet recommended.
    """
    if odds_decimal is None or odds_decimal <= 1.0:
        return {"bet": False, "reason": "no_odds"}

    implied_prob = 1.0 / odds_decimal
    edge = win_prob - implied_prob

    if edge < min_edge:
        return {
            "bet": False,
            "reason": "insufficient_edge",
            "edge": round(edge, 4),
            "implied_prob": round(implied_prob, 4),
        }

    # Kelly formula: f* = (bp - q) / b
    # where b = odds - 1, p = win_prob, q = 1 - win_prob
    b = odds_decimal - 1
    f_star = (b * win_prob - (1 - win_prob)) / b

    # Cap at kelly_fraction of full Kelly for safety
    fractional_kelly = max(0, f_star * kelly_fraction)

    # Also cap at 5% of bankroll as absolute max
    max_stake_pct = 0.05
    stake_pct = min(fractional_kelly, max_stake_pct)
    stake = round(bankroll * stake_pct, 2)

    return {
        "bet": True,
        "stake": stake,
        "stake_pct": round(stake_pct * 100, 2),
        "full_kelly_pct": round(f_star * 100, 2),
        "edge": round(edge, 4),
        "implied_prob": round(implied_prob, 4),
        "expected_value": round(win_prob * (odds_decimal - 1) - (1 - win_prob), 4),
    }


def predict_race(
    db: Session,
    experiment_id: int,
    race_id: int,
    bankroll: float = 100.0,
    precomputed_features: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """
    Generate predictions for all entries in a race.

    Returns list of prediction dicts with dog info, confidence metrics,
    and Kelly staking recommendations.
    Raises ValueError if the race falls within the training data period.

    `precomputed_features` lets a batch caller (e.g. predict-by-date) supply
    a feature DataFrame indexed by race_entry_id covering many races at once,
    so the expensive ELO/builtin pass runs only once per request instead of
    per race. When provided, this race's slice is taken from it.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment or experiment.status != "completed":
        raise ValueError(f"Experiment {experiment_id} not found or not completed")

    # Load model artifact (trainer + preprocessing info + calibrator)
    artifact = load_trained_model(experiment)
    trainer = artifact["trainer"]
    feature_medians = artifact.get("feature_medians", {})
    trained_feature_names = artifact.get("feature_names", []) or []
    # Default to "median_fill" for any artifact saved before this field
    # was added — preserves legacy behaviour for older models.
    nan_policy = artifact.get("nan_policy", "median_fill")
    is_ranking = artifact.get("is_ranking", False)

    # Get race entries with SP odds
    entries = (
        db.query(RaceEntry, Dog.name.label("dog_name"), Race.race_date, Race.status)
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.race_id == race_id)
        .order_by(RaceEntry.trap)
        .all()
    )

    # Guard against predicting on training data
    if entries:
        race_date = entries[0].race_date
        if isinstance(race_date, datetime):
            race_date = race_date.date()
        train_cutoff = _get_train_cutoff(experiment)
        if train_cutoff and race_date < train_cutoff:
            raise ValueError(
                f"Race date {race_date} falls within the training data period "
                f"(before {train_cutoff}). Predicting on training data would give "
                f"misleadingly optimistic results. Use a race dated after {train_cutoff}, "
                f"or retrain with an earlier cutoff."
            )

    if not entries:
        return []

    entry_ids = [e.RaceEntry.id for e in entries]
    race_status = entries[0].status
    is_scheduled = race_status == "scheduled"

    # If the race hasn't run yet, refuse up-front to use any feature whose
    # value structurally depends on post-race data (current SP, weigh-in
    # weight, live odds-snapshot drift). The training pipeline has these
    # values for every resulted race in the dataset; at predict time they
    # come back NaN and would silently be filled with the training median,
    # producing a distribution shift that the user has explicitly asked us
    # to fail loudly on instead.
    if is_scheduled:
        offending = post_race_features_in_use(trained_feature_names)
        if offending:
            details = ", ".join(f"{name} ({why})" for name, why in offending.items())
            raise PredictionDataError(
                f"Experiment {experiment_id} was trained with post-race-only "
                f"features that are not available on a scheduled race "
                f"(race_id={race_id}): {details}. Retrain without these "
                f"features, or wait until after the race has been resulted."
            )

    # Honour the same feature-group toggles that were used at training time,
    # so predict-time feature columns align with what the model saw.
    split_cfg = experiment.split_config or {}
    include_builtin = split_cfg.get("include_builtin_features", True)
    include_relative = split_cfg.get("include_race_relative_features", True)
    include_pace_shape = split_cfg.get("include_pace_shape_features", True)
    # Default OFF: the odds_snapshots table is not currently populated by
    # the scraper, so leaving this on at training time produces all-NaN
    # columns that get median-filled to 0 in both train and predict, which
    # is a no-op today but flips into a real train/serve skew the moment
    # the snapshot scraper starts running for resulted races but not for
    # upcoming ones. Experiments that genuinely have a live odds feed can
    # opt back in by setting include_odds_snapshot_features=True in their
    # split_config.
    include_odds_snapshot = split_cfg.get("include_odds_snapshot_features", False)

    # Get feature definitions
    feature_defs = (
        db.query(FeatureDefinition)
        .filter(FeatureDefinition.id.in_(experiment.feature_set))
        .all()
    )

    # Compute features (or take a slice of a precomputed batch).
    if precomputed_features is not None and not precomputed_features.empty:
        available = [eid for eid in entry_ids if eid in precomputed_features.index]
        X = precomputed_features.loc[available].copy()
    else:
        X = compute_features_for_entries(
            db, entry_ids, feature_defs, include_builtin=include_builtin,
        )

    if X.empty:
        return []

    # Coerce to numeric: when every entry returns None for a feature
    # (e.g. future-date races where no dog history exists yet), pandas
    # infers `object` dtype, which XGBoost/LightGBM reject at predict time.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Per-entry data-completeness score, computed BEFORE any imputation:
    # the fraction of feature cells that came back with a real value.
    # Surfaced on each prediction so downstream Kelly staking can shrink
    # bet size when a dog's history is sparse rather than refusing the
    # whole race. Driver of the post-research design change away from
    # hard-refuse-on-any-NaN.
    if not X.empty:
        completeness_per_entry = X.notna().mean(axis=1).to_dict()
    else:
        completeness_per_entry = {}

    # NaN handling mirrors the training pipeline. Two policies:
    #
    #  - "passthrough" (GBM trainers): the model was fit on data with
    #    NaN in it and learned an optimal default split direction at each
    #    node, so we send NaN through here too. Median-filling at predict
    #    time would erase the missingness signal the model relies on.
    #    Critical: XGBoost trained without NaN routes NaN to the right
    #    branch by default, so train and serve must use the same policy.
    #
    #  - "median_fill" (sklearn trainers, plus every legacy artifact):
    #    fill with the training-set medians captured at fit time.
    #    Post-race-only features were already refused above on scheduled
    #    races, so this path only sees history-dependent NaN.
    if nan_policy == "passthrough":
        nan_count = int(X.isna().sum().sum())
        if nan_count > 0:
            logger.info(
                "predict_race(%s): NaN passthrough — preserving %d NaN "
                "cells (scheduled=%s)",
                race_id, nan_count, is_scheduled,
            )
    else:
        if feature_medians:
            nan_before = int(X.isna().sum().sum())
            X = X.fillna(feature_medians)
            if nan_before > 0:
                logger.info(
                    "predict_race(%s): median-filled %d NaN feature values "
                    "(scheduled=%s) — matches training-time imputation",
                    race_id, nan_before, is_scheduled,
                )
        X = X.fillna(0)

    # Ensure base feature columns from the FeatureDefinition set are
    # present so race-relative derivatives can be computed on them. Use
    # NaN as the sentinel under passthrough so the model sees genuine
    # missingness; use 0 under median_fill to match the legacy path.
    sentinel = float("nan") if nan_policy == "passthrough" else 0
    for col in (f.name for f in feature_defs):
        if col not in X.columns:
            X[col] = sentinel

    # Match the training pipeline order: odds-snapshot → race-relative → pace-shape.
    # Pace-shape and odds-snapshot were previously training-only, which silently
    # filled them with the training-set median at predict time — making them
    # constant across every dog in the race.  With most pace/market signal
    # collapsed to a constant, the only features that varied within a race
    # were trap-derived (trap_win_rate_at_track, early_speed_x_trap, …),
    # producing predictions that lined up with trap order.
    race_id_series = pd.Series(race_id, index=X.index, name="race_id")

    if include_odds_snapshot:
        from ml.dataset_builder import _add_odds_snapshot_features
        # _add_odds_snapshot_features only uses entries_df.index to filter
        # RaceEntry rows; it fetches race_id/dog_id/sp_decimal itself.
        entries_df = pd.DataFrame(index=X.index)
        X = _add_odds_snapshot_features(db, X, entries_df)

    if include_relative:
        from ml.dataset_builder import add_race_relative_features
        X = add_race_relative_features(X, race_id_series)

    if include_pace_shape:
        from ml.dataset_builder import _add_pace_shape_features
        X = _add_pace_shape_features(X, race_id_series)

    # Final alignment to the exact column set/order the model was trained on.
    # Race-relative / built-in / ELO / H2H features are generated dynamically
    # and the resulting column set depends on the data, so without this step
    # the model can see a different shape than it was fit on.
    if trained_feature_names:
        missing_cols = [c for c in trained_feature_names if c not in X.columns]
        if missing_cols:
            if is_scheduled:
                raise PredictionDataError(
                    f"Predict-time feature matrix for scheduled race "
                    f"{race_id} is missing columns the model was trained on: "
                    f"{missing_cols}. This usually means a feature group "
                    f"toggle (builtin/relative/pace-shape/odds-snapshot) was "
                    f"on at training time but off at predict time, or a "
                    f"FeatureDefinition was deleted. Re-enable the toggle in "
                    f"the experiment's split_config or retrain."
                )
            # Resulted-race backtests get the trained median (or NaN
            # under passthrough) so accuracy diagnostics still produce a
            # number rather than crashing on a missing dynamic column.
            for col in missing_cols:
                if nan_policy == "passthrough":
                    X[col] = float("nan")
                else:
                    X[col] = feature_medians.get(col, 0.0) if feature_medians else 0.0
        X = X[list(trained_feature_names)]

    # Final NaN sweep for the median_fill path — race-relative and
    # pace-shape derivations can introduce NaN when a feature was
    # constant within the race (denominator zero, etc.). Under
    # passthrough we leave them as NaN so the trainer's learned default
    # direction handles them.
    if nan_policy != "passthrough":
        X = X.fillna(0)

    # Generate predictions
    raw_scores = trainer.predict(X)

    # Compute probabilities (calibration is built into the trainer)
    if is_ranking:
        # LambdaRank: softmax + isotonic calibration (built into scores_to_proba)
        win_probs = trainer.scores_to_proba(raw_scores, group_sizes=[len(X)])
    elif hasattr(trainer, "predict_proba"):
        # Pointwise classifiers: predict_proba applies Platt calibration internally
        raw_proba = trainer.predict_proba(X)
        if raw_proba is not None:
            # Normalize within race so probabilities sum to 1
            total = raw_proba.sum()
            win_probs = raw_proba / total if total > 0 else raw_proba
        else:
            win_probs = None
    else:
        win_probs = None

    # Note: calibration is handled inside each trainer (Platt scaling).
    # A second Isotonic calibration layer was removed — it was compressing
    # edge signals and causing the model to underperform the SP baseline.

    # Compute race-level confidence metrics
    race_confidence = _compute_confidence(win_probs) if win_probs is not None else None

    # Key model outputs by entry_id (= X.index) so a partial X — e.g. an
    # entry was missing from `precomputed_features` and got dropped — can't
    # silently misalign with the entries list.  Position-based pairing was
    # safe in the happy path but failed loudly (or, worse, off-by-one
    # silently) when any row was dropped.
    eid_to_win_prob: dict[int, float] = {}
    eid_to_raw_score: dict[int, float] = {}
    if win_probs is not None:
        eid_to_win_prob = {int(eid): float(p) for eid, p in zip(X.index, win_probs)}
    if raw_scores is not None:
        eid_to_raw_score = {int(eid): float(s) for eid, s in zip(X.index, raw_scores)}

    # Forecast / trio layer: take the calibrated win probabilities and
    # expand them into ordered multi-position probabilities via the
    # Henery-discounted Plackett-Luce sampler in `race_ordering`. Cheap
    # — runs in a few milliseconds for a 6-dog field — and reuses the
    # already-trained win model rather than asking for a second model
    # head. Combos are deterministic per race (seeded by race_id) so
    # repeated calls return the same numbers.
    ordering: OrderingResult | None = None
    if eid_to_win_prob:
        ordering_entry_ids = [int(eid) for eid in X.index]
        ordering_win_probs = [
            float(eid_to_win_prob[eid]) for eid in ordering_entry_ids
        ]
        try:
            ordering = compute_ordering(
                entry_ids=ordering_entry_ids,
                win_probs=ordering_win_probs,
                seed=int(race_id) if race_id is not None else None,
            )
        except Exception as e:
            # Never let the ordering layer kill a prediction request — if
            # the Monte Carlo blows up we just fall back to win-only
            # output and log the failure for diagnosis.
            logger.warning(
                "predict_race(%s): ordering layer failed: %s — falling "
                "back to win-only predictions",
                race_id, e,
            )
            ordering = None

    # Pre-serialize the ordered combo lists once for storage — every dog
    # in the race gets the same JSON cached on its prediction row so a
    # single-row API fetch can render the combos panel.
    forecast_combos_payload: list[dict[str, Any]] = []
    trio_combos_payload: list[dict[str, Any]] = []
    forecast_combos_json: str | None = None
    trio_combos_json: str | None = None
    if ordering is not None:
        forecast_combos_payload = [
            {
                "first_entry_id": c.first_entry_id,
                "second_entry_id": c.second_entry_id,
                "probability": c.probability,
            }
            for c in ordering.forecast
        ]
        trio_combos_payload = [
            {
                "first_entry_id": c.first_entry_id,
                "second_entry_id": c.second_entry_id,
                "third_entry_id": c.third_entry_id,
                "probability": c.probability,
            }
            for c in ordering.trio
        ]
        if forecast_combos_payload:
            forecast_combos_json = json.dumps(forecast_combos_payload)
        if trio_combos_payload:
            trio_combos_json = json.dumps(trio_combos_payload)

    # Build prediction list
    predictions = []
    for entry_row, entry_id in zip(entries, entry_ids):
        entry = entry_row.RaceEntry
        dog_name = entry_row.dog_name

        # Data completeness for this entry — fraction of the feature
        # matrix that came back populated before any imputation. 1.0 = a
        # veteran with full history; 0.0 = a debutant whose features
        # were entirely median-filled. Surfaced for downstream Kelly
        # staking and UI confidence display.
        completeness = completeness_per_entry.get(entry_id)
        if completeness is not None:
            completeness = round(float(completeness), 4)

        pred_data = {
            "race_entry_id": entry_id,
            "dog_name": dog_name,
            "trap": entry.trap,
            "experiment_id": experiment_id,
            "data_completeness": completeness,
            "bankroll_used": bankroll,
            "sp_decimal_at_pred": entry.sp_decimal,
            # Forecast/trio outputs default to None and only get filled
            # in below for ranking experiments where the win-prob layer
            # produced a real probability for this dog.
            "place_probability": None,
            "show_probability": None,
            "forecast_combos": forecast_combos_payload,
            "trio_combos": trio_combos_payload,
            "forecast_combos_json": forecast_combos_json,
            "trio_combos_json": trio_combos_json,
        }

        if is_ranking or experiment.target == "win_prob":
            win_prob = eid_to_win_prob.get(int(entry_id))
            pred_data["win_probability"] = win_prob
            pred_data["predicted_position"] = None
            pred_data["predicted_time"] = None

            # Per-dog place/show probabilities from the ordering layer.
            # Ordering is keyed by entry_id so a dog dropped from the
            # feature matrix simply gets None here rather than an
            # off-by-one from another dog's draw.
            if ordering is not None:
                pred_data["place_probability"] = ordering.place_prob.get(
                    int(entry_id)
                )
                pred_data["show_probability"] = ordering.show_prob.get(
                    int(entry_id)
                )

            # Confidence score for this dog (race-level confidence * individual probability)
            if race_confidence and win_prob is not None:
                pred_data["confidence"] = round(race_confidence["confidence_score"], 4)
                pred_data["confidence_tier"] = race_confidence["confidence_tier"]
                pred_data["margin"] = race_confidence["margin"]
                pred_data["entropy"] = race_confidence["entropy"]

            # Kelly staking recommendation
            if win_prob is not None:
                kelly = _compute_kelly_stake(
                    win_prob, entry.sp_decimal, bankroll=bankroll,
                )
                pred_data["kelly"] = kelly
            else:
                pred_data["kelly"] = {"bet": False, "reason": "no_probability"}

            # Edge vs market
            if win_prob is not None and entry.sp_decimal and entry.sp_decimal > 1:
                implied = 1.0 / entry.sp_decimal
                pred_data["edge"] = round(win_prob - implied, 4)
                pred_data["is_value"] = win_prob > implied * 1.05  # 5% min edge
            else:
                pred_data["edge"] = None
                pred_data["is_value"] = None

        elif experiment.target == "finish_position":
            score = eid_to_raw_score.get(int(entry_id))
            pred_data["win_probability"] = None
            pred_data["predicted_position"] = score
            pred_data["predicted_time"] = None
            pred_data["confidence"] = None
        elif experiment.target == "finish_time":
            score = eid_to_raw_score.get(int(entry_id))
            pred_data["win_probability"] = None
            pred_data["predicted_position"] = None
            pred_data["predicted_time"] = score
            pred_data["confidence"] = None

        predictions.append(pred_data)

    # Sort by win probability (highest first) or predicted position/time
    if is_ranking or experiment.target == "win_prob":
        predictions.sort(key=lambda p: p.get("win_probability") or 0, reverse=True)
    elif experiment.target == "finish_position":
        predictions.sort(key=lambda p: p.get("predicted_position") or 999)
    elif experiment.target == "finish_time":
        predictions.sort(key=lambda p: p.get("predicted_time") or 999)

    return predictions


def compute_kelly_stake(
    win_prob: float,
    odds_decimal: float | None,
    bankroll: float = 100.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
) -> dict[str, Any]:
    """Public wrapper around the Kelly calculation. Same math used at
    prediction time — exposed so endpoints recomputing against
    user-supplied odds stay consistent with the original recommendation."""
    return _compute_kelly_stake(
        win_prob, odds_decimal, bankroll=bankroll,
        kelly_fraction=kelly_fraction, min_edge=min_edge,
    )


def recompute_kelly_for_saved_predictions(
    db: Session,
    experiment_id: int,
    race_id: int,
    odds_by_entry: dict[int, float | None],
    bankroll: float,
) -> list[dict[str, Any]]:
    """Recompute Kelly + edge for saved predictions using user-supplied odds.

    Updates the persisted Kelly snapshot in place — the user's odds become
    the new authoritative bet recommendation for this race + experiment, so
    the History view reflects what they actually decided to back.

    Returns the updated predictions in the same shape as predict_race.
    """
    rows = (
        db.query(Prediction, Dog.name.label("dog_name"), RaceEntry.trap)
        .join(RaceEntry, Prediction.race_entry_id == RaceEntry.id)
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .filter(
            Prediction.experiment_id == experiment_id,
            RaceEntry.race_id == race_id,
        )
        .all()
    )
    if not rows:
        return []

    now = datetime.utcnow()
    updated: list[dict[str, Any]] = []
    for pred, dog_name, trap in rows:
        odds = odds_by_entry.get(pred.race_entry_id)
        wp = pred.win_probability

        if wp is not None:
            kelly = compute_kelly_stake(wp, odds, bankroll=bankroll)
        else:
            kelly = {"bet": False, "reason": "no_probability"}

        # Edge vs market — mirrors predict_race's logic so saved rows stay
        # internally consistent (kelly.edge and pred.edge agree).
        if wp is not None and odds is not None and odds > 1:
            implied = 1.0 / odds
            edge = round(wp - implied, 4)
            is_value = wp > implied * 1.05
        else:
            edge = None
            is_value = None

        pred.kelly_bet = kelly.get("bet")
        pred.kelly_stake = kelly.get("stake")
        pred.kelly_stake_pct = kelly.get("stake_pct")
        pred.kelly_full_pct = kelly.get("full_kelly_pct")
        pred.kelly_expected_value = kelly.get("expected_value")
        pred.kelly_implied_prob = kelly.get("implied_prob")
        pred.kelly_reason = kelly.get("reason")
        pred.edge = edge
        pred.is_value = is_value
        pred.bankroll_used = bankroll
        # Note: sp_decimal_at_pred is the SP snapshot at original predict time;
        # don't clobber it. The user's odds are recoverable from
        # kelly_implied_prob (= 1 / odds) when the Kelly compute returns one.
        pred.updated_at = now

        updated.append(hydrate_saved_prediction(pred, dog_name, trap))

    db.commit()
    updated.sort(key=lambda p: p.get("win_probability") or 0, reverse=True)
    return updated


def hydrate_saved_prediction(
    pred: Prediction, dog_name: str | None, trap: int | None,
) -> dict[str, Any]:
    """Reconstruct a `predict_race`-shaped dict from a saved Prediction row.

    Lets the API hand back the same JSON shape whether predictions were
    just computed or fetched from the DB — so the frontend can render
    cached and live results identically.
    """
    kelly: dict[str, Any] = {
        "bet": bool(pred.kelly_bet) if pred.kelly_bet is not None else False,
    }
    if pred.kelly_reason is not None:
        kelly["reason"] = pred.kelly_reason
    if pred.kelly_stake is not None:
        kelly["stake"] = pred.kelly_stake
    if pred.kelly_stake_pct is not None:
        kelly["stake_pct"] = pred.kelly_stake_pct
    if pred.kelly_full_pct is not None:
        kelly["full_kelly_pct"] = pred.kelly_full_pct
    if pred.kelly_expected_value is not None:
        kelly["expected_value"] = pred.kelly_expected_value
    if pred.kelly_implied_prob is not None:
        kelly["implied_prob"] = pred.kelly_implied_prob
    if pred.edge is not None:
        kelly["edge"] = pred.edge

    forecast_combos: list[dict[str, Any]] = []
    trio_combos: list[dict[str, Any]] = []
    if pred.forecast_combos_json:
        try:
            forecast_combos = json.loads(pred.forecast_combos_json)
        except (json.JSONDecodeError, TypeError):
            forecast_combos = []
    if pred.trio_combos_json:
        try:
            trio_combos = json.loads(pred.trio_combos_json)
        except (json.JSONDecodeError, TypeError):
            trio_combos = []

    return {
        "race_entry_id": pred.race_entry_id,
        "dog_name": dog_name,
        "trap": trap,
        "experiment_id": pred.experiment_id,
        "win_probability": pred.win_probability,
        "place_probability": pred.place_probability,
        "show_probability": pred.show_probability,
        "predicted_position": pred.predicted_position,
        "predicted_time": pred.predicted_time,
        "confidence": pred.confidence,
        "confidence_tier": pred.confidence_tier,
        "margin": pred.margin,
        "entropy": pred.entropy,
        "edge": pred.edge,
        "is_value": pred.is_value,
        "kelly": kelly,
        "data_completeness": pred.data_completeness,
        "bankroll_used": pred.bankroll_used,
        "sp_decimal_at_pred": pred.sp_decimal_at_pred,
        "forecast_combos": forecast_combos,
        "trio_combos": trio_combos,
        "created_at": pred.created_at.isoformat() if pred.created_at else None,
        "updated_at": pred.updated_at.isoformat() if pred.updated_at else None,
    }


def get_saved_predictions_for_race(
    db: Session, experiment_id: int, race_id: int,
) -> list[dict[str, Any]]:
    """Fetch saved predictions for a (experiment, race) pair without recomputing.

    Returns rows in the same shape as `predict_race`, sorted by win
    probability descending. Empty list if no predictions are saved yet.
    """
    rows = (
        db.query(Prediction, Dog.name.label("dog_name"), RaceEntry.trap)
        .join(RaceEntry, Prediction.race_entry_id == RaceEntry.id)
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .filter(
            Prediction.experiment_id == experiment_id,
            RaceEntry.race_id == race_id,
        )
        .all()
    )
    preds = [hydrate_saved_prediction(p, name, trap) for p, name, trap in rows]
    preds.sort(key=lambda p: p.get("win_probability") or 0, reverse=True)
    return preds


def _flatten_prediction_for_storage(pred: dict[str, Any]) -> dict[str, Any]:
    """Map a `predict_race`-shaped dict onto Prediction column kwargs.

    Persists the full betting context (Kelly snapshot, edge, confidence
    breakdown, bankroll/SP at prediction time) so a saved prediction can
    be replayed in the UI without re-running the model.
    """
    kelly = pred.get("kelly") or {}
    return {
        "win_probability": pred.get("win_probability"),
        "place_probability": pred.get("place_probability"),
        "show_probability": pred.get("show_probability"),
        "predicted_position": pred.get("predicted_position"),
        "predicted_time": pred.get("predicted_time"),
        "confidence": pred.get("confidence"),
        "confidence_tier": pred.get("confidence_tier"),
        "margin": pred.get("margin"),
        "entropy": pred.get("entropy"),
        "edge": pred.get("edge"),
        "is_value": pred.get("is_value"),
        "kelly_bet": kelly.get("bet"),
        "kelly_stake": kelly.get("stake"),
        "kelly_stake_pct": kelly.get("stake_pct"),
        "kelly_full_pct": kelly.get("full_kelly_pct"),
        "kelly_expected_value": kelly.get("expected_value"),
        "kelly_implied_prob": kelly.get("implied_prob"),
        "kelly_reason": kelly.get("reason"),
        "data_completeness": pred.get("data_completeness"),
        "bankroll_used": pred.get("bankroll_used"),
        "sp_decimal_at_pred": pred.get("sp_decimal_at_pred"),
        # Combo cache: full forecast/trio JSON repeated on each row.
        # Storage cost is small (a 6-dog race has 30 forecasts and 120
        # trios at most) and the duplication keeps the read path simple.
        "forecast_combos_json": pred.get("forecast_combos_json"),
        "trio_combos_json": pred.get("trio_combos_json"),
    }


def save_predictions(db: Session, predictions: list[dict[str, Any]]) -> int:
    """Save predictions to DB, upserting on (experiment_id, race_entry_id)."""
    saved = 0
    now = datetime.utcnow()
    for pred in predictions:
        fields = _flatten_prediction_for_storage(pred)
        existing = (
            db.query(Prediction)
            .filter(
                Prediction.experiment_id == pred["experiment_id"],
                Prediction.race_entry_id == pred["race_entry_id"],
            )
            .first()
        )

        if existing:
            for k, v in fields.items():
                setattr(existing, k, v)
            existing.updated_at = now
        else:
            db.add(Prediction(
                experiment_id=pred["experiment_id"],
                race_entry_id=pred["race_entry_id"],
                created_at=now,
                updated_at=now,
                **fields,
            ))
        saved += 1

    db.commit()
    return saved


def predict_race_ensemble(
    db: Session,
    experiment_ids: list[int],
    race_id: int,
    weights: list[float] | None = None,
    bankroll: float = 100.0,
) -> list[dict[str, Any]]:
    """
    Generate ensemble predictions by combining multiple trained models.

    Averages calibrated win probabilities across models (weighted if weights provided),
    then re-normalizes to sum to 1 within the race.

    Args:
        experiment_ids: List of completed experiment IDs to ensemble
        weights: Optional weights for each model (must sum to 1). If None, equal weighting.
        bankroll: Current bankroll for Kelly staking

    Returns list of prediction dicts (same format as predict_race).
    """
    if len(experiment_ids) < 2:
        raise ValueError("Ensemble requires at least 2 experiments")

    if weights is None:
        weights = [1.0 / len(experiment_ids)] * len(experiment_ids)
    elif len(weights) != len(experiment_ids):
        raise ValueError("weights must match experiment_ids length")
    else:
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

    # Get individual predictions from each model
    all_predictions: list[list[dict[str, Any]]] = []
    for exp_id in experiment_ids:
        try:
            preds = predict_race(db, exp_id, race_id, bankroll=bankroll)
            if preds:
                all_predictions.append(preds)
        except Exception as e:
            logger.warning("Ensemble: experiment %d failed: %s", exp_id, e)

    if not all_predictions:
        raise ValueError("No experiments produced predictions")

    if len(all_predictions) == 1:
        return all_predictions[0]

    # Build a map of entry_id -> weighted average win_probability
    entry_probs: dict[int, float] = {}
    entry_data: dict[int, dict] = {}

    for model_idx, preds in enumerate(all_predictions):
        w = weights[model_idx] if model_idx < len(weights) else weights[-1]
        for pred in preds:
            eid = pred["race_entry_id"]
            wp = pred.get("win_probability") or 0.0
            entry_probs[eid] = entry_probs.get(eid, 0.0) + wp * w
            if eid not in entry_data:
                entry_data[eid] = pred.copy()

    # Re-normalize probabilities to sum to 1
    total_prob = sum(entry_probs.values())
    if total_prob > 0:
        for eid in entry_probs:
            entry_probs[eid] /= total_prob

    # Build final prediction list with ensemble probabilities
    # Batch-load SP decimals for all entries (avoids N+1 queries)
    all_entry_ids = list(entry_data.keys())
    sp_map = {}
    if all_entry_ids:
        sp_rows = (
            db.query(RaceEntry.id, RaceEntry.sp_decimal)
            .filter(RaceEntry.id.in_(all_entry_ids))
            .all()
        )
        sp_map = {r.id: r.sp_decimal for r in sp_rows}

    predictions = []
    for eid, base_pred in entry_data.items():
        win_prob = entry_probs[eid]
        pred = base_pred.copy()
        pred["win_probability"] = win_prob
        pred["experiment_id"] = experiment_ids  # list to indicate ensemble

        # Recompute Kelly staking with ensemble probability
        sp_decimal = sp_map.get(eid)

        if win_prob is not None and win_prob > 0:
            kelly = _compute_kelly_stake(win_prob, sp_decimal, bankroll=bankroll)
            pred["kelly"] = kelly

            if sp_decimal and sp_decimal > 1:
                implied = 1.0 / sp_decimal
                pred["edge"] = round(win_prob - implied, 4)
                pred["is_value"] = win_prob > implied * 1.05
            else:
                pred["edge"] = None
                pred["is_value"] = None

        predictions.append(pred)

    # Compute race-level confidence
    probs = np.array([p["win_probability"] for p in predictions])
    race_confidence = _compute_confidence(probs)
    for pred in predictions:
        pred["confidence"] = race_confidence["confidence_score"]
        pred["confidence_tier"] = race_confidence["confidence_tier"]
        pred["margin"] = race_confidence["margin"]
        pred["entropy"] = race_confidence["entropy"]

    predictions.sort(key=lambda p: p.get("win_probability") or 0, reverse=True)
    return predictions


def predict_upcoming_races(
    db: Session,
    experiment_id: int,
    bankroll: float = 100.0,
) -> list[dict[str, Any]]:
    """
    Generate predictions for all scheduled (upcoming) races.
    Returns grouped predictions by race.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment or experiment.status != "completed":
        raise ValueError(f"Experiment {experiment_id} not ready")

    # Find scheduled races
    scheduled_races = (
        db.query(Race, Track.name.label("track_name"), Track.code.label("track_code"))
        .join(Track)
        .filter(Race.status == "scheduled")
        .order_by(Race.race_date, Race.race_number)
        .all()
    )

    results = []
    for race_row in scheduled_races:
        race = race_row.Race
        try:
            preds = predict_race(db, experiment_id, race.id, bankroll=bankroll)
            if preds:
                save_predictions(db, preds)
                results.append({
                    "race_id": race.id,
                    "race_date": str(race.race_date),
                    "race_number": race.race_number,
                    "track_name": race_row.track_name,
                    "track_code": race_row.track_code,
                    "distance_m": race.distance_m,
                    "grade": race.grade,
                    "predictions": preds,
                })
        except Exception as e:
            logger.error("Failed to predict race %d: %s", race.id, e)

    return results

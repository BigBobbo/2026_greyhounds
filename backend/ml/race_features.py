"""
Built-in race-context features that require field-level or track-level data.

These features capture information that the visual feature builder cannot:
- Trap statistics at specific tracks/distances
- Grade movement direction
- Weight changes relative to recent average
- Early speed profiles and pace scenarios
- Days since last race (from DB field)

These are computed on-the-fly alongside user-defined features.
"""

import logging
from collections import defaultdict
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import case, func, text
from sqlalchemy.orm import Session

from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track

logger = logging.getLogger(__name__)

# Ordered grade list for computing grade movement (lower index = higher class)
GRADE_ORDER = [
    "OR", "S1", "S2", "S3", "S4", "S5",
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
]


def compute_race_context_features(
    db: Session,
    entry_id: int,
    dog_history: pd.DataFrame,
    race_context: dict[str, Any],
) -> dict[str, float | None]:
    """Compute all built-in race-context features for a single entry.

    Returns a dict of feature_name -> value. None values indicate the
    feature could not be computed (e.g., no prior history).
    """
    features: dict[str, float | None] = {}

    # 1. Trap bias: historical win rate for this trap at this track+distance
    features["trap_win_rate_at_track"] = _trap_win_rate(
        db, race_context.get("track_id"), race_context.get("distance_m"),
        race_context.get("trap"), race_context.get("race_date"),
    )

    # 2. Grade movement: positive = dropping (easier), negative = rising (harder)
    features["grade_movement"] = _grade_movement(
        dog_history, race_context.get("grade"),
    )

    # 3. Days since last race (from entry field, but also compute from history as fallback)
    entry = db.query(RaceEntry).filter(RaceEntry.id == entry_id).first()
    if entry and entry.days_since_last is not None:
        features["days_since_last"] = float(entry.days_since_last)
    elif not dog_history.empty:
        last_date = dog_history["race_date"].max()
        race_date = race_context.get("race_date")
        if last_date and race_date:
            features["days_since_last"] = float((race_date - last_date).days)
        else:
            features["days_since_last"] = None
    else:
        features["days_since_last"] = None

    # 4. Weight change vs recent average
    features["weight_change"] = _weight_change(
        dog_history, entry.weight_kg if entry else None,
    )

    # 5. Early speed ratio (sectional time / finish time) from recent races
    features["early_speed_ratio"] = _early_speed_ratio(dog_history)

    # 6. Is front runner (based on comments mentioning "led" or low sectional times)
    features["is_front_runner"] = _is_front_runner(dog_history)

    # 7. Number of prior races (experience)
    features["career_races"] = float(len(dog_history)) if not dog_history.empty else 0.0

    # 8. Consistency score (stdev of finish positions, lower = more consistent)
    features["position_consistency"] = _position_consistency(dog_history)

    # 9. Dog age in years (greyhounds peak at 2-3.5 years)
    dog_id = race_context.get("dog_id")
    dog = db.query(Dog).filter(Dog.id == dog_id).first() if dog_id else None

    if dog and dog.birth_date and race_context.get("race_date"):
        age = (race_context["race_date"] - dog.birth_date).days / 365.25
        features["dog_age_years"] = age
        features["dog_age_squared"] = age ** 2
    else:
        features["dog_age_years"] = None
        features["dog_age_squared"] = None

    # 10. Pace/trap interaction features
    trap = race_context.get("trap")
    early_speed = features.get("early_speed_ratio")
    front_runner = features.get("is_front_runner")

    if trap is not None and early_speed is not None:
        features["early_speed_x_trap"] = early_speed * trap
        features["early_speed_x_inside"] = early_speed * (1.0 if trap <= 2 else 0.0)
        features["early_speed_x_outside"] = early_speed * (1.0 if trap >= 5 else 0.0)
    else:
        features["early_speed_x_trap"] = None
        features["early_speed_x_inside"] = None
        features["early_speed_x_outside"] = None

    if trap is not None and front_runner is not None:
        features["front_runner_x_inside"] = front_runner * (1.0 if trap <= 2 else 0.0)
        features["front_runner_x_outside"] = front_runner * (1.0 if trap >= 5 else 0.0)
    else:
        features["front_runner_x_inside"] = None
        features["front_runner_x_outside"] = None

    # 11. Trainer performance stats (win/place rate overall and at this track)
    trainer_stats = _trainer_stats(
        db,
        dog.trainer_name if dog else None,
        race_context.get("track_id"),
        race_context.get("race_date"),
    )
    features.update(trainer_stats)

    # 12. Sire progeny stats (breeding quality signal)
    sire_stats = _sire_stats(
        db,
        dog.sire if dog else None,
        race_context.get("distance_m"),
        race_context.get("race_date"),
    )
    features.update(sire_stats)

    # 13. Track-normalized speed rating (dog's best time vs track/distance average)
    features["track_speed_rating"] = _track_speed_rating(
        db, dog_history, race_context.get("track_id"),
        race_context.get("distance_m"), race_context.get("race_date"),
    )

    return features


def _trap_win_rate(
    db: Session,
    track_id: int | None,
    distance_m: int | None,
    trap: int | None,
    before_date=None,
    min_races: int = 30,
) -> float | None:
    """Win rate for a specific trap at a track/distance combination."""
    if not all([track_id, trap]):
        return None

    query = (
        db.query(
            func.count(RaceEntry.id).label("total"),
            func.sum(
                case((RaceEntry.finish_position == 1, 1), else_=0)
            ).label("wins"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.track_id == track_id,
            Race.status == "resulted",
            RaceEntry.trap == trap,
            RaceEntry.finish_position.isnot(None),
        )
    )

    if distance_m:
        query = query.filter(Race.distance_m == distance_m)
    if before_date:
        query = query.filter(Race.race_date < before_date)

    row = query.first()
    if not row or not row.total or row.total < min_races:
        return None

    return float(row.wins) / float(row.total)


def _grade_movement(
    dog_history: pd.DataFrame,
    current_grade: str | None,
) -> float | None:
    """Grade movement: positive = dropping to easier grade, negative = rising.

    Returns the number of grade levels moved. E.g., A3 -> A5 = +2 (dropping).
    """
    if current_grade is None or dog_history.empty:
        return None

    last_grade = dog_history["grade"].dropna().iloc[-1] if "grade" in dog_history.columns else None
    if last_grade is None:
        return None

    current_grade_upper = current_grade.upper().strip()
    last_grade_upper = str(last_grade).upper().strip()

    try:
        current_idx = GRADE_ORDER.index(current_grade_upper)
        last_idx = GRADE_ORDER.index(last_grade_upper)
        return float(current_idx - last_idx)
    except ValueError:
        return None


def _weight_change(
    dog_history: pd.DataFrame,
    current_weight: float | None,
    n_recent: int = 5,
) -> float | None:
    """Weight change vs average of last N races. Positive = heavier than usual."""
    if current_weight is None or dog_history.empty:
        return None

    weights = dog_history["weight_kg"].dropna().tail(n_recent)
    if weights.empty:
        return None

    avg_weight = weights.mean()
    return float(current_weight - avg_weight)


def _early_speed_ratio(dog_history: pd.DataFrame, n_recent: int = 5) -> float | None:
    """Average (sectional_time / finish_time) over recent races.

    Lower ratio = faster to first bend relative to overall time = front-runner.
    """
    if dog_history.empty:
        return None

    recent = dog_history.tail(n_recent)
    valid = recent.dropna(subset=["sectional_time", "finish_time"])
    valid = valid[valid["finish_time"] > 0]

    if valid.empty:
        return None

    ratios = valid["sectional_time"] / valid["finish_time"]
    return float(ratios.mean())


def _is_front_runner(dog_history: pd.DataFrame, n_recent: int = 10) -> float:
    """Score 0-1 indicating how often the dog leads or runs prominently.

    Uses comment field keywords and low sectional times.
    """
    if dog_history.empty:
        return 0.0

    recent = dog_history.tail(n_recent)
    led_count = 0
    total = len(recent)

    for _, row in recent.iterrows():
        comment = str(row.get("comment", "")).lower()
        if any(kw in comment for kw in ["led", "ld", "disp ld", "disp lead", "made all"]):
            led_count += 1

    return float(led_count / max(total, 1))


def _position_consistency(dog_history: pd.DataFrame, n_recent: int = 10) -> float | None:
    """Standard deviation of finish positions (lower = more consistent)."""
    if dog_history.empty:
        return None

    positions = dog_history["finish_position"].dropna().tail(n_recent)
    if len(positions) < 2:
        return None

    return float(positions.std())


def _trainer_stats(
    db: Session,
    trainer_name: str | None,
    track_id: int | None,
    before_date=None,
    days_window: int = 90,
    min_runners: int = 20,
) -> dict[str, float | None]:
    """Trainer win/place rate in the last N days, overall and at this track."""
    result = {
        "trainer_win_rate": None,
        "trainer_place_rate": None,
        "trainer_win_rate_at_track": None,
    }

    if not trainer_name:
        return result

    cutoff = before_date - timedelta(days=days_window) if before_date else None

    base_query = (
        db.query(
            func.count(RaceEntry.id).label("total"),
            func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
            func.sum(case((RaceEntry.finish_position <= 3, 1), else_=0)).label("places"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .filter(
            Dog.trainer_name == trainer_name,
            Race.status == "resulted",
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date:
        base_query = base_query.filter(Race.race_date < before_date)
    if cutoff:
        base_query = base_query.filter(Race.race_date >= cutoff)

    row = base_query.first()
    if row and row.total and row.total >= min_runners:
        result["trainer_win_rate"] = float(row.wins) / float(row.total)
        result["trainer_place_rate"] = float(row.places) / float(row.total)

    # Trainer at this specific track
    if track_id:
        track_row = base_query.filter(Race.track_id == track_id).first()
        if track_row and track_row.total and track_row.total >= 10:
            result["trainer_win_rate_at_track"] = float(track_row.wins) / float(track_row.total)

    return result


def _sire_stats(
    db: Session,
    sire_name: str | None,
    distance_m: int | None,
    before_date=None,
    min_progeny_runs: int = 50,
) -> dict[str, float | None]:
    """Win rate and mean time of sire's progeny."""
    result = {"sire_progeny_win_rate": None, "sire_progeny_mean_time_at_dist": None}

    if not sire_name:
        return result

    query = (
        db.query(
            func.count(RaceEntry.id).label("total"),
            func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
        )
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Dog.sire == sire_name,
            Race.status == "resulted",
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date:
        query = query.filter(Race.race_date < before_date)

    row = query.first()
    if row and row.total and row.total >= min_progeny_runs:
        result["sire_progeny_win_rate"] = float(row.wins) / float(row.total)

    # Mean time of progeny at this distance
    if distance_m:
        time_query = (
            db.query(func.avg(RaceEntry.finish_time))
            .join(Dog, RaceEntry.dog_id == Dog.id)
            .join(Race, RaceEntry.race_id == Race.id)
            .filter(
                Dog.sire == sire_name,
                Race.status == "resulted",
                Race.distance_m == distance_m,
                RaceEntry.finish_time.isnot(None),
            )
        )
        if before_date:
            time_query = time_query.filter(Race.race_date < before_date)

        avg_time = time_query.scalar()
        if avg_time:
            result["sire_progeny_mean_time_at_dist"] = float(avg_time)

    return result


def _track_speed_rating(
    db: Session,
    dog_history: pd.DataFrame,
    track_id: int | None,
    distance_m: int | None,
    before_date=None,
    days_window: int = 180,
    min_races: int = 50,
) -> float | None:
    """Dog's best adjusted time vs the track/distance average.

    Negative = faster than standard (good). Positive = slower.
    """
    if dog_history.empty or not track_id or not distance_m:
        return None

    # Get dog's best adjusted time at this distance from history
    same_dist = dog_history[dog_history["distance_m"] == distance_m]
    if "adjusted_time" not in same_dist.columns:
        return None
    dog_times = same_dist["adjusted_time"].dropna().tail(10)
    if dog_times.empty:
        return None
    dog_best = float(dog_times.min())

    cutoff = before_date - timedelta(days=days_window) if before_date else None

    query = (
        db.query(func.count(RaceEntry.id), func.avg(RaceEntry.adjusted_time))
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.track_id == track_id,
            Race.distance_m == distance_m,
            Race.status == "resulted",
            RaceEntry.adjusted_time.isnot(None),
        )
    )
    if before_date:
        query = query.filter(Race.race_date < before_date)
    if cutoff:
        query = query.filter(Race.race_date >= cutoff)

    count, avg_time = query.first()
    if not count or count < min_races or avg_time is None:
        return None

    return float(dog_best - avg_time)


def compute_builtin_features_batch(
    db: Session,
    entry_ids: list[int],
) -> pd.DataFrame:
    """Compute all built-in race-context features via bulk queries.

    This replaces the per-entry approach (which issued ~8 DB queries per entry)
    with a small number of bulk queries followed by in-memory joins.
    """
    if not entry_ids:
        return pd.DataFrame()

    entry_set = set(entry_ids)

    # --- 1. Bulk fetch entry + race + track + dog context ---
    logger.info("Batch builtin: fetching entry/race/dog context for %d entries...", len(entry_ids))
    ctx_rows = (
        db.query(
            RaceEntry.id.label("entry_id"),
            RaceEntry.trap,
            RaceEntry.dog_id,
            RaceEntry.days_since_last,
            RaceEntry.weight_kg,
            RaceEntry.race_id,
            Race.track_id,
            Race.distance_m,
            Race.grade,
            Race.race_date,
            Race.race_type,
            Dog.birth_date,
            Dog.trainer_name,
            Dog.sire,
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .filter(RaceEntry.id.in_(entry_ids))
        .all()
    )

    if not ctx_rows:
        return pd.DataFrame()

    ctx_df = pd.DataFrame(ctx_rows, columns=[
        "entry_id", "trap", "dog_id", "days_since_last", "weight_kg", "race_id",
        "track_id", "distance_m", "grade", "race_date", "race_type",
        "birth_date", "trainer_name", "sire",
    ]).set_index("entry_id")

    logger.info("Batch builtin: fetched context for %d entries", len(ctx_df))

    # --- 2. Bulk fetch all dog histories (last 100 races per dog, before race date) ---
    # We need history per (dog_id, race_date) pair. Fetch all historical entries for
    # relevant dogs, then filter per-entry in memory.
    unique_dog_ids = ctx_df["dog_id"].unique().tolist()
    logger.info("Batch builtin: fetching histories for %d unique dogs...", len(unique_dog_ids))

    hist_rows = (
        db.query(
            RaceEntry.dog_id,
            RaceEntry.trap,
            RaceEntry.finish_position,
            RaceEntry.finish_time,
            RaceEntry.sectional_time,
            RaceEntry.adjusted_time,
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
            Race.going_allowance,
            Race.num_runners,
            Race.prize_money,
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.dog_id.in_(unique_dog_ids),
            Race.status == "resulted",
        )
        .order_by(RaceEntry.dog_id, Race.race_date.desc())
        .all()
    )

    hist_columns = [
        "dog_id", "trap", "finish_position", "finish_time", "sectional_time",
        "adjusted_time", "beaten_distance", "weight_kg", "sp_decimal",
        "starting_price", "comment", "race_date", "track_id", "distance_m",
        "grade", "race_type", "going", "going_allowance", "num_runners",
        "prize_money",
    ]
    all_hist_df = pd.DataFrame(hist_rows, columns=hist_columns) if hist_rows else pd.DataFrame(columns=hist_columns)

    # Index by dog for fast lookup
    dog_histories: dict[int, pd.DataFrame] = {}
    if not all_hist_df.empty:
        for dog_id, group in all_hist_df.groupby("dog_id"):
            dog_histories[dog_id] = group.sort_values("race_date").reset_index(drop=True)

    logger.info("Batch builtin: loaded %d history rows for %d dogs", len(all_hist_df), len(dog_histories))

    # --- 3. Bulk compute trap win rates (per track/distance/trap combo) ---
    unique_combos = ctx_df[["track_id", "distance_m", "trap"]].drop_duplicates()
    logger.info("Batch builtin: computing trap win rates for %d combos...", len(unique_combos))

    trap_stats = (
        db.query(
            Race.track_id,
            Race.distance_m,
            RaceEntry.trap,
            func.count(RaceEntry.id).label("total"),
            func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.status == "resulted",
            RaceEntry.finish_position.isnot(None),
        )
        .group_by(Race.track_id, Race.distance_m, RaceEntry.trap)
        .all()
    )
    # Key: (track_id, distance_m, trap) -> win_rate
    trap_win_rates: dict[tuple, float | None] = {}
    for row in trap_stats:
        if row.total and row.total >= 30:
            trap_win_rates[(row.track_id, row.distance_m, row.trap)] = float(row.wins) / float(row.total)

    # --- 4. Bulk compute trainer stats (win/place rate in last 90 days) ---
    unique_trainers = ctx_df["trainer_name"].dropna().unique().tolist()
    logger.info("Batch builtin: computing trainer stats for %d trainers...", len(unique_trainers))

    trainer_overall: dict[str, dict] = {}
    if unique_trainers:
        trainer_rows = (
            db.query(
                Dog.trainer_name,
                func.count(RaceEntry.id).label("total"),
                func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
                func.sum(case((RaceEntry.finish_position <= 3, 1), else_=0)).label("places"),
            )
            .join(Dog, RaceEntry.dog_id == Dog.id)
            .join(Race, RaceEntry.race_id == Race.id)
            .filter(
                Dog.trainer_name.in_(unique_trainers),
                Race.status == "resulted",
                RaceEntry.finish_position.isnot(None),
            )
            .group_by(Dog.trainer_name)
            .all()
        )
        for row in trainer_rows:
            if row.total and row.total >= 20:
                trainer_overall[row.trainer_name] = {
                    "win_rate": float(row.wins) / float(row.total),
                    "place_rate": float(row.places) / float(row.total),
                }

    # Trainer at track
    trainer_at_track: dict[tuple, float | None] = {}
    if unique_trainers:
        trainer_track_rows = (
            db.query(
                Dog.trainer_name,
                Race.track_id,
                func.count(RaceEntry.id).label("total"),
                func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
            )
            .join(Dog, RaceEntry.dog_id == Dog.id)
            .join(Race, RaceEntry.race_id == Race.id)
            .filter(
                Dog.trainer_name.in_(unique_trainers),
                Race.status == "resulted",
                RaceEntry.finish_position.isnot(None),
            )
            .group_by(Dog.trainer_name, Race.track_id)
            .all()
        )
        for row in trainer_track_rows:
            if row.total and row.total >= 10:
                trainer_at_track[(row.trainer_name, row.track_id)] = float(row.wins) / float(row.total)

    # --- 5. Bulk compute sire stats ---
    unique_sires = ctx_df["sire"].dropna().unique().tolist()
    logger.info("Batch builtin: computing sire stats for %d sires...", len(unique_sires))

    sire_win: dict[str, float | None] = {}
    if unique_sires:
        sire_rows = (
            db.query(
                Dog.sire,
                func.count(RaceEntry.id).label("total"),
                func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
            )
            .join(Dog, RaceEntry.dog_id == Dog.id)
            .join(Race, RaceEntry.race_id == Race.id)
            .filter(
                Dog.sire.in_(unique_sires),
                Race.status == "resulted",
                RaceEntry.finish_position.isnot(None),
            )
            .group_by(Dog.sire)
            .all()
        )
        for row in sire_rows:
            if row.total and row.total >= 50:
                sire_win[row.sire] = float(row.wins) / float(row.total)

    # Sire mean time at distance
    sire_time_at_dist: dict[tuple, float | None] = {}
    if unique_sires:
        sire_time_rows = (
            db.query(
                Dog.sire,
                Race.distance_m,
                func.avg(RaceEntry.finish_time).label("avg_time"),
            )
            .join(Dog, RaceEntry.dog_id == Dog.id)
            .join(Race, RaceEntry.race_id == Race.id)
            .filter(
                Dog.sire.in_(unique_sires),
                Race.status == "resulted",
                RaceEntry.finish_time.isnot(None),
            )
            .group_by(Dog.sire, Race.distance_m)
            .all()
        )
        for row in sire_time_rows:
            if row.avg_time:
                sire_time_at_dist[(row.sire, row.distance_m)] = float(row.avg_time)

    # --- 6. Bulk compute track/distance average times (for speed rating) ---
    logger.info("Batch builtin: computing track speed baselines...")
    track_avg_rows = (
        db.query(
            Race.track_id,
            Race.distance_m,
            func.count(RaceEntry.id).label("cnt"),
            func.avg(RaceEntry.adjusted_time).label("avg_time"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.status == "resulted",
            RaceEntry.adjusted_time.isnot(None),
        )
        .group_by(Race.track_id, Race.distance_m)
        .all()
    )
    track_avg_time: dict[tuple, float | None] = {}
    for row in track_avg_rows:
        if row.cnt and row.cnt >= 50 and row.avg_time:
            track_avg_time[(row.track_id, row.distance_m)] = float(row.avg_time)

    # --- 7. Assemble features per entry (pure Python, no DB) ---
    logger.info("Batch builtin: assembling features for %d entries...", len(ctx_df))
    rows: dict[int, dict[str, float | None]] = {}

    grade_map = {g: i for i, g in enumerate(GRADE_ORDER)}

    for entry_id, ctx in ctx_df.iterrows():
        features: dict[str, float | None] = {}
        dog_id = ctx["dog_id"]
        race_date = ctx["race_date"]
        trap = ctx["trap"]
        track_id = ctx["track_id"]
        distance_m = ctx["distance_m"]
        current_grade = ctx["grade"]
        trainer_name = ctx["trainer_name"]
        sire = ctx["sire"]

        # Get history for this dog, filtered to before race_date
        full_hist = dog_histories.get(dog_id, pd.DataFrame())
        if not full_hist.empty and race_date is not None:
            hist = full_hist[full_hist["race_date"] < race_date].tail(100)
        else:
            hist = pd.DataFrame()

        # 1. Trap win rate
        features["trap_win_rate_at_track"] = trap_win_rates.get(
            (track_id, distance_m, trap)
        )

        # 2. Grade movement
        if current_grade and not hist.empty and "grade" in hist.columns:
            last_grade = hist["grade"].dropna()
            if not last_grade.empty:
                last_g = str(last_grade.iloc[-1]).upper().strip()
                curr_g = str(current_grade).upper().strip()
                if curr_g in grade_map and last_g in grade_map:
                    features["grade_movement"] = float(grade_map[curr_g] - grade_map[last_g])
                else:
                    features["grade_movement"] = None
            else:
                features["grade_movement"] = None
        else:
            features["grade_movement"] = None

        # 3. Days since last
        if ctx["days_since_last"] is not None:
            features["days_since_last"] = float(ctx["days_since_last"])
        elif not hist.empty:
            last_date = hist["race_date"].max()
            if last_date and race_date:
                features["days_since_last"] = float((race_date - last_date).days)
            else:
                features["days_since_last"] = None
        else:
            features["days_since_last"] = None

        # 4. Weight change
        current_weight = ctx["weight_kg"]
        if current_weight is not None and not hist.empty:
            weights = hist["weight_kg"].dropna().tail(5)
            if not weights.empty:
                features["weight_change"] = float(current_weight - weights.mean())
            else:
                features["weight_change"] = None
        else:
            features["weight_change"] = None

        # 5. Early speed ratio
        if not hist.empty:
            recent = hist.tail(5)
            valid = recent.dropna(subset=["sectional_time", "finish_time"])
            valid = valid[valid["finish_time"] > 0]
            if not valid.empty:
                ratios = valid["sectional_time"] / valid["finish_time"]
                features["early_speed_ratio"] = float(ratios.mean())
            else:
                features["early_speed_ratio"] = None
        else:
            features["early_speed_ratio"] = None

        # 6. Is front runner
        if not hist.empty:
            recent = hist.tail(10)
            led = 0
            for _, row in recent.iterrows():
                comment = str(row.get("comment", "")).lower()
                if any(kw in comment for kw in ["led", "ld", "disp ld", "disp lead", "made all"]):
                    led += 1
            features["is_front_runner"] = float(led / max(len(recent), 1))
        else:
            features["is_front_runner"] = 0.0

        # 7. Career races
        features["career_races"] = float(len(hist)) if not hist.empty else 0.0

        # 8. Position consistency
        if not hist.empty:
            positions = hist["finish_position"].dropna().tail(10)
            features["position_consistency"] = float(positions.std()) if len(positions) >= 2 else None
        else:
            features["position_consistency"] = None

        # 9. Dog age
        birth_date = ctx["birth_date"]
        if birth_date and race_date:
            age = (race_date - birth_date).days / 365.25
            features["dog_age_years"] = age
            features["dog_age_squared"] = age ** 2
        else:
            features["dog_age_years"] = None
            features["dog_age_squared"] = None

        # 10. Pace/trap interaction
        early_speed = features.get("early_speed_ratio")
        front_runner = features.get("is_front_runner")

        if trap is not None and early_speed is not None:
            features["early_speed_x_trap"] = early_speed * trap
            features["early_speed_x_inside"] = early_speed * (1.0 if trap <= 2 else 0.0)
            features["early_speed_x_outside"] = early_speed * (1.0 if trap >= 5 else 0.0)
        else:
            features["early_speed_x_trap"] = None
            features["early_speed_x_inside"] = None
            features["early_speed_x_outside"] = None

        if trap is not None and front_runner is not None:
            features["front_runner_x_inside"] = front_runner * (1.0 if trap <= 2 else 0.0)
            features["front_runner_x_outside"] = front_runner * (1.0 if trap >= 5 else 0.0)
        else:
            features["front_runner_x_inside"] = None
            features["front_runner_x_outside"] = None

        # 11. Trainer stats
        t_stats = trainer_overall.get(trainer_name, {})
        features["trainer_win_rate"] = t_stats.get("win_rate")
        features["trainer_place_rate"] = t_stats.get("place_rate")
        features["trainer_win_rate_at_track"] = trainer_at_track.get(
            (trainer_name, track_id)
        )

        # 12. Sire stats
        features["sire_progeny_win_rate"] = sire_win.get(sire)
        features["sire_progeny_mean_time_at_dist"] = sire_time_at_dist.get(
            (sire, distance_m)
        )

        # 13. Track speed rating
        if not hist.empty and track_id and distance_m:
            same_dist = hist[hist["distance_m"] == distance_m]
            if "adjusted_time" in same_dist.columns:
                dog_times = same_dist["adjusted_time"].dropna().tail(10)
                avg = track_avg_time.get((track_id, distance_m))
                if not dog_times.empty and avg is not None:
                    features["track_speed_rating"] = float(dog_times.min()) - avg
                else:
                    features["track_speed_rating"] = None
            else:
                features["track_speed_rating"] = None
        else:
            features["track_speed_rating"] = None

        rows[entry_id] = features

    logger.info("Batch builtin: done, computed features for %d entries", len(rows))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "race_entry_id"
    return df


# Registry of built-in feature names for the UI/API
BUILTIN_FEATURE_NAMES = [
    "trap_win_rate_at_track",
    "grade_movement",
    "days_since_last",
    "weight_change",
    "early_speed_ratio",
    "is_front_runner",
    "career_races",
    "position_consistency",
    "dog_age_years",
    "dog_age_squared",
    "early_speed_x_trap",
    "early_speed_x_inside",
    "early_speed_x_outside",
    "front_runner_x_inside",
    "front_runner_x_outside",
    "trainer_win_rate",
    "trainer_place_rate",
    "trainer_win_rate_at_track",
    "sire_progeny_win_rate",
    "sire_progeny_mean_time_at_dist",
    "track_speed_rating",
]

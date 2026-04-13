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

    if "grade" not in dog_history.columns:
        return None
    grades = dog_history["grade"].dropna()
    if grades.empty:
        return None
    last_grade = grades.iloc[-1]

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


_SQLITE_VAR_LIMIT = 900  # SQLite default max is 999; leave headroom for other params


def _chunked_query(db, query_fn, ids, id_column):
    """Execute a query in chunks to avoid SQLite's variable limit.

    query_fn: callable(chunk) that returns a SQLAlchemy query with .all()
    Returns concatenated results from all chunks.
    """
    results = []
    for i in range(0, len(ids), _SQLITE_VAR_LIMIT):
        chunk = ids[i:i + _SQLITE_VAR_LIMIT]
        results.extend(query_fn(chunk))
    return results


def compute_builtin_features_batch(
    db: Session,
    entry_ids: list[int],
    heartbeat_fn=None,
) -> pd.DataFrame:
    """Compute all built-in race-context features via bulk queries.

    This replaces the per-entry approach (which issued ~8 DB queries per entry)
    with a small number of bulk queries followed by in-memory joins.

    Args:
        heartbeat_fn: Optional callable() to keep training heartbeat alive
            during long-running computation.
    """
    def _hb():
        if heartbeat_fn is not None:
            heartbeat_fn()
    if not entry_ids:
        return pd.DataFrame()

    entry_set = set(entry_ids)

    # --- 1. Bulk fetch entry + race + track + dog context ---
    logger.info("Batch builtin: fetching entry/race/dog context for %d entries...", len(entry_ids))

    def _ctx_query(chunk):
        return (
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
            .filter(RaceEntry.id.in_(chunk))
            .all()
        )

    ctx_rows = _chunked_query(db, _ctx_query, entry_ids, RaceEntry.id)

    if not ctx_rows:
        return pd.DataFrame()

    ctx_df = pd.DataFrame(ctx_rows, columns=[
        "entry_id", "trap", "dog_id", "days_since_last", "weight_kg", "race_id",
        "track_id", "distance_m", "grade", "race_date", "race_type",
        "birth_date", "trainer_name", "sire",
    ]).set_index("entry_id")

    logger.info("Batch builtin: fetched context for %d entries", len(ctx_df))
    _hb()

    # --- 2. Bulk fetch all dog histories (last 100 races per dog, before race date) ---
    # We need history per (dog_id, race_date) pair. Fetch all historical entries for
    # relevant dogs, then filter per-entry in memory.
    unique_dog_ids = ctx_df["dog_id"].unique().tolist()
    logger.info("Batch builtin: fetching histories for %d unique dogs...", len(unique_dog_ids))

    def _hist_query(chunk):
        return (
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
                RaceEntry.dog_id.in_(chunk),
                Race.status == "resulted",
            )
            .all()
        )

    hist_rows = _chunked_query(db, _hist_query, unique_dog_ids, RaceEntry.dog_id)

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
    _hb()

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
        def _trainer_overall_query(chunk):
            return (
                db.query(
                    Dog.trainer_name,
                    func.count(RaceEntry.id).label("total"),
                    func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
                    func.sum(case((RaceEntry.finish_position <= 3, 1), else_=0)).label("places"),
                )
                .join(Dog, RaceEntry.dog_id == Dog.id)
                .join(Race, RaceEntry.race_id == Race.id)
                .filter(
                    Dog.trainer_name.in_(chunk),
                    Race.status == "resulted",
                    RaceEntry.finish_position.isnot(None),
                )
                .group_by(Dog.trainer_name)
                .all()
            )

        trainer_rows = _chunked_query(db, _trainer_overall_query, unique_trainers, Dog.trainer_name)
        for row in trainer_rows:
            if row.total and row.total >= 20:
                trainer_overall[row.trainer_name] = {
                    "win_rate": float(row.wins) / float(row.total),
                    "place_rate": float(row.places) / float(row.total),
                }

    # Trainer at track
    trainer_at_track: dict[tuple, float | None] = {}
    if unique_trainers:
        def _trainer_track_query(chunk):
            return (
                db.query(
                    Dog.trainer_name,
                    Race.track_id,
                    func.count(RaceEntry.id).label("total"),
                    func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
                )
                .join(Dog, RaceEntry.dog_id == Dog.id)
                .join(Race, RaceEntry.race_id == Race.id)
                .filter(
                    Dog.trainer_name.in_(chunk),
                    Race.status == "resulted",
                    RaceEntry.finish_position.isnot(None),
                )
                .group_by(Dog.trainer_name, Race.track_id)
                .all()
            )

        trainer_track_rows = _chunked_query(db, _trainer_track_query, unique_trainers, Dog.trainer_name)
        for row in trainer_track_rows:
            if row.total and row.total >= 10:
                trainer_at_track[(row.trainer_name, row.track_id)] = float(row.wins) / float(row.total)

    # --- 5. Bulk compute sire stats ---
    unique_sires = ctx_df["sire"].dropna().unique().tolist()
    logger.info("Batch builtin: computing sire stats for %d sires...", len(unique_sires))

    sire_win: dict[str, float | None] = {}
    if unique_sires:
        def _sire_win_query(chunk):
            return (
                db.query(
                    Dog.sire,
                    func.count(RaceEntry.id).label("total"),
                    func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
                )
                .join(Dog, RaceEntry.dog_id == Dog.id)
                .join(Race, RaceEntry.race_id == Race.id)
                .filter(
                    Dog.sire.in_(chunk),
                    Race.status == "resulted",
                    RaceEntry.finish_position.isnot(None),
                )
                .group_by(Dog.sire)
                .all()
            )

        sire_rows = _chunked_query(db, _sire_win_query, unique_sires, Dog.sire)
        for row in sire_rows:
            if row.total and row.total >= 50:
                sire_win[row.sire] = float(row.wins) / float(row.total)

    # Sire mean time at distance
    sire_time_at_dist: dict[tuple, float | None] = {}
    if unique_sires:
        def _sire_time_query(chunk):
            return (
                db.query(
                    Dog.sire,
                    Race.distance_m,
                    func.avg(RaceEntry.finish_time).label("avg_time"),
                )
                .join(Dog, RaceEntry.dog_id == Dog.id)
                .join(Race, RaceEntry.race_id == Race.id)
                .filter(
                    Dog.sire.in_(chunk),
                    Race.status == "resulted",
                    RaceEntry.finish_time.isnot(None),
                )
                .group_by(Dog.sire, Race.distance_m)
                .all()
            )

        sire_time_rows = _chunked_query(db, _sire_time_query, unique_sires, Dog.sire)
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

    # --- 7. Precompute per-(dog, race_date) history aggregates ---
    # Instead of filtering history DataFrames 300k times in a Python loop,
    # we iterate each dog's sorted history ONCE and emit aggregates keyed
    # by (dog_id, race_date).  Entries then look these up via dict.
    logger.info("Batch builtin: precomputing per-entry history aggregates for %d dogs...", len(dog_histories))

    grade_map = {g: i for i, g in enumerate(GRADE_ORDER)}
    _FRONT_RUNNER_KW = {"led", "ld", "disp ld", "disp lead", "made all"}

    # Keyed by (dog_id, race_date) -> dict of history-derived features
    hist_agg: dict[tuple, dict] = {}

    # Collect all (dog_id, race_date) pairs we need
    needed_pairs: dict[int, set] = defaultdict(set)
    for entry_id, ctx in ctx_df.iterrows():
        needed_pairs[ctx["dog_id"]].add(ctx["race_date"])

    dogs_done = 0
    for dog_id, race_dates_needed in needed_pairs.items():
        full_hist = dog_histories.get(dog_id)
        if full_hist is None or full_hist.empty:
            # No history: emit defaults for all dates
            for rd in race_dates_needed:
                hist_agg[(dog_id, rd)] = {
                    "grade_movement_last": None,
                    "days_since_last_hist": None,
                    "weight_avg_5": None,
                    "early_speed_ratio": None,
                    "is_front_runner": 0.0,
                    "career_races": 0.0,
                    "position_consistency": None,
                    "track_speed_best": {},  # (track_id, distance_m) -> best_time
                }
            dogs_done += 1
            continue

        # Sort by race_date ascending (should already be sorted)
        hist_sorted = full_hist.sort_values("race_date").reset_index(drop=True)
        race_dates_sorted = hist_sorted["race_date"].values
        n_hist = len(hist_sorted)

        # Precompute columns as numpy arrays for speed
        h_grades = hist_sorted["grade"].values
        h_dates = hist_sorted["race_date"].values
        h_weights = hist_sorted["weight_kg"].values
        h_sect = hist_sorted["sectional_time"].values
        h_finish = hist_sorted["finish_time"].values
        h_comments = hist_sorted["comment"].values
        h_positions = hist_sorted["finish_position"].values
        h_adj_time = hist_sorted["adjusted_time"].values
        h_distance = hist_sorted["distance_m"].values

        # For each needed race_date, find the cutoff index via binary search
        sorted_needed = sorted(race_dates_needed)
        for rd in sorted_needed:
            # Find index of first row with race_date >= rd (everything before is history)
            cut = np.searchsorted(race_dates_sorted, rd, side="left")
            # Take last 100 entries before cut
            start = max(0, cut - 100)
            hist_len = cut - start

            if hist_len == 0:
                hist_agg[(dog_id, rd)] = {
                    "grade_movement_last": None,
                    "days_since_last_hist": None,
                    "weight_avg_5": None,
                    "early_speed_ratio": None,
                    "is_front_runner": 0.0,
                    "career_races": 0.0,
                    "position_consistency": None,
                    "track_speed_best": {},
                }
                continue

            # Slice arrays (last 100 before cutoff)
            sl = slice(start, cut)

            # Grade movement: last non-NaN grade
            grades_sl = h_grades[sl]
            non_nan_grades = [g for g in grades_sl if g is not None and pd.notna(g)]
            if non_nan_grades:
                last_g = str(non_nan_grades[-1]).upper().strip()
                grade_idx = grade_map.get(last_g)
            else:
                grade_idx = None

            # Days since last (from history)
            days_since = float((rd - h_dates[cut - 1]).days) if cut > 0 else None

            # Weight: avg of last 5 non-NaN
            weights_sl = h_weights[max(0, cut - 5):cut]
            valid_weights = [w for w in weights_sl if w is not None and not np.isnan(w)]
            weight_avg = float(np.mean(valid_weights)) if valid_weights else None

            # Early speed ratio: last 5 with valid sectional/finish
            recent_sl = slice(max(0, cut - 5), cut)
            sect_r = h_sect[recent_sl]
            fin_r = h_finish[recent_sl]
            valid_speed = [
                float(s / f) for s, f in zip(sect_r, fin_r)
                if s is not None and f is not None
                and not np.isnan(s) and not np.isnan(f) and f > 0
            ]
            early_speed = float(np.mean(valid_speed)) if valid_speed else None

            # Front runner: last 10 comments
            recent_10 = slice(max(0, cut - 10), cut)
            comments = h_comments[recent_10]
            n_recent = len(comments)
            led = 0
            for c in comments:
                if c is not None:
                    c_lower = str(c).lower()
                    if any(kw in c_lower for kw in _FRONT_RUNNER_KW):
                        led += 1
            front_runner = float(led / max(n_recent, 1))

            # Career races
            career = float(hist_len)

            # Position consistency: std of last 10 finish positions
            pos_sl = h_positions[slice(max(0, cut - 10), cut)]
            valid_pos = [p for p in pos_sl if p is not None and not np.isnan(p)]
            pos_consistency = float(np.std(valid_pos, ddof=1)) if len(valid_pos) >= 2 else None

            # Track speed: best adjusted_time per (track_id, distance_m) from last 10
            best_times: dict[tuple, float] = {}
            for i in range(max(0, cut - 10), cut):
                at = h_adj_time[i]
                if at is not None and not np.isnan(at):
                    key = (int(h_distance[i]) if not np.isnan(h_distance[i]) else 0,)
                    if key[0] not in best_times or at < best_times[key[0]]:
                        best_times[key[0]] = float(at)

            hist_agg[(dog_id, rd)] = {
                "grade_movement_last": grade_idx,
                "days_since_last_hist": days_since,
                "weight_avg_5": weight_avg,
                "early_speed_ratio": early_speed,
                "is_front_runner": front_runner,
                "career_races": career,
                "position_consistency": pos_consistency,
                "track_speed_best": best_times,
            }

        dogs_done += 1
        if dogs_done % 5000 == 0:
            logger.info("Batch builtin: history aggregates %d/%d dogs", dogs_done, len(needed_pairs))
            _hb()

    logger.info("Batch builtin: history aggregates done, assembling DataFrame...")
    _hb()

    # --- 8. Assemble features vectorized from precomputed lookups ---
    result_rows: dict[int, dict[str, float | None]] = {}

    for entry_id, ctx in ctx_df.iterrows():
        dog_id = ctx["dog_id"]
        race_date = ctx["race_date"]
        trap = ctx["trap"]
        track_id = ctx["track_id"]
        distance_m = ctx["distance_m"]
        current_grade = ctx["grade"]
        trainer_name = ctx["trainer_name"]
        sire = ctx["sire"]

        agg = hist_agg.get((dog_id, race_date), {})

        f: dict[str, float | None] = {}

        # 1. Trap win rate (lookup)
        f["trap_win_rate_at_track"] = trap_win_rates.get((track_id, distance_m, trap))

        # 2. Grade movement
        last_grade_idx = agg.get("grade_movement_last")
        if current_grade and last_grade_idx is not None:
            curr_g = str(current_grade).upper().strip()
            curr_idx = grade_map.get(curr_g)
            f["grade_movement"] = float(curr_idx - last_grade_idx) if curr_idx is not None else None
        else:
            f["grade_movement"] = None

        # 3. Days since last
        if ctx["days_since_last"] is not None:
            f["days_since_last"] = float(ctx["days_since_last"])
        else:
            f["days_since_last"] = agg.get("days_since_last_hist")

        # 4. Weight change
        weight_avg = agg.get("weight_avg_5")
        current_weight = ctx["weight_kg"]
        if current_weight is not None and weight_avg is not None:
            f["weight_change"] = float(current_weight - weight_avg)
        else:
            f["weight_change"] = None

        # 5. Early speed ratio
        early_speed = agg.get("early_speed_ratio")
        f["early_speed_ratio"] = early_speed

        # 6. Front runner
        front_runner = agg.get("is_front_runner", 0.0)
        f["is_front_runner"] = front_runner

        # 7. Career races
        f["career_races"] = agg.get("career_races", 0.0)

        # 8. Position consistency
        f["position_consistency"] = agg.get("position_consistency")

        # 9. Dog age
        birth_date = ctx["birth_date"]
        if birth_date and race_date:
            age = (race_date - birth_date).days / 365.25
            f["dog_age_years"] = age
            f["dog_age_squared"] = age ** 2
        else:
            f["dog_age_years"] = None
            f["dog_age_squared"] = None

        # 10. Pace/trap interactions
        if trap is not None and early_speed is not None:
            f["early_speed_x_trap"] = early_speed * trap
            f["early_speed_x_inside"] = early_speed * (1.0 if trap <= 2 else 0.0)
            f["early_speed_x_outside"] = early_speed * (1.0 if trap >= 5 else 0.0)
        else:
            f["early_speed_x_trap"] = None
            f["early_speed_x_inside"] = None
            f["early_speed_x_outside"] = None

        if trap is not None and front_runner is not None:
            f["front_runner_x_inside"] = front_runner * (1.0 if trap <= 2 else 0.0)
            f["front_runner_x_outside"] = front_runner * (1.0 if trap >= 5 else 0.0)
        else:
            f["front_runner_x_inside"] = None
            f["front_runner_x_outside"] = None

        # 11. Trainer stats (lookup)
        t_stats = trainer_overall.get(trainer_name, {})
        f["trainer_win_rate"] = t_stats.get("win_rate")
        f["trainer_place_rate"] = t_stats.get("place_rate")
        f["trainer_win_rate_at_track"] = trainer_at_track.get((trainer_name, track_id))

        # 12. Sire stats (lookup)
        f["sire_progeny_win_rate"] = sire_win.get(sire)
        f["sire_progeny_mean_time_at_dist"] = sire_time_at_dist.get((sire, distance_m))

        # 13. Track speed rating
        best_times = agg.get("track_speed_best", {})
        dog_best = best_times.get(distance_m)
        track_avg = track_avg_time.get((track_id, distance_m))
        if dog_best is not None and track_avg is not None:
            f["track_speed_rating"] = dog_best - track_avg
        else:
            f["track_speed_rating"] = None

        result_rows[entry_id] = f

    logger.info("Batch builtin: done, computed features for %d entries", len(result_rows))

    if not result_rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(result_rows, orient="index")
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

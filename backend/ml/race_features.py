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
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import case, func
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

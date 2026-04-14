"""
Feature computation engine for visual (JSON config) features.

Visual feature configs follow this schema:
{
    "metric": "finish_time" | "finish_position" | "weight_kg" | "sp_decimal" | "beaten_distance" | "sectional_time" | "adjusted_time",
    "aggregation": "mean" | "median" | "min" | "max" | "stdev" | "count" | "win_rate" | "place_rate" | "trend" | "ewm",
    "window": {"type": "last_n", "n": 5} | {"type": "days", "n": 90} | {"type": "all"},
    "filters": {
        "same_track": true/false,
        "same_distance": true/false,
        "same_grade": true/false,
        "same_trap": true/false
    },
    "normalize": "none" | "z_score" | "min_max"
}
"""

import logging
import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track

logger = logging.getLogger(__name__)


def get_dog_history(
    db: Session,
    dog_id: int,
    before_date: date,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Get a dog's race history as a DataFrame, only including races before the given date.
    This prevents data leakage when computing features for a target race.
    """
    rows = (
        db.query(
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
            Track.name.label("track_name"),
            Track.code.label("track_code"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Track, Race.track_id == Track.id)
        .filter(
            RaceEntry.dog_id == dog_id,
            Race.race_date < before_date,
            Race.status == "resulted",
        )
        .order_by(Race.race_date.desc())
        .limit(limit)
        .all()
    )

    if not rows:
        return pd.DataFrame()

    columns = [
        "trap", "finish_position", "finish_time", "sectional_time",
        "adjusted_time", "beaten_distance", "weight_kg", "sp_decimal",
        "starting_price", "comment", "race_date", "track_id", "distance_m",
        "grade", "race_type", "going", "going_allowance", "num_runners",
        "prize_money", "track_name", "track_code",
    ]
    df = pd.DataFrame(rows, columns=columns)
    # Sort chronologically (oldest first)
    df = df.sort_values("race_date").reset_index(drop=True)
    return df


def compute_visual_feature(
    dog_history: pd.DataFrame,
    config: dict[str, Any],
    race_context: dict[str, Any],
) -> float | None:
    """
    Compute a single visual feature value from a dog's history and config.

    race_context: {"track_id", "distance_m", "grade", "trap", "race_date", "track_code"}
    """
    if dog_history.empty:
        return None

    metric = config.get("metric", "finish_time")
    aggregation = config.get("aggregation", "mean")
    window = config.get("window", {"type": "last_n", "n": 5})
    filters = config.get("filters", {})

    # Apply filters
    df = dog_history.copy()

    if filters.get("same_track") and race_context.get("track_id"):
        df = df[df["track_id"] == race_context["track_id"]]
    if filters.get("same_distance") and race_context.get("distance_m"):
        df = df[df["distance_m"] == race_context["distance_m"]]
    if filters.get("same_grade") and race_context.get("grade"):
        df = df[df["grade"] == race_context["grade"]]
    if filters.get("same_trap") and race_context.get("trap"):
        df = df[df["trap"] == race_context["trap"]]

    if df.empty:
        return None

    # Apply window
    window_type = window.get("type", "last_n")
    window_n = window.get("n", 5)

    if window_type == "last_n":
        df = df.tail(window_n)
    elif window_type == "days":
        cutoff = race_context.get("race_date")
        if cutoff:
            from datetime import timedelta
            earliest = cutoff - timedelta(days=window_n)
            df = df[df["race_date"] >= earliest]
    # "all" means no windowing

    if df.empty:
        return None

    # Get the metric series, dropping NaN
    if metric not in df.columns:
        return None
    series = df[metric].dropna()

    if series.empty:
        return None

    # Compute aggregation
    value = _aggregate(series, aggregation, df)

    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None

    return float(value)


def _aggregate(series: pd.Series, aggregation: str, df: pd.DataFrame) -> float | None:
    """Apply an aggregation function to a series."""
    if aggregation == "mean":
        return series.mean()
    elif aggregation == "median":
        return series.median()
    elif aggregation == "min":
        return series.min()
    elif aggregation == "max":
        return series.max()
    elif aggregation == "stdev":
        return series.std() if len(series) > 1 else 0.0
    elif aggregation == "count":
        return float(len(series))
    elif aggregation == "win_rate":
        positions = df["finish_position"].dropna()
        if positions.empty:
            return None
        return float((positions == 1).sum() / len(positions))
    elif aggregation == "place_rate":
        positions = df["finish_position"].dropna()
        if positions.empty:
            return None
        return float((positions <= 3).sum() / len(positions))
    elif aggregation == "trend":
        # Linear regression slope — positive means improving (if metric is position, lower is better)
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series), dtype=float)
        y = series.values.astype(float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    elif aggregation == "ewm":
        # Exponential weighted mean: most recent race gets highest weight.
        # alpha=0.5 means each older race gets half the weight of the next newer one.
        # Series must be in chronological order (oldest first, which it is after sort).
        if len(series) < 2:
            return float(series.iloc[0])
        return float(series.ewm(alpha=0.5, adjust=True).mean().iloc[-1])
    else:
        return series.mean()


def get_race_context(db: Session, race_entry_id: int) -> dict[str, Any] | None:
    """Build race_context dict for a specific race entry."""
    row = (
        db.query(
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
        .filter(RaceEntry.id == race_entry_id)
        .first()
    )

    if not row:
        return None

    return {
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

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

import gc
import logging
from bisect import bisect_left
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
from ml.elo import EloRatings
from ml.race_comments import COMMENT_FEATURE_NAMES, parse_race_comment

logger = logging.getLogger(__name__)

# Speed figure: scale to centre around 100 with ~20 points per stdev so values
# are roughly comparable in feel to industry "speed ratings" (Beyer/Timeform).
_SPEED_FIGURE_CENTER = 100.0
_SPEED_FIGURE_STDEV_SCALE = 20.0
_SPEED_FIGURE_MIN_BUCKET = 30  # require this many runs in a (track, distance)
                               # bucket before trusting its baseline

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


class _DatePrefix:
    """Per-key, date-ordered prefix sums answering "totals strictly before
    date D" (optionally within a trailing day window) in O(log n).

    This is the time-awareness backbone for the batch aggregate features:
    without it, training rows saw aggregates computed over ALL resulted
    races — including the row's own race and every future race — which
    leaked outcomes into the training matrix and inflated CV/test scores.
    Lookups here use only data strictly before the target race date, which
    is exactly what is reproducible at prediction time.
    """

    def __init__(self):
        self._dates: dict = {}
        self._cums: dict = {}

    @classmethod
    def from_rows(cls, rows, n_values: int) -> "_DatePrefix":
        """rows: iterable of (key, race_date, v1, ..., vn)."""
        grouped: dict = defaultdict(list)
        for row in rows:
            grouped[row[0]].append(row[1:])
        obj = cls()
        for key, items in grouped.items():
            items.sort(key=lambda r: r[0])
            dates = [r[0] for r in items]
            cums = []
            running = [0.0] * n_values
            for r in items:
                for i in range(n_values):
                    running[i] += float(r[1 + i] or 0)
                cums.append(tuple(running))
            obj._dates[key] = dates
            obj._cums[key] = cums
        return obj

    def before(self, key, target_date) -> tuple | None:
        """Cumulative totals over dates strictly before target_date."""
        dates = self._dates.get(key)
        if not dates or target_date is None:
            return None
        idx = bisect_left(dates, target_date)
        if idx == 0:
            return None
        return self._cums[key][idx - 1]

    def window(self, key, target_date, days: int) -> tuple | None:
        """Totals over dates in [target_date - days, target_date)."""
        dates = self._dates.get(key)
        if not dates or target_date is None:
            return None
        hi = bisect_left(dates, target_date)
        if hi == 0:
            return None
        lo = bisect_left(dates, target_date - timedelta(days=days))
        if lo >= hi:
            return None
        upper = self._cums[key][hi - 1]
        if lo == 0:
            return upper
        lower = self._cums[key][lo - 1]
        return tuple(u - l for u, l in zip(upper, lower))


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
                RaceEntry.wide_runner,
                RaceEntry.race_id,
                Race.track_id,
                Race.distance_m,
                Race.grade,
                Race.going,
                Race.num_runners,
                Race.race_date,
                Race.race_time,
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
        "entry_id", "trap", "dog_id", "days_since_last", "weight_kg",
        "wide_runner", "race_id",
        "track_id", "distance_m", "grade", "going", "num_runners",
        "race_date", "race_time", "race_type",
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
    del hist_rows  # free raw tuples — now redundant

    # Index by dog for fast lookup
    dog_histories: dict[int, pd.DataFrame] = {}
    n_hist_rows = len(all_hist_df)
    if not all_hist_df.empty:
        for dog_id, group in all_hist_df.groupby("dog_id"):
            dog_histories[dog_id] = group.sort_values("race_date").reset_index(drop=True)

    del all_hist_df  # free full DataFrame — now split into per-dog dicts
    gc.collect()

    logger.info("Batch builtin: loaded %d history rows for %d dogs", n_hist_rows, len(dog_histories))
    _hb()

    # --- 3. Bulk compute trap win rates (per track/distance/trap combo) ---
    # Grouped by race_date and folded into prefix sums so every lookup is
    # "stats strictly before this entry's race date" — anything else leaks
    # the entry's own result (and the future) into the training matrix.
    unique_combos = ctx_df[["track_id", "distance_m", "trap"]].drop_duplicates()
    logger.info("Batch builtin: computing trap win rates for %d combos...", len(unique_combos))

    trap_stats = (
        db.query(
            Race.track_id,
            Race.distance_m,
            RaceEntry.trap,
            Race.race_date,
            func.count(RaceEntry.id).label("total"),
            func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.status == "resulted",
            RaceEntry.finish_position.isnot(None),
        )
        .group_by(Race.track_id, Race.distance_m, RaceEntry.trap, Race.race_date)
        .all()
    )
    trap_prefix = _DatePrefix.from_rows(
        (
            ((r.track_id, r.distance_m, r.trap), r.race_date, r.total, r.wins)
            for r in trap_stats
        ),
        n_values=2,
    )
    del trap_stats

    def _trap_win_rate_asof(track_id, distance_m, trap, as_of, min_races: int = 30):
        totals = trap_prefix.before((track_id, distance_m, trap), as_of)
        if totals is None or totals[0] < min_races:
            return None
        return totals[1] / totals[0]

    # --- 3b. Going-conditional trap win rates ---
    # Going materially affects trap bias: heavy going churns the inside rail,
    # fast going amplifies rail shortcuts.  Compute per (track, distance,
    # going, trap) with a higher min-sample bar since buckets are sparser.
    trap_going_stats = (
        db.query(
            Race.track_id,
            Race.distance_m,
            Race.going,
            RaceEntry.trap,
            Race.race_date,
            func.count(RaceEntry.id).label("total"),
            func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.status == "resulted",
            RaceEntry.finish_position.isnot(None),
            Race.going.isnot(None),
        )
        .group_by(Race.track_id, Race.distance_m, Race.going, RaceEntry.trap, Race.race_date)
        .all()
    )
    trap_going_prefix = _DatePrefix.from_rows(
        (
            ((r.track_id, r.distance_m, r.going, r.trap), r.race_date, r.total, r.wins)
            for r in trap_going_stats
        ),
        n_values=2,
    )
    del trap_going_stats

    def _trap_going_rate_asof(track_id, distance_m, going, trap, as_of, min_races: int = 20):
        totals = trap_going_prefix.before((track_id, distance_m, going, trap), as_of)
        if totals is None or totals[0] < min_races:
            return None
        return totals[1] / totals[0]

    # --- 4. Bulk compute trainer stats (win/place rate in last 90 days) ---
    # Time-aware: a 90-day rolling window ending strictly before each entry's
    # race date. This both removes target leakage (the old version aggregated
    # over all time, including the entry's own race and the future) and
    # restores the recency window the single-entry path (_trainer_stats)
    # documents — the batch path had silently dropped it.
    unique_trainers = ctx_df["trainer_name"].dropna().unique().tolist()
    logger.info("Batch builtin: computing trainer stats for %d trainers...", len(unique_trainers))

    _TRAINER_WINDOW_DAYS = 90

    trainer_prefix = _DatePrefix.from_rows((), n_values=3)
    trainer_track_prefix = _DatePrefix.from_rows((), n_values=2)
    if unique_trainers:
        def _trainer_overall_query(chunk):
            return (
                db.query(
                    Dog.trainer_name,
                    Race.race_date,
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
                .group_by(Dog.trainer_name, Race.race_date)
                .all()
            )

        trainer_rows = _chunked_query(db, _trainer_overall_query, unique_trainers, Dog.trainer_name)
        trainer_prefix = _DatePrefix.from_rows(
            ((r.trainer_name, r.race_date, r.total, r.wins, r.places) for r in trainer_rows),
            n_values=3,
        )
        del trainer_rows

        def _trainer_track_query(chunk):
            return (
                db.query(
                    Dog.trainer_name,
                    Race.track_id,
                    Race.race_date,
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
                .group_by(Dog.trainer_name, Race.track_id, Race.race_date)
                .all()
            )

        trainer_track_rows = _chunked_query(db, _trainer_track_query, unique_trainers, Dog.trainer_name)
        trainer_track_prefix = _DatePrefix.from_rows(
            (
                ((r.trainer_name, r.track_id), r.race_date, r.total, r.wins)
                for r in trainer_track_rows
            ),
            n_values=2,
        )
        del trainer_track_rows

    def _trainer_stats_asof(trainer_name, track_id, as_of):
        out = {"win_rate": None, "place_rate": None, "at_track": None}
        if not trainer_name:
            return out
        totals = trainer_prefix.window(trainer_name, as_of, _TRAINER_WINDOW_DAYS)
        if totals is not None and totals[0] >= 20:
            out["win_rate"] = totals[1] / totals[0]
            out["place_rate"] = totals[2] / totals[0]
        if track_id is not None:
            t = trainer_track_prefix.window((trainer_name, track_id), as_of, _TRAINER_WINDOW_DAYS)
            if t is not None and t[0] >= 10:
                out["at_track"] = t[1] / t[0]
        return out

    # --- 5. Bulk compute sire stats (time-aware prefix sums) ---
    unique_sires = ctx_df["sire"].dropna().unique().tolist()
    logger.info("Batch builtin: computing sire stats for %d sires...", len(unique_sires))

    sire_prefix = _DatePrefix.from_rows((), n_values=2)
    sire_time_prefix = _DatePrefix.from_rows((), n_values=2)
    if unique_sires:
        def _sire_win_query(chunk):
            return (
                db.query(
                    Dog.sire,
                    Race.race_date,
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
                .group_by(Dog.sire, Race.race_date)
                .all()
            )

        sire_rows = _chunked_query(db, _sire_win_query, unique_sires, Dog.sire)
        sire_prefix = _DatePrefix.from_rows(
            ((r.sire, r.race_date, r.total, r.wins) for r in sire_rows),
            n_values=2,
        )
        del sire_rows

        # Sire mean time at distance: store (count, time_sum) so the mean as
        # of any date is derivable from the prefix.
        def _sire_time_query(chunk):
            return (
                db.query(
                    Dog.sire,
                    Race.distance_m,
                    Race.race_date,
                    func.count(RaceEntry.id).label("cnt"),
                    func.sum(RaceEntry.finish_time).label("time_sum"),
                )
                .join(Dog, RaceEntry.dog_id == Dog.id)
                .join(Race, RaceEntry.race_id == Race.id)
                .filter(
                    Dog.sire.in_(chunk),
                    Race.status == "resulted",
                    RaceEntry.finish_time.isnot(None),
                )
                .group_by(Dog.sire, Race.distance_m, Race.race_date)
                .all()
            )

        sire_time_rows = _chunked_query(db, _sire_time_query, unique_sires, Dog.sire)
        sire_time_prefix = _DatePrefix.from_rows(
            (
                ((r.sire, r.distance_m), r.race_date, r.cnt, r.time_sum)
                for r in sire_time_rows
            ),
            n_values=2,
        )
        del sire_time_rows

    def _sire_stats_asof(sire, distance_m, as_of):
        out = {"win_rate": None, "mean_time": None}
        if not sire:
            return out
        totals = sire_prefix.before(sire, as_of)
        if totals is not None and totals[0] >= 50:
            out["win_rate"] = totals[1] / totals[0]
        if distance_m is not None:
            t = sire_time_prefix.before((sire, distance_m), as_of)
            if t is not None and t[0] > 0 and t[1]:
                out["mean_time"] = t[1] / t[0]
        return out

    # --- 6. Bulk compute track/distance average times (for speed rating) ---
    # Time-aware with the 180-day window the single-entry path
    # (_track_speed_rating) documents; the batch path had dropped both the
    # window and the date cutoff.
    logger.info("Batch builtin: computing track speed baselines...")
    _TRACK_TIME_WINDOW_DAYS = 180
    track_time_rows = (
        db.query(
            Race.track_id,
            Race.distance_m,
            Race.race_date,
            func.count(RaceEntry.id).label("cnt"),
            func.sum(RaceEntry.adjusted_time).label("time_sum"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.status == "resulted",
            RaceEntry.adjusted_time.isnot(None),
        )
        .group_by(Race.track_id, Race.distance_m, Race.race_date)
        .all()
    )
    track_time_prefix = _DatePrefix.from_rows(
        (
            ((r.track_id, r.distance_m), r.race_date, r.cnt, r.time_sum)
            for r in track_time_rows
        ),
        n_values=2,
    )

    def _track_avg_time_asof(track_id, distance_m, as_of, min_races: int = 50):
        totals = track_time_prefix.window(
            (track_id, distance_m), as_of, _TRACK_TIME_WINDOW_DAYS
        )
        if totals is None or totals[0] < min_races or not totals[1]:
            return None
        return totals[1] / totals[0]

    # --- 6b. Speed-figure baselines: mean & stdev of adjusted_time per
    # (track, distance) bucket, as of a given date.  Used to normalise every
    # historical run into a Beyer-style speed figure comparable across
    # tracks/distances.  Stdev is derived from prefix sums of (count, sum,
    # sum-of-squares), so each history row is normalised against only the
    # data that existed strictly before that row's own race date — the old
    # all-time baselines let future track conditions leak into the figures.
    logger.info("Batch builtin: computing speed-figure baselines...")
    sf_rows = (
        db.query(
            Race.track_id,
            Race.distance_m,
            Race.race_date,
            func.count(RaceEntry.id).label("cnt"),
            func.sum(RaceEntry.adjusted_time).label("time_sum"),
            func.sum(RaceEntry.adjusted_time * RaceEntry.adjusted_time).label("time_sumsq"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.status == "resulted",
            RaceEntry.adjusted_time.isnot(None),
        )
        .group_by(Race.track_id, Race.distance_m, Race.race_date)
        .all()
    )
    sf_prefix = _DatePrefix.from_rows(
        (
            ((r.track_id, r.distance_m), r.race_date, r.cnt, r.time_sum, r.time_sumsq)
            for r in sf_rows
        ),
        n_values=3,
    )
    del track_time_rows, sf_rows
    gc.collect()

    def _speed_figure(
        adj_time: float, track_id: int, distance_m: int, as_of
    ) -> float | None:
        totals = sf_prefix.before((track_id, distance_m), as_of)
        if totals is None:
            return None
        n, s, ss = totals
        if n < _SPEED_FIGURE_MIN_BUCKET:
            return None
        mean = s / n
        var = ss / n - mean * mean
        if var <= 1e-12:
            return None
        std = var ** 0.5
        if std <= 1e-6:
            return None
        return (
            _SPEED_FIGURE_CENTER
            + _SPEED_FIGURE_STDEV_SCALE * (mean - adj_time) / std
        )

    # --- 7. Precompute per-(dog, race_date) history aggregates ---
    # Instead of filtering history DataFrames 300k times in a Python loop,
    # we iterate each dog's sorted history ONCE and emit aggregates keyed
    # by (dog_id, race_date).  Entries then look these up via dict.
    logger.info("Batch builtin: precomputing per-entry history aggregates for %d dogs...", len(dog_histories))

    grade_map = {g: i for i, g in enumerate(GRADE_ORDER)}
    _FRONT_RUNNER_KW = {"led", "ld", "disp ld", "disp lead", "made all"}
    _TROUBLE_KW = {
        "ck", "bmp", "crd", "fell", "hampered", "baulked", "stumbled",
        "crowded", "checked", "bumped",
    }

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
                    "speed_figure_best_last10": None,
                    "speed_figure_mean_last5": None,
                    "speed_figure_ewm_last10": None,
                    "speed_figure_trend_last5": None,
                    "career_peak_speed_figure": None,
                    "last_trap": None,
                    "win_rate_last5": None,
                    "last_position": None,
                    "median_career_grade_idx": None,
                    "finishing_speed_ratio": None,
                    "finishing_speed_trend": None,
                    "finishing_speed_ewm10": None,
                    "last_distance": None,
                    "prior_tracks": set(),
                    "prior_distances": set(),
                    "prior_grades": set(),
                    "second_after_layoff": 0.0,
                    "races_last_14_days": 0.0,
                    "races_last_60_days": 0.0,
                    "workload_trend": 0.0,
                    "weight_3race_stdev": None,
                    "career_avg_weight": None,
                    "clean_run_win_rate_last10": None,
                    "clean_run_mean_position_last10": None,
                    "trouble_run_mean_position_last10": None,
                    "trouble_recovery_ratio_last10": None,
                    "clean_run_count_last10": 0.0,
                    "trouble_run_count_last10": 0.0,
                    "comment_rates": {name: None for name in COMMENT_FEATURE_NAMES},
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
        h_track = hist_sorted["track_id"].values
        h_trap = hist_sorted["trap"].values

        # Precompute speed figures for the whole sorted history once.
        # NaN where the bucket has no baseline or adjusted_time is missing.
        h_sf = np.full(n_hist, np.nan, dtype=float)
        for i in range(n_hist):
            at = h_adj_time[i]
            if at is None or (isinstance(at, float) and np.isnan(at)):
                continue
            tid_raw = h_track[i]
            dst_raw = h_distance[i]
            if tid_raw is None or dst_raw is None:
                continue
            try:
                tid = int(tid_raw)
                dst = int(dst_raw)
            except (TypeError, ValueError):
                continue
            # Baselines as of the run's own date: reproducible at serve time
            # (history rows always predate the target race) and immune to
            # future track-condition shifts contaminating the figure.
            sf = _speed_figure(float(at), tid, dst, as_of=h_dates[i])
            if sf is not None:
                h_sf[i] = sf

        # Precompute parsed race comments once per history row.  Each parsed
        # result is a dict of flags; we aggregate into rates over last-10
        # windows during the per-date loop below.
        h_parsed = [parse_race_comment(c) for c in h_comments]

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
                    "speed_figure_best_last10": None,
                    "speed_figure_mean_last5": None,
                    "speed_figure_ewm_last10": None,
                    "speed_figure_trend_last5": None,
                    "career_peak_speed_figure": None,
                    "last_trap": None,
                    "win_rate_last5": None,
                    "last_position": None,
                    "median_career_grade_idx": None,
                    "finishing_speed_ratio": None,
                    "finishing_speed_trend": None,
                    "finishing_speed_ewm10": None,
                    "last_distance": None,
                    "prior_tracks": set(),
                    "prior_distances": set(),
                    "prior_grades": set(),
                    "second_after_layoff": 0.0,
                    "races_last_14_days": 0.0,
                    "races_last_60_days": 0.0,
                    "workload_trend": 0.0,
                    "weight_3race_stdev": None,
                    "career_avg_weight": None,
                    "clean_run_win_rate_last10": None,
                    "clean_run_mean_position_last10": None,
                    "trouble_run_mean_position_last10": None,
                    "trouble_recovery_ratio_last10": None,
                    "clean_run_count_last10": 0.0,
                    "trouble_run_count_last10": 0.0,
                    "comment_rates": {name: None for name in COMMENT_FEATURE_NAMES},
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

            # Last trap (used for trap_switch below)
            last_trap = h_trap[cut - 1] if cut > 0 else None
            if last_trap is not None and isinstance(last_trap, float) and np.isnan(last_trap):
                last_trap = None

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

            # Tier 8: finish-stretch / stamina profile from the same last-5 slice.
            # finishing_speed_ratio = (finish_time - sectional_time) / finish_time
            # Low = front-loaded (early burst, faded late), high = strong finish.
            finishing_ratios = [
                float((f - s) / f) for s, f in zip(sect_r, fin_r)
                if s is not None and f is not None
                and not np.isnan(s) and not np.isnan(f) and f > 0
            ]
            finishing_speed_ratio = (
                float(np.mean(finishing_ratios)) if finishing_ratios else None
            )
            # Trend: slope of the last-5 ratios (positive = becoming stronger at finish)
            finishing_trend = None
            if len(finishing_ratios) >= 3:
                xs = np.arange(len(finishing_ratios), dtype=float)
                finishing_trend = float(
                    np.polyfit(xs, np.asarray(finishing_ratios, dtype=float), 1)[0]
                )

            # EWM over last 10 for a smoothed view
            recent10_sect = h_sect[slice(max(0, cut - 10), cut)]
            recent10_fin = h_finish[slice(max(0, cut - 10), cut)]
            finishing_ratios_10 = [
                float((f - s) / f) for s, f in zip(recent10_sect, recent10_fin)
                if s is not None and f is not None
                and not np.isnan(s) and not np.isnan(f) and f > 0
            ]
            finishing_ewm = None
            if finishing_ratios_10:
                weight = 0.0
                acc = 0.0
                w = 1.0
                for v in finishing_ratios_10:
                    acc += w * v
                    weight += w
                    w *= 0.5
                finishing_ewm = float(acc / weight) if weight > 0 else None

            # Last race distance — used downstream for distance_change_from_last
            last_distance = (
                int(h_distance[cut - 1])
                if cut > 0 and h_distance[cut - 1] is not None
                and not np.isnan(h_distance[cut - 1])
                else None
            )

            # Tier 9: sets of tracks / distances / grades raced prior to this date
            # (used to detect "first time at X" signals at the assembly step)
            prior_tracks = set()
            for v in h_track[:cut]:
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    prior_tracks.add(int(v))
            prior_distances = set()
            for v in h_distance[:cut]:
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    prior_distances.add(int(v))
            prior_grades = set()
            for v in h_grades[:cut]:
                if v is not None and pd.notna(v):
                    prior_grades.add(str(v).upper().strip())

            # Was the previous race itself preceded by a long layoff? If so, the
            # current race is the "second after layoff" — often a meaningful
            # form indicator.  We check gap between race (cut-2) and (cut-1).
            second_after_layoff = 0.0
            if cut >= 2:
                prev_gap = (h_dates[cut - 1] - h_dates[cut - 2]).days
                if prev_gap >= 29:
                    second_after_layoff = 1.0

            # Tier 10: workload / fitness-cycle counts over 14 and 60 days
            cutoff_14 = rd - timedelta(days=14)
            cutoff_60 = rd - timedelta(days=60)
            cutoff_90 = rd - timedelta(days=90)
            races_14 = 0
            races_60 = 0
            races_90 = 0
            for i in range(cut - 1, -1, -1):
                d = h_dates[i]
                if d is None:
                    continue
                if d >= cutoff_14:
                    races_14 += 1
                if d >= cutoff_60:
                    races_60 += 1
                if d >= cutoff_90:
                    races_90 += 1
                else:
                    break

            # Workload trend: number of races in the first 45 days vs the last
            # 45 days of the 90-day window — positive = workload increasing.
            mid_90 = rd - timedelta(days=45)
            races_first_45 = 0
            races_last_45 = 0
            for i in range(cut - 1, -1, -1):
                d = h_dates[i]
                if d is None:
                    break
                if d < cutoff_90:
                    break
                if d >= mid_90:
                    races_last_45 += 1
                else:
                    races_first_45 += 1
            workload_trend = float(races_last_45 - races_first_45)

            # Weight dynamics: stdev of last-3 weight differences, and
            # current weight as pct of career average.
            weights_arr = h_weights[:cut]
            valid_weights_career = [
                float(w) for w in weights_arr
                if w is not None and not (isinstance(w, float) and np.isnan(w))
            ]
            career_avg_weight = (
                float(np.mean(valid_weights_career)) if valid_weights_career else None
            )
            recent_w3 = [
                float(w) for w in weights_arr[-3:]
                if w is not None and not (isinstance(w, float) and np.isnan(w))
            ]
            weight_3race_stdev = (
                float(np.std(recent_w3, ddof=1)) if len(recent_w3) >= 2 else None
            )

            # Tier 11: trouble-adjusted performance.  Separate the last 10
            # races into "clean" vs "trouble" runs and compute rates on each.
            last10_slice = slice(max(0, cut - 10), cut)
            last10_comments = h_comments[last10_slice]
            last10_positions = h_positions[last10_slice]
            clean_positions: list[float] = []
            trouble_positions: list[float] = []
            for c, p in zip(last10_comments, last10_positions):
                if p is None or (isinstance(p, float) and np.isnan(p)):
                    continue
                c_lower = "" if c is None else str(c).lower()
                has_trouble = any(kw in c_lower for kw in _TROUBLE_KW)
                if has_trouble:
                    trouble_positions.append(float(p))
                else:
                    clean_positions.append(float(p))

            clean_win_rate = (
                float(sum(1 for p in clean_positions if p == 1) / len(clean_positions))
                if clean_positions else None
            )
            clean_mean_position = (
                float(np.mean(clean_positions)) if clean_positions else None
            )
            trouble_mean_position = (
                float(np.mean(trouble_positions)) if trouble_positions else None
            )
            # Recovery ratio: out of trouble runs, how close to the podium did
            # the dog finish on average?  Lower = better recovery.
            # Clamp at 1 (fell / disqualified etc. often get high positions).
            trouble_recovery = (
                float(np.mean([p - 1 for p in trouble_positions]))
                if trouble_positions else None
            )

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

            # Tier 7: class / grade dynamics
            # Win rate over the last 5 finish positions and last finish
            pos_last5 = h_positions[slice(max(0, cut - 5), cut)]
            valid_pos5 = [p for p in pos_last5 if p is not None and not np.isnan(p)]
            win_rate_last5 = (
                float(sum(1 for p in valid_pos5 if p == 1) / len(valid_pos5))
                if valid_pos5 else None
            )
            last_position = int(valid_pos[-1]) if valid_pos else None

            # Career grade indices (converted via grade_map) — lower = higher class
            grade_idx_career = [
                grade_map[str(g).upper().strip()]
                for g in h_grades[:cut]
                if g is not None and pd.notna(g)
                and str(g).upper().strip() in grade_map
            ]
            median_career_grade_idx = (
                float(np.median(grade_idx_career)) if grade_idx_career else None
            )

            # Track speed: best adjusted_time per (track_id, distance_m) from last 10
            best_times: dict[tuple, float] = {}
            for i in range(max(0, cut - 10), cut):
                at = h_adj_time[i]
                if at is not None and not np.isnan(at):
                    key = (int(h_distance[i]) if not np.isnan(h_distance[i]) else 0,)
                    if key[0] not in best_times or at < best_times[key[0]]:
                        best_times[key[0]] = float(at)

            # Speed-figure aggregates from precomputed h_sf series
            sf_window10 = h_sf[slice(max(0, cut - 10), cut)]
            sf_window5 = h_sf[slice(max(0, cut - 5), cut)]
            sf_career = h_sf[:cut]

            sf10_valid = sf_window10[~np.isnan(sf_window10)]
            sf5_valid = sf_window5[~np.isnan(sf_window5)]
            sf_career_valid = sf_career[~np.isnan(sf_career)]

            sf_best10 = float(sf10_valid.max()) if sf10_valid.size else None
            sf_mean5 = float(sf5_valid.mean()) if sf5_valid.size else None
            sf_peak = float(sf_career_valid.max()) if sf_career_valid.size else None

            # EWM (alpha=0.5) over last 10, walking forward through valid points
            sf_ewm10: float | None = None
            if sf10_valid.size:
                weight = 0.0
                acc = 0.0
                w = 1.0
                for v in sf10_valid:
                    acc += w * v
                    weight += w
                    w *= 0.5
                sf_ewm10 = float(acc / weight) if weight > 0 else None

            # Trend (slope of speed figure over last 5)
            sf_trend: float | None = None
            if sf5_valid.size >= 3:
                xs = np.arange(sf5_valid.size, dtype=float)
                # Higher SF is better, so a positive slope = improving
                slope = float(np.polyfit(xs, sf5_valid.astype(float), 1)[0])
                sf_trend = slope

            # Comment-derived rates over the last 10 history rows.
            # h_parsed was precomputed once per history; slice here.
            parsed_last10 = h_parsed[slice(max(0, cut - 10), cut)]
            pos_last10_for_cmt = h_positions[slice(max(0, cut - 10), cut)]
            cmt_n = len(parsed_last10)
            if cmt_n > 0:
                def _rate(key: str) -> float:
                    return float(sum(1 for p in parsed_last10 if p.get(key))) / cmt_n

                def _bend_rate(field: str, bend: int) -> float:
                    return float(
                        sum(1 for p in parsed_last10 if bend in p.get(field, set()))
                    ) / cmt_n

                clear_win_hits = 0
                for p, pos in zip(parsed_last10, pos_last10_for_cmt):
                    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
                        continue
                    if p.get("cleared_field") and int(pos) == 1:
                        clear_win_hits += 1
                clear_win_rate = float(clear_win_hits) / cmt_n

                comment_rates = {
                    "running_style_ep_rate_last10": _rate("is_early_pace"),
                    "running_style_mp_rate_last10": _rate("is_mid_pace"),
                    "running_style_lp_rate_last10": _rate("is_late_pace"),
                    "quick_away_rate_last10": _rate("quick_away"),
                    "slow_away_rate_last10": _rate("slow_away"),
                    "awkward_start_rate_last10": _rate("awkward_start"),
                    "led_at_bend1_rate_last10": _bend_rate("led_bends", 1),
                    "led_at_bend2_rate_last10": _bend_rate("led_bends", 2),
                    "led_at_bend3_rate_last10": _bend_rate("led_bends", 3),
                    "led_at_bend4_rate_last10": _bend_rate("led_bends", 4),
                    "disputed_lead_rate_last10": _rate("disputed_lead"),
                    "finish_well_rate_last10": _rate("finished_well"),
                    "faded_rate_last10": _rate("faded"),
                    "trouble_bend1_rate_last10": _bend_rate("trouble_bends", 1),
                    "trouble_bend2_rate_last10": _bend_rate("trouble_bends", 2),
                    "trouble_bend3_rate_last10": _bend_rate("trouble_bends", 3),
                    "trouble_bend4_rate_last10": _bend_rate("trouble_bends", 4),
                    "railed_rate_last10": _rate("railed"),
                    "ran_wide_rate_last10": _rate("ran_wide"),
                    "clear_win_rate_last10": clear_win_rate,
                }
            else:
                comment_rates = {name: None for name in COMMENT_FEATURE_NAMES}

            hist_agg[(dog_id, rd)] = {
                "grade_movement_last": grade_idx,
                "days_since_last_hist": days_since,
                "weight_avg_5": weight_avg,
                "early_speed_ratio": early_speed,
                "is_front_runner": front_runner,
                "career_races": career,
                "position_consistency": pos_consistency,
                "track_speed_best": best_times,
                "speed_figure_best_last10": sf_best10,
                "speed_figure_mean_last5": sf_mean5,
                "speed_figure_ewm_last10": sf_ewm10,
                "speed_figure_trend_last5": sf_trend,
                "career_peak_speed_figure": sf_peak,
                "last_trap": last_trap,
                "win_rate_last5": win_rate_last5,
                "last_position": last_position,
                "median_career_grade_idx": median_career_grade_idx,
                "finishing_speed_ratio": finishing_speed_ratio,
                "finishing_speed_trend": finishing_trend,
                "finishing_speed_ewm10": finishing_ewm,
                "last_distance": last_distance,
                "prior_tracks": prior_tracks,
                "prior_distances": prior_distances,
                "prior_grades": prior_grades,
                "second_after_layoff": second_after_layoff,
                "races_last_14_days": float(races_14),
                "races_last_60_days": float(races_60),
                "workload_trend": workload_trend,
                "weight_3race_stdev": weight_3race_stdev,
                "career_avg_weight": career_avg_weight,
                "clean_run_win_rate_last10": clean_win_rate,
                "clean_run_mean_position_last10": clean_mean_position,
                "trouble_run_mean_position_last10": trouble_mean_position,
                "trouble_recovery_ratio_last10": trouble_recovery,
                "clean_run_count_last10": float(len(clean_positions)),
                "trouble_run_count_last10": float(len(trouble_positions)),
                "comment_rates": comment_rates,
            }

        dogs_done += 1
        if dogs_done % 5000 == 0:
            logger.info("Batch builtin: history aggregates %d/%d dogs", dogs_done, len(needed_pairs))
            _hb()

    del dog_histories, needed_pairs  # free per-dog DataFrames
    gc.collect()

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
        current_going = ctx["going"]
        num_runners = ctx["num_runners"]
        wide_runner = ctx["wide_runner"]
        trainer_name = ctx["trainer_name"]
        sire = ctx["sire"]

        agg = hist_agg.get((dog_id, race_date), {})

        f: dict[str, float | None] = {}

        # 1. Trap win rate (as of this entry's race date)
        f["trap_win_rate_at_track"] = _trap_win_rate_asof(track_id, distance_m, trap, race_date)

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

        # 11. Trainer stats (90-day window ending before this race date)
        t_stats = _trainer_stats_asof(trainer_name, track_id, race_date)
        f["trainer_win_rate"] = t_stats["win_rate"]
        f["trainer_place_rate"] = t_stats["place_rate"]
        f["trainer_win_rate_at_track"] = t_stats["at_track"]

        # 12. Sire stats (as of this race date)
        s_stats = _sire_stats_asof(sire, distance_m, race_date)
        f["sire_progeny_win_rate"] = s_stats["win_rate"]
        f["sire_progeny_mean_time_at_dist"] = s_stats["mean_time"]

        # 13. Track speed rating (180-day window ending before this race date)
        best_times = agg.get("track_speed_best", {})
        dog_best = best_times.get(distance_m)
        track_avg = _track_avg_time_asof(track_id, distance_m, race_date)
        if dog_best is not None and track_avg is not None:
            f["track_speed_rating"] = dog_best - track_avg
        else:
            f["track_speed_rating"] = None

        # 14. Speed figure aggregates (Beyer-style normalisation)
        f["speed_figure_best_last10"] = agg.get("speed_figure_best_last10")
        f["speed_figure_mean_last5"] = agg.get("speed_figure_mean_last5")
        f["speed_figure_ewm_last10"] = agg.get("speed_figure_ewm_last10")
        f["speed_figure_trend_last5"] = agg.get("speed_figure_trend_last5")
        career_peak = agg.get("career_peak_speed_figure")
        f["career_peak_speed_figure"] = career_peak
        recent = agg.get("speed_figure_ewm_last10")
        if recent is not None and career_peak is not None and career_peak != 0:
            f["recent_vs_peak_speed_figure"] = recent - career_peak
        else:
            f["recent_vs_peak_speed_figure"] = None

        # 15. Tier 6 — draw x bias x running-style interactions
        # Expected trap win rate for a neutral dog (= 1 / field size).  Any
        # deviation from this is a track bias signal that can be exploited.
        expected_rate = None
        if num_runners and num_runners > 0:
            expected_rate = 1.0 / float(num_runners)

        trap_rate = f["trap_win_rate_at_track"]
        if trap_rate is not None and expected_rate is not None:
            f["trap_bias_deviation"] = trap_rate - expected_rate
        else:
            f["trap_bias_deviation"] = None

        # Going-conditional trap bias (may be None if the bucket is too small)
        if current_going is not None:
            going_rate = _trap_going_rate_asof(
                track_id, distance_m, current_going, trap, race_date
            )
            if going_rate is not None and expected_rate is not None:
                f["trap_bias_deviation_going"] = going_rate - expected_rate
            else:
                f["trap_bias_deviation_going"] = None
        else:
            f["trap_bias_deviation_going"] = None

        # Running-style × trap-bias interaction
        front_runner = f.get("is_front_runner", 0.0) or 0.0
        bias_dev = f["trap_bias_deviation"]
        if bias_dev is not None:
            f["trap_bias_x_front_runner"] = bias_dev * float(front_runner)
        else:
            f["trap_bias_x_front_runner"] = None

        # Wide runner flag × outside trap indicator
        if trap is not None:
            is_outside = 1.0 if trap >= 5 else 0.0
            is_inside = 1.0 if trap <= 2 else 0.0
            wr_flag = 1.0 if bool(wide_runner) else 0.0
            f["wide_runner_x_outside"] = wr_flag * is_outside
            f["wide_runner_x_inside"] = wr_flag * is_inside
        else:
            f["wide_runner_x_outside"] = None
            f["wide_runner_x_inside"] = None

        # Trap switch — absolute change from last race's trap
        last_trap = agg.get("last_trap")
        if trap is not None and last_trap is not None:
            try:
                f["trap_switch"] = float(abs(int(trap) - int(last_trap)))
            except (TypeError, ValueError):
                f["trap_switch"] = None
        else:
            f["trap_switch"] = None

        # 16. Tier 7 — class/grade dynamics
        # grade_movement is set earlier (index delta vs last grade).  Combine
        # with recent form to flag "class drop in form" (drop + recent wins
        # = classic overlay) and "class rise on win" (rise + just won = often
        # overbet by the market).
        grade_mv = f.get("grade_movement")
        win_rate_5 = agg.get("win_rate_last5")
        last_pos = agg.get("last_position")
        if grade_mv is not None and win_rate_5 is not None:
            f["class_drop_in_form"] = (
                1.0 if (grade_mv > 0 and win_rate_5 >= 0.3) else 0.0
            )
        else:
            f["class_drop_in_form"] = None
        if grade_mv is not None and last_pos is not None:
            f["class_rise_on_win"] = (
                1.0 if (grade_mv < 0 and last_pos == 1) else 0.0
            )
        else:
            f["class_rise_on_win"] = None

        # Median career grade index (exposed so the relative-features pass
        # can compute gap-to-field — raw value is lower-is-better since low
        # grade index = higher class)
        f["dog_median_career_grade_index"] = agg.get("median_career_grade_idx")

        # Typical grade gap: current grade index minus median career index.
        # Positive = dog is dropping to an easier grade than usual.
        curr_g = None
        if current_grade is not None:
            curr_g = grade_map.get(str(current_grade).upper().strip())
        median_idx = agg.get("median_career_grade_idx")
        if curr_g is not None and median_idx is not None:
            f["dog_typical_grade_gap"] = float(curr_g - median_idx)
        else:
            f["dog_typical_grade_gap"] = None

        # Race-type flags (open race, stakes race)
        if current_grade is not None:
            g_up = str(current_grade).upper().strip()
            f["is_open_race"] = 1.0 if g_up == "OR" else 0.0
            f["is_stakes_race"] = 1.0 if g_up in {"S1", "S2", "S3", "S4", "S5"} else 0.0
        else:
            f["is_open_race"] = None
            f["is_stakes_race"] = None

        # 17. Tier 8 — stamina & finishing-profile
        f["finishing_speed_ratio_last5"] = agg.get("finishing_speed_ratio")
        f["finishing_speed_ewm_last10"] = agg.get("finishing_speed_ewm10")
        f["finishing_speed_trend_last5"] = agg.get("finishing_speed_trend")

        last_distance = agg.get("last_distance")
        if last_distance is not None and distance_m is not None:
            dist_change = float(distance_m) - float(last_distance)
            f["distance_change_from_last"] = dist_change
            f["is_distance_step_up"] = 1.0 if dist_change > 0 else 0.0
            f["is_distance_step_down"] = 1.0 if dist_change < 0 else 0.0
        else:
            f["distance_change_from_last"] = None
            f["is_distance_step_up"] = None
            f["is_distance_step_down"] = None

        # 18. Tier 9 — first-time / change flags
        prior_tracks = agg.get("prior_tracks") or set()
        prior_distances = agg.get("prior_distances") or set()
        prior_grades = agg.get("prior_grades") or set()
        career = agg.get("career_races", 0.0) or 0.0

        # "First time at X" only makes sense after the dog has some career
        # form; otherwise every feature degenerates to 1.0 for debut runners.
        if career > 0:
            f["first_time_at_track"] = (
                1.0 if track_id is not None and int(track_id) not in prior_tracks else 0.0
            )
            f["first_time_at_distance"] = (
                1.0 if distance_m is not None and int(distance_m) not in prior_distances else 0.0
            )
            if current_grade is not None:
                f["first_time_at_grade"] = (
                    1.0 if str(current_grade).upper().strip() not in prior_grades else 0.0
                )
            else:
                f["first_time_at_grade"] = None
        else:
            # Debut: flag separately via career_races (already exposed)
            f["first_time_at_track"] = None
            f["first_time_at_distance"] = None
            f["first_time_at_grade"] = None

        # Layoff flags derived from the days_since_last we computed upstream
        days_since = f.get("days_since_last")
        if days_since is not None:
            f["first_race_after_layoff"] = 1.0 if days_since >= 29 else 0.0
        else:
            f["first_race_after_layoff"] = None
        f["second_race_after_layoff"] = agg.get("second_after_layoff", 0.0)

        # 19. Tier 10 — workload / fitness-cycle
        f["races_last_14_days"] = agg.get("races_last_14_days")
        f["races_last_60_days"] = agg.get("races_last_60_days")
        f["workload_trend"] = agg.get("workload_trend")
        f["weight_3race_stdev"] = agg.get("weight_3race_stdev")

        career_avg_w = agg.get("career_avg_weight")
        current_w = ctx["weight_kg"]
        if (
            current_w is not None and career_avg_w is not None
            and career_avg_w > 0
        ):
            f["weight_pct_of_career_avg"] = float(current_w) / float(career_avg_w)
        else:
            f["weight_pct_of_career_avg"] = None

        # 20. Tier 11 — trouble-adjusted performance
        f["clean_run_win_rate_last10"] = agg.get("clean_run_win_rate_last10")
        f["clean_run_mean_position_last10"] = agg.get("clean_run_mean_position_last10")
        f["trouble_run_mean_position_last10"] = agg.get("trouble_run_mean_position_last10")
        f["trouble_recovery_ratio_last10"] = agg.get("trouble_recovery_ratio_last10")
        f["clean_run_count_last10"] = agg.get("clean_run_count_last10")
        f["trouble_run_count_last10"] = agg.get("trouble_run_count_last10")

        # 21. Comment-derived features (running style, trip shape, stamina,
        # bend-by-bend trouble, rail/wide preference, clear wins).
        comment_rates = agg.get("comment_rates") or {}
        for name in COMMENT_FEATURE_NAMES:
            f[name] = comment_rates.get(name)

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
    # Tier 3 — speed figure (Beyer-style normalised time ratings)
    "speed_figure_best_last10",
    "speed_figure_mean_last5",
    "speed_figure_ewm_last10",
    "speed_figure_trend_last5",
    "career_peak_speed_figure",
    "recent_vs_peak_speed_figure",
    # Tier 6 — draw x bias x running-style interactions
    "trap_bias_deviation",
    "trap_bias_deviation_going",
    "trap_bias_x_front_runner",
    "wide_runner_x_outside",
    "wide_runner_x_inside",
    "trap_switch",
    # Tier 7 — class/grade dynamics
    "class_drop_in_form",
    "class_rise_on_win",
    "dog_median_career_grade_index",
    "dog_typical_grade_gap",
    "is_open_race",
    "is_stakes_race",
    # Tier 8 — stamina & finishing profile
    "finishing_speed_ratio_last5",
    "finishing_speed_ewm_last10",
    "finishing_speed_trend_last5",
    "distance_change_from_last",
    "is_distance_step_up",
    "is_distance_step_down",
    # Tier 9 — first-time / change flags
    "first_time_at_track",
    "first_time_at_distance",
    "first_time_at_grade",
    "first_race_after_layoff",
    "second_race_after_layoff",
    # Tier 10 — workload / fitness-cycle
    "races_last_14_days",
    "races_last_60_days",
    "workload_trend",
    "weight_3race_stdev",
    "weight_pct_of_career_avg",
    # Tier 11 — trouble-adjusted performance
    "clean_run_win_rate_last10",
    "clean_run_mean_position_last10",
    "trouble_run_mean_position_last10",
    "trouble_recovery_ratio_last10",
    "clean_run_count_last10",
    "trouble_run_count_last10",
    # Comment-derived structured features (parsed from free-text comments)
    *COMMENT_FEATURE_NAMES,
]


# Registry of ELO feature names emitted by compute_elo_features_batch
ELO_FEATURE_NAMES = [
    "dog_elo",
    "dog_elo_at_distance",
    "dog_elo_at_track",
    "dog_elo_races",
    "field_avg_elo",
    "field_max_elo",
    "elo_rank_in_field",
    "elo_gap_to_best",
    "elo_gap_to_avg",
]


# Registry of head-to-head feature names emitted by compute_h2h_features_batch
H2H_FEATURE_NAMES = [
    "h2h_meetings_vs_field",
    "h2h_wins_vs_field",
    "h2h_losses_vs_field",
    "h2h_win_rate_vs_field",
    "h2h_avg_beaten_length_vs_field",
    "best_opponent_beaten_count",
]


def compute_h2h_features_batch(
    db: Session,
    entry_ids: list[int],
    heartbeat_fn=None,
) -> pd.DataFrame:
    """Compute head-to-head features measuring each dog's prior record
    against the specific opponents it faces in today's race.

    For each target entry:
      * h2h_meetings_vs_field           — count of prior races where the dog
                                            faced at least one opponent who is
                                            also in today's race.
      * h2h_wins_vs_field               — count of prior pairwise wins against
                                            those same opponents.
      * h2h_losses_vs_field             — count of prior pairwise losses.
      * h2h_win_rate_vs_field           — Beta(1,3)-smoothed rate.
      * h2h_avg_beaten_length_vs_field  — average beaten-distance margin in
                                            losses to today's opponents.
      * best_opponent_beaten_count      — number of distinct current-field
                                            opponents the dog has beaten at
                                            least once.
    """
    def _hb():
        if heartbeat_fn is not None:
            heartbeat_fn()

    if not entry_ids:
        return pd.DataFrame()

    requested = list(set(entry_ids))

    # Step 1: load target entry context (race_id, dog_id, race_date)
    ctx_rows = (
        db.query(
            RaceEntry.id.label("entry_id"),
            RaceEntry.race_id,
            RaceEntry.dog_id,
            Race.race_date,
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.id.in_(requested))
        .all()
    )
    if not ctx_rows:
        return pd.DataFrame()

    target_df = pd.DataFrame(ctx_rows, columns=[
        "entry_id", "race_id", "dog_id", "race_date",
    ]).set_index("entry_id")

    # Step 2: collect the full field for each target race, including
    # unresulted races (prediction-time).
    # Cast to plain ints — SQLite's in_() binding rejects numpy.int64 values
    # silently (returns no rows) when they arrive from a pandas unique().
    target_race_ids = [int(r) for r in target_df["race_id"].unique()]
    field_rows = (
        db.query(RaceEntry.race_id, RaceEntry.dog_id)
        .filter(RaceEntry.race_id.in_(target_race_ids))
        .all()
    )
    field_by_race: dict[int, set[int]] = defaultdict(set)
    for rid, did in field_rows:
        field_by_race[int(rid)].add(int(did))

    dogs_of_interest: set[int] = set()
    for dogs in field_by_race.values():
        dogs_of_interest.update(dogs)

    if not dogs_of_interest:
        return pd.DataFrame()

    _hb()

    # Step 3: fetch resulted race entries for all dogs-of-interest with their
    # finish_position and beaten_distance.  We fetch only what we need and
    # group per race.
    def _entries_query(chunk):
        return (
            db.query(
                RaceEntry.race_id,
                RaceEntry.dog_id,
                RaceEntry.finish_position,
                RaceEntry.beaten_distance,
                Race.race_date,
            )
            .join(Race, RaceEntry.race_id == Race.id)
            .filter(
                RaceEntry.dog_id.in_(chunk),
                Race.status == "resulted",
                RaceEntry.finish_position.isnot(None),
            )
            .all()
        )

    history_rows = _chunked_query(
        db, _entries_query, list(dogs_of_interest), RaceEntry.dog_id,
    )

    if not history_rows:
        return pd.DataFrame()

    # race_id -> list of (dog_id, finish_position, beaten_distance, race_date)
    race_to_entries: dict[int, list[tuple]] = defaultdict(list)
    for r in history_rows:
        race_to_entries[r.race_id].append(
            (r.dog_id, r.finish_position, r.beaten_distance, r.race_date)
        )

    _hb()

    # Step 4: build dog -> races timeline (race_id, race_date, fin_pos, bd)
    dog_to_races: dict[int, list[tuple]] = defaultdict(list)
    for race_id, entries in race_to_entries.items():
        for dog_id, fp, bd, rd in entries:
            dog_to_races[dog_id].append((race_id, rd, fp, bd))
    # Sort each dog's timeline by race_date for efficient filtering
    for d in dog_to_races:
        dog_to_races[d].sort(key=lambda t: t[1])

    _hb()

    # Step 5: compute per-entry H2H stats
    rows: dict[int, dict[str, float | None]] = {}
    for entry_id, ctx in target_df.iterrows():
        dog_id = int(ctx["dog_id"])
        race_id = int(ctx["race_id"])
        race_date = ctx["race_date"]

        opponents = field_by_race.get(race_id, set()) - {dog_id}
        if not opponents:
            # Solo race or no field data — leave NaN
            rows[entry_id] = {
                "h2h_meetings_vs_field": None,
                "h2h_wins_vs_field": None,
                "h2h_losses_vs_field": None,
                "h2h_win_rate_vs_field": None,
                "h2h_avg_beaten_length_vs_field": None,
                "best_opponent_beaten_count": None,
            }
            continue

        wins = 0
        losses = 0
        meetings = 0
        beaten_lengths: list[float] = []
        opponents_beaten: set[int] = set()

        for (past_race_id, past_date, fp, bd) in dog_to_races.get(dog_id, []):
            if past_date >= race_date:
                break  # history is sorted; all remaining are on/after target
            entries_in_past = race_to_entries.get(past_race_id, [])
            # Look up this dog's and each opponent's finish in that past race
            for other_dog_id, other_fp, other_bd, _ in entries_in_past:
                if other_dog_id == dog_id or other_dog_id not in opponents:
                    continue
                if fp is None or other_fp is None:
                    continue
                meetings += 1
                if fp < other_fp:
                    wins += 1
                    opponents_beaten.add(other_dog_id)
                elif fp > other_fp:
                    losses += 1
                    if bd is not None:
                        beaten_lengths.append(float(bd))

        # Beta(1,3) shrinkage keeps the feature well-behaved for small samples.
        # Prior favours "not particularly good vs this field" so the raw rate
        # only starts to influence the estimate once meetings accumulate.
        win_rate = (wins + 1.0) / (meetings + 4.0) if meetings >= 0 else None
        avg_beaten = float(np.mean(beaten_lengths)) if beaten_lengths else None

        rows[entry_id] = {
            "h2h_meetings_vs_field": float(meetings),
            "h2h_wins_vs_field": float(wins),
            "h2h_losses_vs_field": float(losses),
            "h2h_win_rate_vs_field": win_rate,
            "h2h_avg_beaten_length_vs_field": avg_beaten,
            "best_opponent_beaten_count": float(len(opponents_beaten)),
        }

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "race_entry_id"
    logger.info(
        "H2H: computed head-to-head features for %d/%d requested entries",
        len(df), len(requested),
    )
    return df


def compute_elo_features_batch(
    db: Session,
    entry_ids: list[int],
    heartbeat_fn=None,
    k: float = 24.0,
    initial: float = 1500.0,
) -> pd.DataFrame:
    """Compute pre-race ELO ratings (overall, per-track, per-distance) for a
    set of entries.

    Walks every resulted race in chronological order, maintaining three
    independent ELO tables (overall, per-distance, per-track).  For each
    requested entry, snapshots the dog's pre-race rating *before* applying
    that race's update — ensuring no leakage from the entry's own outcome.

    Field-relative features (rank, gap-to-best, gap-to-avg, average) are
    computed within each requested race using the snapshotted pre-race
    overall ELO.

    Returns a DataFrame indexed by race_entry_id.  Returns an empty
    DataFrame when entry_ids is empty.
    """
    def _hb():
        if heartbeat_fn is not None:
            heartbeat_fn()

    if not entry_ids:
        return pd.DataFrame()

    requested_set = set(entry_ids)

    logger.info(
        "ELO: loading all resulted entries up to max race date for %d targets...",
        len(entry_ids),
    )

    # Find the latest race date among the requested entries; we only need to
    # walk history up through that date.  Includes unresulted races so that
    # prediction-time requests for future races are also bounded correctly.
    target_max_date = (
        db.query(func.max(Race.race_date))
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.id.in_(list(requested_set)))
        .scalar()
    )
    if target_max_date is None:
        return pd.DataFrame()

    # Pull every resulted race entry up to (and including) target_max_date,
    # ordered chronologically.  This is the same pass used by Glicko-style
    # rating systems — we must process races in the order they happened.
    # Also include requested entries whose race is NOT yet resulted (e.g.
    # live prediction time) so we can still snapshot pre-race ELO for them.
    requested_ids_list = list(requested_set)
    rows = (
        db.query(
            RaceEntry.id.label("entry_id"),
            RaceEntry.race_id,
            RaceEntry.dog_id,
            RaceEntry.finish_position,
            Race.status.label("race_status"),
            Race.race_date,
            Race.race_time,
            Race.track_id,
            Race.distance_m,
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            (
                (Race.status == "resulted") & (Race.race_date <= target_max_date)
            ) | (
                RaceEntry.id.in_(requested_ids_list)
            ),
        )
        .order_by(Race.race_date.asc(), Race.race_time.asc().nullsfirst(),
                  Race.id.asc())
        .all()
    )
    _hb()

    if not rows:
        return pd.DataFrame()

    elo_overall = EloRatings(k=k, initial=initial)
    elo_distance: dict[int, EloRatings] = {}
    elo_track: dict[int, EloRatings] = {}

    # Group rows by race_id while preserving order
    current_race_id = None
    current_group: list = []
    snapshots: dict[int, dict[str, float | int | None]] = {}

    def _flush(group):
        if not group:
            return
        race_track_id = group[0].track_id
        race_distance = group[0].distance_m
        dog_ids = [r.dog_id for r in group]

        track_book = elo_track.setdefault(
            race_track_id, EloRatings(k=k, initial=initial),
        )
        dist_book = elo_distance.setdefault(
            race_distance, EloRatings(k=k, initial=initial),
        )

        # Snapshot pre-race ELO (overall + context) for any requested entries
        pre_overall = {d: elo_overall.get(d) for d in dog_ids}
        avg_elo = sum(pre_overall.values()) / len(pre_overall)
        max_elo = max(pre_overall.values())
        # rank_in_field: 1 = highest ELO
        sorted_dogs = sorted(pre_overall.items(), key=lambda x: -x[1])
        rank_map = {d: i + 1 for i, (d, _) in enumerate(sorted_dogs)}

        for entry in group:
            if entry.entry_id not in requested_set:
                continue
            d = entry.dog_id
            dog_elo = pre_overall[d]
            snapshots[entry.entry_id] = {
                "dog_elo": dog_elo,
                "dog_elo_at_distance": dist_book.get(d),
                "dog_elo_at_track": track_book.get(d),
                "dog_elo_races": float(elo_overall.count(d)),
                "field_avg_elo": avg_elo,
                "field_max_elo": max_elo,
                "elo_rank_in_field": float(rank_map[d]),
                "elo_gap_to_best": max_elo - dog_elo,
                "elo_gap_to_avg": dog_elo - avg_elo,
            }

        # Apply updates only if the race itself is resulted.  Unresulted
        # target races (prediction-time) must not feed back into ELO state.
        is_resulted = group[0].race_status == "resulted"
        if is_resulted:
            results = [(r.dog_id, r.finish_position) for r in group
                       if r.finish_position is not None]
            if results:
                elo_overall.update_race(results)
                dist_book.update_race(results)
                track_book.update_race(results)

    races_seen = 0
    for r in rows:
        if r.race_id != current_race_id:
            _flush(current_group)
            current_group = []
            current_race_id = r.race_id
            races_seen += 1
            if races_seen % 5000 == 0:
                logger.info("ELO: processed %d races...", races_seen)
                _hb()
        current_group.append(r)
    _flush(current_group)

    logger.info(
        "ELO: snapshotted ratings for %d/%d requested entries across %d races",
        len(snapshots), len(requested_set), races_seen,
    )

    if not snapshots:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(snapshots, orient="index")
    df.index.name = "race_entry_id"
    return df

"""
Data integrity checks for feature materialization.

Detects when scraped data is incomplete across tracks/dates, which can cause
features like "mean last 5 race times" to be silently wrong — e.g. if a dog
raced at Dublin and Limerick but only Limerick data has been scraped.
"""

import logging
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track

logger = logging.getLogger(__name__)

# If an active track has no races for this many consecutive days within the
# date range that has *any* racing (i.e. between its first and last scraped
# race), we consider that a coverage gap.  Greyhound tracks typically race
# 2-3 times per week, so 14 days without a single race is suspicious.
DEFAULT_MAX_GAP_DAYS = 14


def get_track_date_coverage(
    db: Session,
    start_date: date,
    end_date: date,
) -> dict[str, list[date]]:
    """
    Return {track_code: [list of dates with at least one resulted race]}
    for all active tracks in the given date range.
    """
    rows = (
        db.query(Track.code, Race.race_date)
        .join(Race, Race.track_id == Track.id)
        .filter(
            Track.active.is_(True),
            Race.status == "resulted",
            Race.race_date >= start_date,
            Race.race_date <= end_date,
        )
        .distinct()
        .all()
    )

    coverage: dict[str, list[date]] = defaultdict(list)
    for code, rd in rows:
        coverage[code].append(rd)

    # Sort each track's dates
    for code in coverage:
        coverage[code].sort()

    return dict(coverage)


def find_coverage_gaps(
    db: Session,
    start_date: date,
    end_date: date,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> list[dict[str, Any]]:
    """
    Find tracks with suspicious gaps in scraped data.

    Returns a list of dicts:
        {"track_code", "track_name", "gap_start", "gap_end", "gap_days"}

    A gap is flagged when a track has no resulted races for more than
    `max_gap_days` consecutive days *within its known racing window*.
    Tracks with zero races in the range are reported as a single gap
    spanning the entire range.
    """
    active_tracks = (
        db.query(Track.code, Track.name)
        .filter(Track.active.is_(True))
        .all()
    )
    track_names = {code: name for code, name in active_tracks}
    coverage = get_track_date_coverage(db, start_date, end_date)

    gaps: list[dict[str, Any]] = []

    for code, name in active_tracks:
        dates = coverage.get(code, [])

        if not dates:
            # Track has no data at all in this range
            gaps.append({
                "track_code": code,
                "track_name": name,
                "gap_start": start_date,
                "gap_end": end_date,
                "gap_days": (end_date - start_date).days,
            })
            continue

        # Check gap from start_date to first race
        if (dates[0] - start_date).days > max_gap_days:
            gaps.append({
                "track_code": code,
                "track_name": name,
                "gap_start": start_date,
                "gap_end": dates[0],
                "gap_days": (dates[0] - start_date).days,
            })

        # Check gaps between consecutive race dates
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days
            if gap > max_gap_days:
                gaps.append({
                    "track_code": code,
                    "track_name": name,
                    "gap_start": dates[i - 1],
                    "gap_end": dates[i],
                    "gap_days": gap,
                })

        # Check gap from last race to end_date
        if (end_date - dates[-1]).days > max_gap_days:
            gaps.append({
                "track_code": code,
                "track_name": name,
                "gap_start": dates[-1],
                "gap_end": end_date,
                "gap_days": (end_date - dates[-1]).days,
            })

    return sorted(gaps, key=lambda g: g["gap_days"], reverse=True)


def tracks_with_gaps_in_range(
    db: Session,
    start_date: date,
    end_date: date,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> set[int]:
    """
    Return the set of track IDs that have coverage gaps in the given range.
    Useful for quickly checking if a dog's history window overlaps with
    incomplete track data.
    """
    active_tracks = (
        db.query(Track.id, Track.code)
        .filter(Track.active.is_(True))
        .all()
    )
    code_to_id = {code: tid for tid, code in active_tracks}

    gaps = find_coverage_gaps(db, start_date, end_date, max_gap_days)
    gap_codes = {g["track_code"] for g in gaps}

    return {code_to_id[c] for c in gap_codes if c in code_to_id}


def check_dog_history_complete(
    db: Session,
    dog_id: int,
    before_date: date,
    window_days: int = 90,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> bool:
    """
    Check whether a dog's race history is likely complete for the given window.

    Logic:
    1. Determine the date range the feature window covers
       (before_date - window_days .. before_date).
    2. Find which tracks the dog has *ever* raced at.
    3. Check if any of those tracks have coverage gaps in the window.

    Returns True if the data looks complete, False if there are gaps that
    could make features unreliable.
    """
    window_start = before_date - timedelta(days=window_days)

    # Tracks this dog has ever raced at (not just in the window — the point
    # is to catch missing data from tracks we *haven't* scraped for this period)
    dog_track_ids = (
        db.query(Race.track_id)
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.dog_id == dog_id)
        .distinct()
        .all()
    )
    dog_track_ids = {row[0] for row in dog_track_ids}

    if not dog_track_ids:
        # No history at all — nothing to validate
        return True

    # Check coverage for those tracks
    gap_track_ids = tracks_with_gaps_in_range(db, window_start, before_date, max_gap_days)

    overlapping = dog_track_ids & gap_track_ids
    if overlapping:
        # Get track names for logging
        tracks = db.query(Track.code).filter(Track.id.in_(overlapping)).all()
        track_codes = [t[0] for t in tracks]
        logger.debug(
            "Dog %d has incomplete data: tracks %s have gaps in %s..%s",
            dog_id, track_codes, window_start, before_date,
        )
        return False

    return True


def assess_materialization_readiness(
    db: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> dict[str, Any]:
    """
    Comprehensive pre-materialization check.

    Returns a report with:
    - coverage_gaps: list of track/date gaps
    - tracks_missing: tracks with zero data in range
    - tracks_ok: tracks with no gaps
    - recommendation: "safe" | "warning" | "incomplete"
    - affected_dogs: count of dogs who raced at tracks with gaps
    """
    if not start_date or not end_date:
        # Default: use the full range of data in the DB
        date_range = (
            db.query(
                func.min(Race.race_date),
                func.max(Race.race_date),
            )
            .filter(Race.status == "resulted")
            .first()
        )
        if not date_range or not date_range[0]:
            return {
                "coverage_gaps": [],
                "tracks_missing": [],
                "tracks_ok": [],
                "recommendation": "incomplete",
                "message": "No resulted races found in database",
                "affected_dog_count": 0,
            }
        start_date = date_range[0]
        end_date = date_range[1]

    gaps = find_coverage_gaps(db, start_date, end_date, max_gap_days)

    active_tracks = (
        db.query(Track.code, Track.name)
        .filter(Track.active.is_(True))
        .all()
    )
    all_codes = {code for code, _ in active_tracks}
    coverage = get_track_date_coverage(db, start_date, end_date)

    tracks_missing = [
        {"code": code, "name": name}
        for code, name in active_tracks
        if code not in coverage
    ]
    tracks_with_data = {code for code in coverage}
    gap_codes = {g["track_code"] for g in gaps}
    tracks_ok = sorted(tracks_with_data - gap_codes)

    # Count dogs affected by gaps
    affected_dog_count = 0
    if gap_codes:
        gap_track_ids = (
            db.query(Track.id)
            .filter(Track.code.in_(gap_codes))
            .all()
        )
        gap_track_ids = [row[0] for row in gap_track_ids]

        if gap_track_ids:
            affected_dog_count = (
                db.query(func.count(func.distinct(RaceEntry.dog_id)))
                .join(Race, RaceEntry.race_id == Race.id)
                .filter(Race.track_id.in_(gap_track_ids))
                .scalar()
            ) or 0

    # Determine recommendation
    if not coverage:
        recommendation = "incomplete"
        message = "No scraped data found in the specified date range"
    elif tracks_missing:
        recommendation = "incomplete"
        message = (
            f"{len(tracks_missing)} active track(s) have no data at all. "
            f"{affected_dog_count} dogs may have incomplete history."
        )
    elif gaps:
        recommendation = "warning"
        message = (
            f"{len(gaps)} coverage gap(s) found across {len(gap_codes)} track(s). "
            f"{affected_dog_count} dogs may have incomplete history. "
            "Features computed for these dogs may be inaccurate."
        )
    else:
        recommendation = "safe"
        message = f"All {len(tracks_ok)} tracks have continuous coverage from {start_date} to {end_date}"

    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "max_gap_days": max_gap_days,
        "coverage_gaps": gaps,
        "tracks_missing": tracks_missing,
        "tracks_ok": tracks_ok,
        "recommendation": recommendation,
        "message": message,
        "affected_dog_count": affected_dog_count,
    }

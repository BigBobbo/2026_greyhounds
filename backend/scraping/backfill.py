"""
Historical backfill script for scraping GRI race results.

Usage:
    python -m scraping.backfill --start 2021-01-01 --end 2026-04-05
    python -m scraping.backfill --start 2021-01-01 --end 2026-04-05 --tracks SHP,CRK
    python -m scraping.backfill --discover-tracks  # discover and update track codes first
"""

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.models.track import Track
from scraping.gri_scraper import scrape_results, discover_track_codes
from scraping.db_pipeline import (
    pop_out_of_order_dogs,
    recompute_days_since_last,
    upsert_race_results,
)
import app.models  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def run_discover_tracks():
    """Discover track codes from GRI website and update the database."""
    logger.info("Discovering track codes from GRI website...")
    tracks = await discover_track_codes()

    db = SessionLocal()
    try:
        for track_info in tracks:
            existing = db.query(Track).filter(Track.code == track_info["code"]).first()
            if existing:
                logger.info("Track %s (%s) already exists", track_info["code"], track_info["name"])
            else:
                # Check if name matches an existing track with different code
                by_name = db.query(Track).filter(Track.name == track_info["name"]).first()
                if by_name:
                    logger.info(
                        "Updating track code: %s -> %s for %s",
                        by_name.code, track_info["code"], track_info["name"],
                    )
                    by_name.code = track_info["code"]
                else:
                    logger.info("Adding new track: %s (%s)", track_info["code"], track_info["name"])
                    db.add(Track(code=track_info["code"], name=track_info["name"]))

        db.commit()
        logger.info("Track discovery complete. %d tracks found.", len(tracks))
    finally:
        db.close()


def run_backfill(
    start_date: date,
    end_date: date,
    track_codes: list[str] | None = None,
    delay: float = 2.0,
):
    """Run historical backfill for specified tracks and date range.

    Synchronous (the shared job runner manages its own event loop). One
    ScrapeLog per track is preserved: each track gets its own
    `run_scrape_job` call, so per-track progress/failures stay visible in
    the scraping UI exactly as before.
    """
    from scraping.job_runner import run_scrape_job

    db = SessionLocal()
    try:
        if track_codes:
            tracks = db.query(Track).filter(Track.code.in_(track_codes)).all()
        else:
            tracks = db.query(Track).filter(Track.active.is_(True)).all()
        codes = [t.code for t in tracks]
    finally:
        db.close()

    if not codes:
        logger.error("No tracks found! Run --discover-tracks first.")
        return

    logger.info(
        "Starting backfill: %d tracks, %s to %s", len(codes), start_date, end_date,
    )

    dates = [
        start_date + timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]

    total_races = 0
    total_entries = 0

    for code in codes:
        result = run_scrape_job(
            SessionLocal,
            [code],
            dates,
            scrape_results,
            upsert_race_results,
            spider_name="gri",
            source_desc=f"backfill {code} {start_date} to {end_date}",
            delay=delay,
        )
        total_races += result["races_new"]
        total_entries += result["entries_new"]
        logger.info(
            "Track %s complete: %d races, %d entries (%d failed day(s))",
            code, result["races_new"], result["entries_new"],
            len(result["failed_pairs"]),
        )

    # Track-by-track iteration inserts races out of chronological order,
    # corrupting days_since_last on later entries written first. Heal
    # once at the end when any out-of-order insert was flagged.
    flagged = pop_out_of_order_dogs()
    if flagged:
        db = SessionLocal()
        try:
            healed = recompute_days_since_last(db)
            logger.info(
                "Healed days_since_last: %d entries corrected (%d dogs flagged)",
                healed, len(flagged),
            )
        finally:
            db.close()

    logger.info(
        "Backfill complete: %d total races, %d total entries",
        total_races, total_entries,
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill GRI race results")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tracks", type=str, help="Comma-separated track codes (e.g. SHP,CRK)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    parser.add_argument("--discover-tracks", action="store_true", help="Discover track codes from GRI")
    args = parser.parse_args()

    if args.discover_tracks:
        asyncio.run(run_discover_tracks())
        return

    if not args.start or not args.end:
        parser.error("--start and --end are required (unless using --discover-tracks)")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    track_codes = args.tracks.split(",") if args.tracks else None

    run_backfill(start, end, track_codes, args.delay)


if __name__ == "__main__":
    main()

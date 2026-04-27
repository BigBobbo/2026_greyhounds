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
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal, engine, Base
from app.models.track import Track
from app.models.scrape_log import ScrapeLog
from scraping.gri_scraper import scrape_results, discover_track_codes
from scraping.db_pipeline import upsert_race_results
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


async def run_backfill(
    start_date: date,
    end_date: date,
    track_codes: list[str] | None = None,
    delay: float = 2.0,
):
    """Run historical backfill for specified tracks and date range."""
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if track_codes:
            tracks = db.query(Track).filter(Track.code.in_(track_codes)).all()
        else:
            tracks = db.query(Track).filter(Track.active.is_(True)).all()

        if not tracks:
            logger.error("No tracks found! Run --discover-tracks first.")
            return

        logger.info(
            "Starting backfill: %d tracks, %s to %s",
            len(tracks), start_date, end_date,
        )

        total_races = 0
        total_entries = 0

        for track in tracks:
            # Create scrape log
            log = ScrapeLog(
                spider_name="gri",
                source=f"backfill {track.code} {start_date} to {end_date}",
                status="running",
                started_at=datetime.utcnow(),
            )
            db.add(log)
            db.commit()

            track_races = 0
            track_entries = 0
            current = start_date

            try:
                while current <= end_date:
                    races = await scrape_results(track.code, current)

                    if races:
                        stats = upsert_race_results(db, races, scrape_log_id=log.id)
                        track_races += stats["races_new"]
                        track_entries += stats["entries_new"]

                    log.heartbeat_at = datetime.utcnow()
                    db.commit()
                    current += timedelta(days=1)
                    await asyncio.sleep(delay)

                log.status = "success"
                log.records_scraped = track_races
                log.records_new = track_entries

            except Exception as e:
                logger.error("Error scraping %s: %s", track.code, e)
                log.status = "failed"
                log.error_message = str(e)

            log.completed_at = datetime.utcnow()
            db.commit()

            total_races += track_races
            total_entries += track_entries
            logger.info(
                "Track %s complete: %d races, %d entries",
                track.code, track_races, track_entries,
            )

        logger.info(
            "Backfill complete: %d total races, %d total entries",
            total_races, total_entries,
        )
    finally:
        db.close()


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

    asyncio.run(run_backfill(start, end, track_codes, args.delay))


if __name__ == "__main__":
    main()

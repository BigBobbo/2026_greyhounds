"""
APScheduler setup for automated scraping jobs.

Integrates with FastAPI app lifecycle.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from threading import Thread

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.database import SessionLocal
from app.models.scrape_log import ScrapeLog
from app.models.track import Track

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def _scrape_all_tracks_today():
    """Scrape today's results for all active tracks."""
    from scraping.gri_scraper import scrape_results
    from scraping.db_pipeline import upsert_race_results

    async def _run():
        db = SessionLocal()
        try:
            tracks = db.query(Track).filter(Track.active.is_(True)).all()
            today = date.today()

            log = ScrapeLog(
                spider_name="gri",
                source=f"scheduled daily {today}",
                status="running",
                started_at=datetime.utcnow(),
            )
            db.add(log)
            db.commit()

            total_races = 0
            total_new = 0

            for track in tracks:
                try:
                    races = await scrape_results(track.code, today)
                    if races:
                        stats = upsert_race_results(db, races)
                        total_races += stats["races_new"] + stats["races_updated"]
                        total_new += stats["races_new"]
                    await asyncio.sleep(2.0)
                except Exception as e:
                    logger.error("Error scraping %s: %s", track.code, e)

            log.status = "success"
            log.records_scraped = total_races
            log.records_new = total_new
            log.completed_at = datetime.utcnow()
            db.commit()

            logger.info("Daily scrape complete: %d races (%d new)", total_races, total_new)

        except Exception as e:
            logger.error("Daily scrape failed: %s", e)
        finally:
            db.close()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_run())
    loop.close()


def _scrape_yesterday_results():
    """Scrape yesterday's results (ensures we catch late-posted results)."""
    from scraping.gri_scraper import scrape_results
    from scraping.db_pipeline import upsert_race_results

    async def _run():
        db = SessionLocal()
        try:
            tracks = db.query(Track).filter(Track.active.is_(True)).all()
            yesterday = date.today() - timedelta(days=1)

            log = ScrapeLog(
                spider_name="gri",
                source=f"scheduled yesterday {yesterday}",
                status="running",
                started_at=datetime.utcnow(),
            )
            db.add(log)
            db.commit()

            total_races = 0
            for track in tracks:
                try:
                    races = await scrape_results(track.code, yesterday)
                    if races:
                        stats = upsert_race_results(db, races)
                        total_races += stats["races_new"] + stats["races_updated"]
                    await asyncio.sleep(2.0)
                except Exception as e:
                    logger.error("Error scraping %s: %s", track.code, e)

            log.status = "success"
            log.records_scraped = total_races
            log.completed_at = datetime.utcnow()
            db.commit()
        except Exception as e:
            logger.error("Yesterday scrape failed: %s", e)
        finally:
            db.close()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_run())
    loop.close()


def start_scheduler():
    """Start the APScheduler with configured jobs."""
    # Scrape today's results at 23:00 daily
    scheduler.add_job(
        _scrape_all_tracks_today,
        trigger=CronTrigger(hour=23, minute=0),
        id="daily_results",
        name="Daily GRI results scrape",
        replace_existing=True,
    )

    # Scrape yesterday's results at 08:00 (catch late posts)
    scheduler.add_job(
        _scrape_yesterday_results,
        trigger=CronTrigger(hour=8, minute=0),
        id="yesterday_results",
        name="Yesterday GRI results scrape",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))


def stop_scheduler():
    """Gracefully stop the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

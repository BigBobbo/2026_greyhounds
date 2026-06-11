"""
APScheduler setup for automated scraping and daily prediction jobs.

Two job kinds run here:

  - Built-in fixed-time scrape jobs (today at 23:00, yesterday at 08:00).
  - Per-``ModelSchedule`` cron jobs registered dynamically. The job ids
    follow the pattern ``schedule_<id>`` so the schedule API can register
    or unregister them in response to CRUD calls.

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


# --- Per-schedule prediction jobs ---


def _schedule_job_id(schedule_id: int) -> str:
    return f"schedule_{schedule_id}"


def _run_schedule_in_thread(schedule_id: int):
    """APScheduler entrypoint: dispatch a daily prediction run.

    The actual work happens in ``schedule_service.run_schedule_job``;
    this wrapper exists so APScheduler stores a reference to a top-level
    function (it can't pickle closures) and so we can isolate the import
    from app start-up.
    """
    from app.services.schedule_service import run_schedule_job
    try:
        run_schedule_job(schedule_id, trigger="scheduled")
    except Exception as e:
        logger.error("Schedule %d cron run crashed: %s", schedule_id, e)


def register_schedule_job(sched) -> None:
    """Register or replace the cron job for a ModelSchedule row.

    Called from the schedule API on create/update. Safe to call if the
    scheduler hasn't started yet — APScheduler queues the job and runs
    it once start() happens.
    """
    try:
        scheduler.add_job(
            _run_schedule_in_thread,
            trigger=CronTrigger(
                hour=sched.cron_hour,
                minute=sched.cron_minute,
                timezone=sched.timezone,
            ),
            id=_schedule_job_id(sched.id),
            name=f"Daily prediction (experiment {sched.experiment_id})",
            args=[sched.id],
            replace_existing=True,
        )
        logger.info(
            "Registered schedule %d cron=%02d:%02d %s",
            sched.id, sched.cron_hour, sched.cron_minute, sched.timezone,
        )
    except Exception as e:
        logger.error("Failed to register schedule %d: %s", sched.id, e)


def unregister_schedule_job(schedule_id: int) -> None:
    """Remove a per-schedule cron job. Idempotent."""
    job_id = _schedule_job_id(schedule_id)
    try:
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
            logger.info("Unregistered schedule %d", schedule_id)
    except Exception as e:
        logger.warning("Unregister schedule %d: %s", schedule_id, e)


def _load_persisted_schedules():
    """Load all enabled ``ModelSchedule`` rows and register their cron jobs.

    Called once at scheduler start-up. New schedules added later are
    registered immediately by the API layer.
    """
    from app.models.schedule import ModelSchedule

    db = SessionLocal()
    try:
        rows = (
            db.query(ModelSchedule).filter(ModelSchedule.enabled.is_(True)).all()
        )
        for sched in rows:
            register_schedule_job(sched)
        logger.info("Loaded %d persisted schedule(s)", len(rows))
    except Exception as e:
        logger.error("Failed loading persisted schedules: %s", e)
    finally:
        db.close()


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

    # Nightly off-site backup of the DB and model artifacts at 02:30.
    # Runs after the late-night results scrape so the snapshot includes it.
    from app.services.backup_service import run_backup_job
    scheduler.add_job(
        run_backup_job,
        trigger=CronTrigger(hour=2, minute=30),
        id="nightly_backup",
        name="Nightly off-site backup",
        replace_existing=True,
    )

    scheduler.start()

    # Register dynamic schedule jobs after start so add_job + cron triggers
    # get evaluated against the running scheduler's clock.
    _load_persisted_schedules()

    logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))


def stop_scheduler():
    """Gracefully stop the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

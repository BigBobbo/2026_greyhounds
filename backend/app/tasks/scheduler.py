"""
APScheduler setup for automated scraping and daily prediction jobs.

Three built-in jobs run here, all scheduled in Europe/Dublin so they track
Irish racing days through DST (the old UTC crons fired an hour off all
summer):

  - Today's results at 23:00 Dublin (after the last race).
  - Yesterday's results at 08:00 Dublin (catches late-posted results).
  - A trailing 14-day amendment re-scrape at 04:30 Dublin. GRI amends
    results after publication — corrected SPs, weights and even runner
    identities (a verified case: a trap's dog changed days after the
    result first appeared). Combined with the pipeline's
    corrections-allowed upsert, this window keeps stored results converged
    with GRI's current record.

Per-``ModelSchedule`` cron jobs are registered dynamically with the job id
pattern ``schedule_<id>`` so the schedule API can add/remove them.

Scrape-log statuses are honest: "success" only when every track scraped
cleanly, "partial" when some failed, "failed" when all did. The old code
marked success unconditionally, making scraper outages indistinguishable
from quiet days.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.database import SessionLocal
from app.models.scrape_log import ScrapeLog
from app.models.track import Track

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

DUBLIN = ZoneInfo("Europe/Dublin")


def _irish_today() -> date:
    """The current date in Ireland — the date GRI keys its pages by."""
    return datetime.now(DUBLIN).date()


def _scrape_dates_for_all_tracks(dates: list[date], source: str) -> None:
    """Scrape a list of dates across all active tracks with honest logging."""
    from scraping.db_pipeline import upsert_race_results
    from scraping.gri_scraper import ScrapeError, scrape_results

    async def _run():
        db = SessionLocal()
        try:
            tracks = db.query(Track).filter(Track.active.is_(True)).all()

            log = ScrapeLog(
                spider_name="gri",
                source=source,
                status="running",
                started_at=datetime.utcnow(),
            )
            db.add(log)
            db.commit()

            total_races = 0
            total_new = 0
            attempts = 0
            failures: list[str] = []

            for track in tracks:
                for d in dates:
                    attempts += 1
                    try:
                        races = await scrape_results(track.code, d)
                        if races:
                            stats = upsert_race_results(
                                db, races, scrape_log_id=log.id,
                            )
                            total_races += stats["races_new"] + stats["races_updated"]
                            total_new += stats["races_new"]
                    except ScrapeError as e:
                        failures.append(f"{track.code} {d}: {e}")
                        logger.error("Scrape failed %s %s: %s", track.code, d, e)
                    except Exception as e:
                        failures.append(f"{track.code} {d}: {e}")
                        logger.error("Error scraping %s %s: %s", track.code, d, e)
                    log.heartbeat_at = datetime.utcnow()
                    db.commit()
                    await asyncio.sleep(2.0)

            if not failures:
                log.status = "success"
            elif len(failures) < attempts:
                log.status = "partial"
            else:
                log.status = "failed"
            if failures:
                log.error_message = "; ".join(failures[:30])
            log.records_scraped = total_races
            log.records_new = total_new
            log.completed_at = datetime.utcnow()
            db.commit()

            logger.info(
                "%s complete: %d races (%d new), %d/%d fetches failed",
                source, total_races, total_new, len(failures), attempts,
            )

        except Exception as e:
            logger.error("%s crashed: %s", source, e)
            try:
                log.status = "failed"
                log.error_message = str(e)[:2000]
                log.completed_at = datetime.utcnow()
                db.commit()
            except Exception:
                pass
        finally:
            db.close()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_run())
    loop.close()


def _scrape_all_tracks_today():
    """Scrape today's results for all active tracks."""
    today = _irish_today()
    _scrape_dates_for_all_tracks([today], f"scheduled daily {today}")


def _scrape_yesterday_results():
    """Scrape yesterday's results (ensures we catch late-posted results)."""
    yesterday = _irish_today() - timedelta(days=1)
    _scrape_dates_for_all_tracks([yesterday], f"scheduled yesterday {yesterday}")


def _rescrape_trailing_window(days: int = 14):
    """Re-scrape the trailing window to pick up GRI's post-publication
    amendments (corrected SPs, weights, comments, runner identities)."""
    today = _irish_today()
    window = [today - timedelta(days=i) for i in range(2, days + 1)]
    _scrape_dates_for_all_tracks(
        window, f"amendment re-scrape {window[-1]}..{window[0]}",
    )


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
    # Scrape today's results at 23:00 Irish time daily
    scheduler.add_job(
        _scrape_all_tracks_today,
        trigger=CronTrigger(hour=23, minute=0, timezone=DUBLIN),
        id="daily_results",
        name="Daily GRI results scrape",
        replace_existing=True,
    )

    # Scrape yesterday's results at 08:00 Irish time (catch late posts)
    scheduler.add_job(
        _scrape_yesterday_results,
        trigger=CronTrigger(hour=8, minute=0, timezone=DUBLIN),
        id="yesterday_results",
        name="Yesterday GRI results scrape",
        replace_existing=True,
    )

    # Amendment reconciliation: re-scrape the trailing 14 days at 04:30
    # Irish time so post-publication corrections converge into the DB.
    scheduler.add_job(
        _rescrape_trailing_window,
        trigger=CronTrigger(hour=4, minute=30, timezone=DUBLIN),
        id="amendment_rescrape",
        name="Trailing 14-day amendment re-scrape",
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

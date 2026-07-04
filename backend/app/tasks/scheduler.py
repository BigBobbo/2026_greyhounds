"""
APScheduler setup for automated scraping and daily prediction jobs.

Two job kinds run here:

  - Built-in fixed-time scrape jobs (today at 23:00, yesterday at 08:00).
  - Per-``ModelSchedule`` cron jobs registered dynamically. The job ids
    follow the pattern ``schedule_<id>`` so the schedule API can register
    or unregister them in response to CRUD calls.

Integrates with FastAPI app lifecycle.
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models.race import Race
from app.models.scrape_log import ScrapeLog
from app.models.track import Track

logger = logging.getLogger(__name__)

# All Irish racing happens on Dublin wall-clock time. Cron hours and
# "today"/"yesterday" computations must use this zone — a UTC container
# would otherwise scrape the wrong day around midnight (and around DST).
DUBLIN_TZ = ZoneInfo("Europe/Dublin")

# Jobs stuck in 'running' with no heartbeat for this long are considered dead
# (process restart, OOM, crash above the cleanup handler).
STALE_JOB_MINUTES = 30

scheduler = BackgroundScheduler()


def _dublin_today():
    """Today's date on Dublin wall-clock time."""
    return datetime.now(DUBLIN_TZ).date()


def reap_stale_jobs(db: Session) -> int:
    """Mark dead 'running' ScrapeLog and Experiment rows as failed.

    A row is stale when its last sign of life — heartbeat_at, falling back
    to started_at/created_at, or nothing at all — is older than
    STALE_JOB_MINUTES. Called at app startup (jobs orphaned by a restart)
    and hourly thereafter. Returns the number of rows reaped.
    """
    from app.models.experiment import Experiment

    now = datetime.utcnow()
    cutoff = now - timedelta(minutes=STALE_JOB_MINUTES)
    reaped = 0
    note = "killed by restart or stalled"

    for log in db.query(ScrapeLog).filter(ScrapeLog.status == "running").all():
        last_alive = log.heartbeat_at or log.started_at
        if last_alive is None or last_alive < cutoff:
            log.status = "failed"
            log.completed_at = now
            log.error_message = (
                f"{log.error_message}; {note}" if log.error_message else note
            )
            reaped += 1

    for exp in db.query(Experiment).filter(Experiment.status == "running").all():
        last_alive = exp.heartbeat_at or exp.created_at
        if last_alive is None or last_alive < cutoff:
            exp.status = "failed"
            exp.completed_at = now
            exp.error_message = (
                f"{exp.error_message}; {note}" if exp.error_message else note
            )
            reaped += 1

    if reaped:
        db.commit()
        logger.warning("Reaped %d stale running job(s): %s", reaped, note)
    return reaped


def _reap_stale_jobs_job():
    """APScheduler entrypoint for the hourly reaper."""
    db = SessionLocal()
    try:
        reap_stale_jobs(db)
    except Exception as e:
        logger.error("Stale-job reaper failed: %s", e)
    finally:
        db.close()


def warn_if_results_stale(db: Session) -> None:
    """Log a LOUD warning when the newest resulted race is older than
    yesterday (Dublin time).

    Deliberately does NOT auto-trigger a scrape: a crash-looping container
    would otherwise hammer GRI on every boot. The operator is pointed at
    POST /api/scraping/scrape-since-last-race-date instead.
    """
    last = (
        db.query(func.max(Race.race_date))
        .filter(Race.status == "resulted")
        .scalar()
    )
    yesterday = _dublin_today() - timedelta(days=1)
    if last is None:
        logger.warning(
            "No resulted races in the database — run a backfill "
            "(POST /api/scraping/backfill) to load history."
        )
    elif last < yesterday:
        logger.warning(
            "Results are stale: newest resulted race is %s (%d days behind "
            "yesterday %s). Catch up with "
            "POST /api/scraping/scrape-since-last-race-date.",
            last, (yesterday - last).days, yesterday,
        )


def _active_track_codes() -> list[str]:
    db = SessionLocal()
    try:
        return [t.code for t in db.query(Track).filter(Track.active.is_(True)).all()]
    finally:
        db.close()


def _scrape_results_for_day(day, source_desc: str) -> None:
    """Scrape one day's results for all active tracks via the job runner (E8)."""
    from scraping.gri_scraper import scrape_results
    from scraping.db_pipeline import upsert_race_results
    from scraping.job_runner import run_scrape_job

    try:
        result = run_scrape_job(
            SessionLocal,
            _active_track_codes(),
            [day],
            scrape_results,
            upsert_race_results,
            spider_name="gri",
            source_desc=source_desc,
            delay=2.0,
        )
        logger.info(
            "%s complete: %d races (%d new), %d failed pair(s)",
            source_desc, result["races_scraped"], result["races_new"],
            len(result["failed_pairs"]),
        )
    except Exception as e:
        logger.error("%s failed: %s", source_desc, e)


def _scrape_all_tracks_today():
    """Scrape today's results for all active tracks."""
    today = _dublin_today()
    _scrape_results_for_day(today, f"scheduled daily {today}")


def _scrape_yesterday_results():
    """Scrape yesterday's results (ensures we catch late-posted results)."""
    yesterday = _dublin_today() - timedelta(days=1)
    _scrape_results_for_day(yesterday, f"scheduled yesterday {yesterday}")


def _void_stale_races_job():
    """APScheduler entrypoint: void scheduled races whose date has long
    passed without results (audit task E10)."""
    from scraping.db_pipeline import void_stale_scheduled_races

    db = SessionLocal()
    try:
        voided = void_stale_scheduled_races(db)
        if voided:
            logger.warning("Void sweep: %d stale scheduled race(s) voided", voided)
    except Exception as e:
        logger.error("Void sweep failed: %s", e)
    finally:
        db.close()


def daily_digest(db: Session) -> dict:
    """Summarize the last 24h of scraping/training activity (audit task E12).

    Returned dict (also served by GET /api/scraping/digest):
      - scrape_jobs_by_status: ScrapeLog counts in the window, keyed by status
      - failed_pairs / total_failed_pairs: structured per-(track, date)
        failure rows (E7) with their error messages
      - races_scraped: records_scraped summed over finished jobs
      - stale_experiments: 'running' experiments with no recent heartbeat
      - anything_failed: True when any of the above indicates a problem
    """
    from app.models.experiment import Experiment

    now = datetime.utcnow()
    since = now - timedelta(hours=24)

    logs = db.query(ScrapeLog).filter(ScrapeLog.started_at >= since).all()

    jobs_by_status: dict[str, int] = {}
    failed_pairs: list[dict] = []
    races_scraped = 0
    for log in logs:
        jobs_by_status[log.status] = jobs_by_status.get(log.status, 0) + 1
        if log.status == "failed" and log.track_code and log.race_date:
            failed_pairs.append({
                "track_code": log.track_code,
                "race_date": str(log.race_date),
                "error": log.error_message,
            })
        if log.status in ("success", "partial"):
            races_scraped += log.records_scraped or 0

    cutoff = now - timedelta(minutes=STALE_JOB_MINUTES)
    stale_experiments = 0
    for exp in db.query(Experiment).filter(Experiment.status == "running").all():
        last_alive = exp.heartbeat_at or exp.created_at
        if last_alive is None or last_alive < cutoff:
            stale_experiments += 1

    anything_failed = bool(
        jobs_by_status.get("failed")
        or jobs_by_status.get("partial")
        or failed_pairs
        or stale_experiments
    )

    return {
        "generated_at": now.isoformat(),
        "window_hours": 24,
        "scrape_jobs_by_status": jobs_by_status,
        "total_failed_pairs": len(failed_pairs),
        "failed_pairs": failed_pairs,
        "races_scraped": races_scraped,
        "stale_experiments": stale_experiments,
        "anything_failed": anything_failed,
    }


def _daily_digest_job():
    """APScheduler entrypoint: log the daily digest and ship it to Sentry.

    WARNING when anything failed in the last 24h, INFO otherwise. The
    Sentry message is only sent when something failed — a quiet day should
    not page anyone.
    """
    db = SessionLocal()
    try:
        digest = daily_digest(db)
    except Exception as e:
        logger.error("Daily digest failed: %s", e)
        return
    finally:
        db.close()

    summary = (
        f"Daily scrape digest: jobs={digest['scrape_jobs_by_status']} "
        f"failed_pairs={digest['total_failed_pairs']} "
        f"races_scraped={digest['races_scraped']} "
        f"stale_experiments={digest['stale_experiments']}"
    )
    if digest["anything_failed"]:
        logger.warning(summary)
    else:
        logger.info(summary)

    if settings.sentry_dsn and digest["anything_failed"]:
        try:
            import sentry_sdk

            sentry_sdk.capture_message(summary, level="warning")
        except Exception as e:
            logger.error("Could not send daily digest to Sentry: %s", e)


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
    """Start the APScheduler with configured jobs.

    Respects the ENABLE_SCHEDULER setting: when false, logs and skips —
    useful for maintenance containers and local dev where in-process cron
    jobs are unwanted.
    """
    if not settings.enable_scheduler:
        logger.info("ENABLE_SCHEDULER is false — scheduler not started")
        return

    # Scrape today's results at 23:00 daily (Dublin wall-clock time —
    # racing schedules and GRI result posting follow Irish local time).
    scheduler.add_job(
        _scrape_all_tracks_today,
        trigger=CronTrigger(hour=23, minute=0, timezone="Europe/Dublin"),
        id="daily_results",
        name="Daily GRI results scrape",
        replace_existing=True,
    )

    # Scrape yesterday's results at 08:00 Dublin time (catch late posts)
    scheduler.add_job(
        _scrape_yesterday_results,
        trigger=CronTrigger(hour=8, minute=0, timezone="Europe/Dublin"),
        id="yesterday_results",
        name="Yesterday GRI results scrape",
        replace_existing=True,
    )

    # Hourly reaper for jobs whose worker died without cleanup.
    scheduler.add_job(
        _reap_stale_jobs_job,
        trigger=IntervalTrigger(hours=1),
        id="stale_job_reaper",
        name="Reap stale running jobs",
        replace_existing=True,
    )

    # Void sweep at 04:00 Dublin (quiet hour, after the late-night results
    # scrape): scheduled races whose date passed days ago never got results
    # and must not stay selectable for prediction (audit task E10).
    scheduler.add_job(
        _void_stale_races_job,
        trigger=CronTrigger(hour=4, minute=0, timezone="Europe/Dublin"),
        id="void_stale_races",
        name="Void stale scheduled races",
        replace_existing=True,
    )

    # Daily failure-visibility digest at 08:30 Dublin, after the 08:00
    # yesterday-results scrape so its outcome is included (audit task E12).
    scheduler.add_job(
        _daily_digest_job,
        trigger=CronTrigger(hour=8, minute=30, timezone="Europe/Dublin"),
        id="daily_digest",
        name="Daily scrape digest",
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

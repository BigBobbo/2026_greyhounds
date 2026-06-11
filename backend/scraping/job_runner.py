"""Single scrape-job runner (audit task E8).

All scrape jobs — manual API triggers, the scheduler's built-in daily jobs,
the CLI backfill and the schedule-service card pre-scrape — funnel through
``run_scrape_job``, which owns the bookkeeping that used to be copy-pasted
at five call sites:

  - ScrapeLog lifecycle: create (or adopt) the parent log, heartbeat every
    iteration, finish with success / partial / failed and a summary of the
    failed (track, date) pairs.
  - Per-(track, date) fault isolation: one bad day rolls back the track's
    session, is recorded, and the rest of the job continues.
  - One ScrapeLog row per failed (track, date) pair (audit task E7.3) with
    the structured race_date/track_code columns set, so POST
    /scraping/retry-failed can re-scrape exactly the failed pairs.
  - The politeness delay between requests (default settings.scrape_delay).
  - A fresh DB session per track so one track's poisoned session can't sink
    the others.

The runner is synchronous and creates its own event loop — call it from a
worker thread or any plain (non-async) context, never from inside a running
event loop.
"""

import asyncio
import logging
import traceback
from collections.abc import Awaitable, Callable, Iterable
from datetime import date, datetime
from typing import Any

import httpx

from app.config import settings
from app.models.scrape_log import ScrapeLog
from scraping.gri_scraper import DEFAULT_HEADERS

logger = logging.getLogger(__name__)

# scrape_fn(track_code, race_date, client) -> list of race dicts
ScrapeFn = Callable[[str, date, httpx.AsyncClient], Awaitable[list[dict[str, Any]]]]
# upsert_fn(db, races, scrape_log_id=...) -> stats dict
UpsertFn = Callable[..., dict[str, int]]

_STAT_KEYS = ("races_new", "races_updated", "entries_new", "entries_updated", "dogs_new")


def format_failed_pairs(failed: list[tuple[str, date]], limit: int = 20) -> str:
    """Summarize failed (track, date) pairs for ScrapeLog.error_message."""
    shown = ", ".join(f"{tc} {d}" for tc, d in failed[:limit])
    extra = f" (+{len(failed) - limit} more)" if len(failed) > limit else ""
    return f"Failed (track, date): {shown}{extra}"


def record_failed_pair(
    db_session_factory,
    *,
    spider_name: str,
    track_code: str,
    race_date: date,
    error_message: str,
    source: str | None = None,
) -> None:
    """Persist one failed-pair ScrapeLog row (audit task E7.3). Best-effort —
    bookkeeping failures must never take down the scrape job itself."""
    db = db_session_factory()
    try:
        now = datetime.utcnow()
        db.add(
            ScrapeLog(
                spider_name=spider_name,
                source=source,
                status="failed",
                race_date=race_date,
                track_code=track_code,
                error_message=error_message,
                started_at=now,
                completed_at=now,
            )
        )
        db.commit()
    except Exception:
        logger.exception("Could not record failed pair %s %s", track_code, race_date)
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


def run_scrape_job(
    db_session_factory,
    tracks: Iterable[str],
    dates: Iterable[date],
    scrape_fn: ScrapeFn,
    upsert_fn: UpsertFn,
    *,
    spider_name: str = "gri",
    source_desc: str,
    delay: float | None = None,
    log_extra: dict[str, Any] | None = None,
    log_id: int | None = None,
) -> dict[str, Any]:
    """Scrape every (track, date) pair and upsert the results.

    Parameters
    ----------
    db_session_factory: zero-arg callable returning a SQLAlchemy session
        (normally ``app.database.SessionLocal``).
    tracks / dates: track codes and dates; the cross product is scraped,
        track-by-track (fresh session per track), dates in order.
    scrape_fn: async ``(track_code, race_date, client) -> races`` (e.g.
        ``gri_scraper.scrape_results`` or ``scrape_card``).
    upsert_fn: ``(db, races, scrape_log_id=...) -> stats`` (normally
        ``db_pipeline.upsert_race_results``).
    spider_name / source_desc: stamped onto the parent ScrapeLog.
    delay: politeness sleep between requests; None = settings.scrape_delay.
    log_extra: extra column values for the parent ScrapeLog (e.g.
        ``{"race_date": d}``).
    log_id: adopt an existing 'running' ScrapeLog instead of creating one
        (API endpoints create the log up-front so they can return its id).

    Returns a dict: ``{"log_id", "status", "races_scraped", "races_new",
    "races_updated", "entries_new", "entries_updated", "dogs_new",
    "failed_pairs", "failed_tracks"}``.
    """
    delay = settings.scrape_delay if delay is None else delay
    track_codes = list(tracks)
    date_list = list(dates)

    # --- parent log: create, or adopt the one the API endpoint already made.
    db = db_session_factory()
    try:
        if log_id is None:
            log = ScrapeLog(
                spider_name=spider_name,
                source=source_desc,
                status="running",
                started_at=datetime.utcnow(),
            )
            db.add(log)
        else:
            log = db.get(ScrapeLog, log_id)
        if log is not None:
            if log.race_date is None and len(date_list) == 1:
                log.race_date = date_list[0]
            if log.track_code is None and len(track_codes) == 1:
                log.track_code = track_codes[0]
            for key, value in (log_extra or {}).items():
                setattr(log, key, value)
        db.commit()
        if log_id is None:
            db.refresh(log)
            log_id = log.id
    finally:
        db.close()

    totals = dict.fromkeys(_STAT_KEYS, 0)
    failed_pairs: list[tuple[str, date]] = []
    failed_tracks: list[str] = []

    def _touch_log(**fields) -> None:
        """Best-effort heartbeat / progress / final update of the parent log."""
        db_log = db_session_factory()
        try:
            entry = db_log.get(ScrapeLog, log_id)
            if entry is not None:
                entry.heartbeat_at = datetime.utcnow()
                for key, value in fields.items():
                    setattr(entry, key, value)
                db_log.commit()
        except Exception:
            try:
                db_log.rollback()
            except Exception:
                pass
        finally:
            db_log.close()

    async def _run() -> None:
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30
        ) as client:
            for t_idx, tc in enumerate(track_codes):
                db_track = db_session_factory()
                try:
                    for d_idx, d in enumerate(date_list):
                        try:
                            races = await scrape_fn(tc, d, client)
                            if races:
                                stats = upsert_fn(db_track, races, scrape_log_id=log_id)
                                for key in _STAT_KEYS:
                                    totals[key] += stats.get(key, 0)
                        except Exception as e:
                            logger.error("Scrape failed for %s %s: %s", tc, d, e)
                            # The session may hold a half-applied upsert —
                            # roll it back so later days start clean.
                            try:
                                db_track.rollback()
                            except Exception:
                                pass
                            failed_pairs.append((tc, d))
                            record_failed_pair(
                                db_session_factory,
                                spider_name=spider_name,
                                track_code=tc,
                                race_date=d,
                                error_message=f"{type(e).__name__}: {e}",
                                source=source_desc,
                            )
                        _touch_log()
                        last_iteration = (
                            t_idx == len(track_codes) - 1 and d_idx == len(date_list) - 1
                        )
                        if delay and not last_iteration:
                            await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(
                        "Track %s crashed: %s\n%s", tc, e, traceback.format_exc()
                    )
                    failed_tracks.append(tc)
                finally:
                    db_track.close()
                _touch_log(
                    records_scraped=totals["races_new"] + totals["races_updated"],
                    records_new=totals["races_new"],
                )

    status = "success"
    error_message: str | None = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()
        parts = []
        if failed_tracks:
            parts.append(f"Failed tracks: {', '.join(failed_tracks)}")
        if failed_pairs:
            parts.append(format_failed_pairs(failed_pairs))
        if parts:
            status = "partial"
            error_message = "; ".join(parts)
    except Exception as e:
        logger.error(
            "Scrape job %r crashed: %s\n%s", source_desc, e, traceback.format_exc()
        )
        status = "failed"
        error_message = f"{type(e).__name__}: {e}"

    db_final = db_session_factory()
    try:
        entry = db_final.get(ScrapeLog, log_id)
        if entry is not None:
            entry.status = status
            entry.records_scraped = totals["races_new"] + totals["races_updated"]
            entry.records_new = totals["races_new"]
            entry.error_message = error_message
            entry.completed_at = datetime.utcnow()
            db_final.commit()
    finally:
        db_final.close()

    logger.info(
        "Scrape job %r finished: status=%s races=%d (%d new), %d failed pair(s)",
        source_desc, status, totals["races_new"] + totals["races_updated"],
        totals["races_new"], len(failed_pairs),
    )

    return {
        "log_id": log_id,
        "status": status,
        "races_scraped": totals["races_new"] + totals["races_updated"],
        **totals,
        "failed_pairs": failed_pairs,
        "failed_tracks": failed_tracks,
    }

"""Scraping management API endpoints — uses httpx (no Playwright)."""

import asyncio
import logging
import traceback
from datetime import date, datetime, timedelta
from threading import Thread
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.scrape_log import ScrapeLog
from app.models.track import Track
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.dog import Dog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scraping", tags=["scraping"])


class ScrapeLogResponse(BaseModel):
    id: int
    spider_name: str
    source: str | None
    status: str
    records_scraped: int
    records_new: int
    records_updated: int
    error_message: str | None
    started_at: datetime | None
    heartbeat_at: datetime | None
    completed_at: datetime | None
    model_config = {"from_attributes": True}


class ScrapingStatusResponse(BaseModel):
    total_races: int
    total_entries: int
    total_dogs: int
    total_tracks: int
    last_scrape: ScrapeLogResponse | None
    recent_logs: list[ScrapeLogResponse]


class ReapStaleRequest(BaseModel):
    stale_minutes: int = 15


class ReapStaleResponse(BaseModel):
    reaped: int
    log_ids: list[int]


class TriggerRequest(BaseModel):
    track_code: str
    date_from: str
    date_to: str | None = None


class BackfillRequest(BaseModel):
    start_date: str
    end_date: str
    track_codes: list[str] | None = None


class ScrapeDateRequest(BaseModel):
    race_date: str
    include_form_detail: bool = False
    track_codes: list[str] | None = None


class ScrapeSinceLastRequest(BaseModel):
    end_date: str | None = None
    track_codes: list[str] | None = None


class LastScrapeInfoResponse(BaseModel):
    last_race_date: str | None
    proposed_start_date: str | None
    today: str
    days_to_scrape: int
    active_track_count: int


class TriggerResponse(BaseModel):
    message: str
    log_id: int | None = None


class CardsStatusTrack(BaseModel):
    code: str
    name: str
    race_count: int
    scheduled_count: int
    resulted_count: int
    last_scraped_at: datetime | None


class CardsStatusLog(BaseModel):
    id: int
    status: str
    records_scraped: int
    records_new: int
    started_at: datetime | None
    completed_at: datetime | None


class CardsStatusResponse(BaseModel):
    race_date: str
    total_races: int
    tracks: list[CardsStatusTrack]
    recent_scrape_logs: list[CardsStatusLog]


@router.get("/status", response_model=ScrapingStatusResponse)
def get_scraping_status(db: Session = Depends(get_db)):
    total_races = db.query(Race).count()
    total_entries = db.query(RaceEntry).count()
    total_dogs = db.query(Dog).count()
    total_tracks = db.query(Track).filter(Track.active.is_(True)).count()
    last_scrape = db.query(ScrapeLog).order_by(ScrapeLog.id.desc()).first()
    recent_logs = db.query(ScrapeLog).order_by(ScrapeLog.id.desc()).limit(20).all()

    return ScrapingStatusResponse(
        total_races=total_races,
        total_entries=total_entries,
        total_dogs=total_dogs,
        total_tracks=total_tracks,
        last_scrape=ScrapeLogResponse.model_validate(last_scrape) if last_scrape else None,
        recent_logs=[ScrapeLogResponse.model_validate(log) for log in recent_logs],
    )


@router.get("/cards-status", response_model=CardsStatusResponse)
def get_cards_status(race_date: str, db: Session = Depends(get_db)):
    """Show which tracks already have race cards scraped for a given date.

    Used by the predictions UI to indicate which tracks are already scraped
    so users don't re-scrape the same future race state repeatedly.
    """
    rd = date.fromisoformat(race_date)

    rows = (
        db.query(Race.status, Race.last_scraped_at, Track.code, Track.name)
        .join(Track, Track.id == Race.track_id)
        .filter(Race.race_date == rd)
        .all()
    )

    by_code: dict[str, dict[str, Any]] = {}
    for status, last_scraped_at, code, name in rows:
        bucket = by_code.setdefault(
            code,
            {
                "code": code,
                "name": name,
                "race_count": 0,
                "scheduled_count": 0,
                "resulted_count": 0,
                "last_scraped_at": None,
            },
        )
        bucket["race_count"] += 1
        if status == "scheduled":
            bucket["scheduled_count"] += 1
        elif status == "resulted":
            bucket["resulted_count"] += 1
        if last_scraped_at and (
            bucket["last_scraped_at"] is None
            or last_scraped_at > bucket["last_scraped_at"]
        ):
            bucket["last_scraped_at"] = last_scraped_at

    tracks = [
        CardsStatusTrack(**bucket)
        for bucket in sorted(by_code.values(), key=lambda b: b["name"])
    ]

    logs = (
        db.query(ScrapeLog)
        .filter(
            ScrapeLog.spider_name == "gri-card",
            ScrapeLog.source.like(f"%{rd}%"),
        )
        .order_by(ScrapeLog.id.desc())
        .limit(5)
        .all()
    )
    log_responses = [
        CardsStatusLog(
            id=l.id,
            status=l.status,
            records_scraped=l.records_scraped or 0,
            records_new=l.records_new or 0,
            started_at=l.started_at,
            completed_at=l.completed_at,
        )
        for l in logs
    ]

    return CardsStatusResponse(
        race_date=str(rd),
        total_races=sum(t.race_count for t in tracks),
        tracks=tracks,
        recent_scrape_logs=log_responses,
    )


@router.post("/reap-stale", response_model=ReapStaleResponse)
def reap_stale_scrape_logs(
    req: ReapStaleRequest, db: Session = Depends(get_db)
):
    """Mark scrape logs stuck in 'running' as failed.

    A log is considered stale if its heartbeat_at (or started_at, if no
    heartbeat was ever recorded) is older than `stale_minutes` ago.
    Worker threads update heartbeat_at while making progress, so a stale
    log indicates the worker died (process restart, OOM, exception above
    the cleanup handler).
    """
    cutoff = datetime.utcnow() - timedelta(minutes=req.stale_minutes)
    stale = (
        db.query(ScrapeLog)
        .filter(
            ScrapeLog.status == "running",
            ScrapeLog.completed_at.is_(None),
        )
        .all()
    )
    reaped: list[int] = []
    now = datetime.utcnow()
    for log in stale:
        last_alive = log.heartbeat_at or log.started_at
        if last_alive is None or last_alive < cutoff:
            log.status = "failed"
            log.completed_at = now
            existing_msg = log.error_message or ""
            reap_note = (
                f"Reaped as stale: no heartbeat since "
                f"{last_alive.isoformat() if last_alive else 'never'}"
            )
            log.error_message = (
                f"{existing_msg}; {reap_note}" if existing_msg else reap_note
            )
            reaped.append(log.id)
    if reaped:
        db.commit()
    return ReapStaleResponse(reaped=len(reaped), log_ids=reaped)


@router.get("/test-scrape")
async def test_scrape(track_code: str = "TRL", date_str: str = "04-Apr-2026"):
    """Scrape one date with httpx, save to DB, return results."""
    from scraping.gri_scraper import scrape_results, parse_results_page, VIEW_RESULTS_URL, format_date, DEFAULT_HEADERS
    from scraping.db_pipeline import upsert_race_results
    from datetime import datetime as dt

    race_date = dt.strptime(date_str, "%d-%b-%Y").date()
    url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
        try:
            resp = await client.get(url)
        except Exception as e:
            return {"error": f"HTTP request failed: {e}", "url": url}

    if resp.status_code != 200:
        return {"error": f"Got status {resp.status_code}", "url": url}

    races = parse_results_page(resp.text, track_code, race_date)

    result: dict[str, Any] = {
        "url": url,
        "status_code": resp.status_code,
        "html_length": len(resp.text),
        "races_parsed": len(races),
    }

    if not races:
        return result

    # Save to DB
    db = SessionLocal()
    try:
        stats = upsert_race_results(db, races)
        result["db_stats"] = stats
        result["first_race"] = {
            "number": races[0]["race_number"],
            "grade": races[0]["grade"],
            "distance": races[0]["distance_m"],
            "entries": len(races[0]["entries"]),
            "first_dog": races[0]["entries"][0].get("dog_name") if races[0]["entries"] else None,
            "first_trap": races[0]["entries"][0].get("trap") if races[0]["entries"] else None,
        }
    except Exception as e:
        db.rollback()
        result["db_error"] = str(e)
    finally:
        db.close()

    return result


def _format_failed_days(failed: list[str], limit: int = 20) -> str:
    """Summarize failed (track, date) pairs for ScrapeLog.error_message."""
    shown = ", ".join(failed[:limit])
    extra = f" (+{len(failed) - limit} more)" if len(failed) > limit else ""
    return f"Failed (track, date): {shown}{extra}"


def _heartbeat(log_id: int) -> None:
    """Mark a running ScrapeLog as alive. Best-effort — swallows errors."""
    db_log = SessionLocal()
    try:
        log_entry = db_log.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
        if log_entry:
            log_entry.heartbeat_at = datetime.utcnow()
            db_log.commit()
    except Exception:
        pass
    finally:
        db_log.close()


def _run_scrape_in_thread(track_code: str, date_from: date, date_to: date, log_id: int):
    """Run scraping in a background thread using httpx."""
    from scraping.gri_scraper import scrape_results, DEFAULT_HEADERS
    from scraping.db_pipeline import upsert_race_results

    def _scrape_sync():
        import asyncio as _asyncio

        db = SessionLocal()
        log = db.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
        total_races = 0
        total_new = 0
        failed_days: list[str] = []

        async def _run():
            nonlocal total_races, total_new
            async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
                current = date_from
                while current <= date_to:
                    try:
                        races = await scrape_results(track_code, current, client)
                        if races:
                            stats = upsert_race_results(db, races, scrape_log_id=log_id)
                            total_races += stats["races_new"] + stats["races_updated"]
                            total_new += stats["races_new"]
                    except Exception as e:
                        logger.error("Error on %s %s: %s", track_code, current, e)
                        failed_days.append(f"{track_code} {current}")
                    _heartbeat(log_id)
                    current += timedelta(days=1)
                    if current <= date_to:
                        await _asyncio.sleep(1.0)

        try:
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            loop.run_until_complete(_run())
            loop.close()
            if failed_days:
                log.status = "partial"
                log.error_message = _format_failed_days(failed_days)
            else:
                log.status = "success"
            log.records_scraped = total_races
            log.records_new = total_new
        except Exception as e:
            logger.error("Scrape thread failed: %s\n%s", e, traceback.format_exc())
            log.status = "failed"
            log.error_message = f"{type(e).__name__}: {e}"
        finally:
            log.completed_at = datetime.utcnow()
            db.commit()
            db.close()

    Thread(target=_scrape_sync, daemon=True).start()


@router.post("/trigger", response_model=TriggerResponse)
def trigger_scrape(req: TriggerRequest, db: Session = Depends(get_db)):
    """Trigger a scrape for a specific track and date range."""
    track = db.query(Track).filter(Track.code == req.track_code).first()
    if not track:
        return TriggerResponse(message=f"Unknown track code: {req.track_code}")

    date_from = date.fromisoformat(req.date_from)
    date_to = date.fromisoformat(req.date_to) if req.date_to else date_from

    log = ScrapeLog(
        spider_name="gri",
        source=f"manual {req.track_code} {date_from} to {date_to}",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    _run_scrape_in_thread(req.track_code, date_from, date_to, log.id)

    return TriggerResponse(
        message=f"Scraping started for {req.track_code} from {date_from} to {date_to}",
        log_id=log.id,
    )


@router.post("/backfill", response_model=TriggerResponse)
def trigger_backfill(req: BackfillRequest, db: Session = Depends(get_db)):
    """Trigger a historical backfill in the background."""
    try:
        start = date.fromisoformat(req.start_date)
        end = date.fromisoformat(req.end_date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid date: {e}")

    if end < start:
        raise HTTPException(status_code=422, detail="end_date is before start_date")
    if (end - start).days > 366:
        raise HTTPException(
            status_code=422,
            detail="Backfill range capped at 366 days per request — split "
            "longer ranges into multiple calls.",
        )

    # Refuse to start while another scrape job is still running: overlapping
    # jobs double-hit GRI and race each other on upserts.
    running = (
        db.query(ScrapeLog)
        .filter(ScrapeLog.status == "running")
        .order_by(ScrapeLog.id.desc())
        .first()
    )
    if running:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Scrape job {running.id} ({running.source}) is still running. "
                "Wait for it to finish or reap stale jobs first."
            ),
        )

    if req.track_codes:
        tracks = db.query(Track).filter(Track.code.in_(req.track_codes)).all()
        unknown = set(req.track_codes) - {t.code for t in tracks}
        if unknown:
            raise HTTPException(
                status_code=404, detail=f"Unknown track codes: {sorted(unknown)}"
            )
    else:
        tracks = db.query(Track).filter(Track.active.is_(True)).all()

    if not tracks:
        raise HTTPException(status_code=404, detail="No tracks found")

    log = ScrapeLog(
        spider_name="gri",
        source=f"backfill {len(tracks)} tracks {start} to {end}",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    log_id = log.id
    track_codes = [t.code for t in tracks]

    def _run_backfill():
        """Run backfill one track at a time. Each track gets its own DB session and async loop."""
        import traceback
        from scraping.gri_scraper import scrape_results, DEFAULT_HEADERS
        from scraping.db_pipeline import upsert_race_results
        import asyncio as _asyncio

        total_races = 0
        total_new = 0
        failed_tracks = []
        failed_days: list[str] = []

        for tc in track_codes:
            track_races = 0
            track_new = 0

            async def _scrape_track():
                nonlocal track_races, track_new
                async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
                    db_track = SessionLocal()
                    try:
                        current = start
                        day_count = 0
                        while current <= end:
                            day_count += 1
                            try:
                                races = await scrape_results(tc, current, client)
                                if races:
                                    stats = upsert_race_results(db_track, races, scrape_log_id=log_id)
                                    track_races += stats["races_new"] + stats["races_updated"]
                                    track_new += stats["races_new"]
                            except Exception as e:
                                logger.error("Error %s %s: %s", tc, current, e)
                                failed_days.append(f"{tc} {current}")

                            # Commit + update log every 50 days
                            if day_count % 50 == 0:
                                db_track.commit()
                                _update_log(total_races + track_races, total_new + track_new)
                            else:
                                _heartbeat(log_id)

                            current += timedelta(days=1)
                            await _asyncio.sleep(1.0)

                        db_track.commit()
                    except Exception as e:
                        logger.error("Track %s failed: %s", tc, e)
                        db_track.rollback()
                        raise
                    finally:
                        db_track.close()

            try:
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)
                loop.run_until_complete(_scrape_track())
                loop.close()

                total_races += track_races
                total_new += track_new
                logger.info("Backfill: %s done — %d races (%d new). Total: %d", tc, track_races, track_new, total_races)
                _update_log(total_races, total_new)

            except Exception as e:
                logger.error("Backfill track %s crashed: %s\n%s", tc, e, traceback.format_exc())
                failed_tracks.append(tc)
                # Continue with next track

        # Final update
        db_final = SessionLocal()
        try:
            log_final = db_final.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_final:
                log_final.status = "success" if not (failed_tracks or failed_days) else "partial"
                log_final.records_scraped = total_races
                log_final.records_new = total_new
                log_final.completed_at = datetime.utcnow()
                error_parts = []
                if failed_tracks:
                    error_parts.append(f"Failed tracks: {', '.join(failed_tracks)}")
                if failed_days:
                    error_parts.append(_format_failed_days(failed_days))
                if error_parts:
                    log_final.error_message = "; ".join(error_parts)
                db_final.commit()
        finally:
            db_final.close()
        logger.info(
            "Backfill complete: %d races, %d new. Failed tracks: %s. Failed days: %d",
            total_races, total_new, failed_tracks or "none", len(failed_days),
        )

    def _update_log(scraped: int, new: int):
        """Update the scrape log with current progress and heartbeat."""
        db_log = SessionLocal()
        try:
            log_entry = db_log.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_entry:
                log_entry.records_scraped = scraped
                log_entry.records_new = new
                log_entry.heartbeat_at = datetime.utcnow()
                db_log.commit()
        except Exception:
            pass
        finally:
            db_log.close()

    Thread(target=_run_backfill, daemon=True).start()

    return TriggerResponse(
        message=f"Backfill started for {len(tracks)} tracks from {start} to {end}",
        log_id=log.id,
    )


@router.get("/last-scrape-info", response_model=LastScrapeInfoResponse)
def get_last_scrape_info(db: Session = Depends(get_db)):
    """Preview info for the 'scrape since last race date' action.

    Returns the most recent resulted race date in the DB (NOT the timestamp
    of the last scrape job), the proposed start date (last + 1), today, and
    how many days/tracks the scrape would cover.
    """
    from sqlalchemy import func

    last_date = (
        db.query(func.max(Race.race_date))
        .filter(Race.status == "resulted")
        .scalar()
    )
    today = date.today()
    proposed_start = (last_date + timedelta(days=1)) if last_date else None
    if proposed_start and proposed_start <= today:
        days = (today - proposed_start).days + 1
    else:
        days = 0
    active = db.query(Track).filter(Track.active.is_(True)).count()

    return LastScrapeInfoResponse(
        last_race_date=str(last_date) if last_date else None,
        proposed_start_date=str(proposed_start) if proposed_start else None,
        today=str(today),
        days_to_scrape=days,
        active_track_count=active,
    )


@router.post("/scrape-since-last-race-date", response_model=TriggerResponse)
def trigger_scrape_since_last_race_date(
    req: ScrapeSinceLastRequest, db: Session = Depends(get_db)
):
    """Discover and scrape all races since the latest race date in the DB.

    Start date is computed as max(Race.race_date) + 1 — i.e. the day after
    the most recent race already stored, NOT the day of the last scrape job.
    Runs a backfill up to `end_date` (default: today) across all active
    tracks (or selected `track_codes`).
    """
    from sqlalchemy import func

    # Per-track last dates: a single global max would skip any track whose
    # coverage lags behind the most recently scraped one (its gap could
    # never heal through this endpoint).
    track_query = db.query(Track).filter(Track.active.is_(True))
    if req.track_codes:
        track_query = db.query(Track).filter(Track.code.in_(req.track_codes))
    tracks = track_query.all()
    if not tracks:
        raise HTTPException(status_code=404, detail="No matching tracks")

    per_track_last = dict(
        db.query(Race.track_id, func.max(Race.race_date))
        .filter(Race.status == "resulted")
        .group_by(Race.track_id)
        .all()
    )
    if not per_track_last:
        return TriggerResponse(
            message="No prior scraped races found. Use /backfill with explicit dates instead."
        )

    end = date.fromisoformat(req.end_date) if req.end_date else date.today()

    # Scrape each track from the day after ITS own last resulted race.
    # Tracks with no coverage at all are skipped (use /backfill for those).
    stale_tracks: list[tuple[Track, date]] = []
    for t in tracks:
        last = per_track_last.get(t.id)
        if last is None:
            continue
        start = last + timedelta(days=1)
        if start <= end:
            stale_tracks.append((t, start))

    if not stale_tracks:
        return TriggerResponse(message="Already up to date for all covered tracks.")

    earliest = min(s for _, s in stale_tracks)
    ranges = ", ".join(
        f"{t.code} from {s.isoformat()}" for t, s in sorted(stale_tracks, key=lambda x: x[1])
    )
    backfill_req = BackfillRequest(
        start_date=earliest.isoformat(),
        end_date=end.isoformat(),
        track_codes=[t.code for t, _ in stale_tracks],
    )
    resp = trigger_backfill(backfill_req, db)
    resp.message = f"{resp.message} Per-track catch-up: {ranges}."
    return resp


def _run_scrape_date_in_thread(
    race_date_val: date,
    track_codes: list[str],
    include_form_detail: bool,
    log_id: int,
):
    """Brute-force scrape all upcoming-race-card summaries for a single date.

    Iterates one track at a time. For each track that returns races on the date,
    optionally fires the per-race form-detail page to enrich dog metadata
    (trainer/sire/dam/owner/best_time).
    """
    from scraping.gri_scraper import (
        scrape_card,
        scrape_card_form,
        merge_card_form_into_race,
        DEFAULT_HEADERS,
    )
    from scraping.db_pipeline import upsert_race_results

    def _run_sync():
        import asyncio as _asyncio

        total_races = 0
        total_new = 0
        tracks_with_races: list[str] = []
        tracks_failed: list[str] = []

        async def _run():
            nonlocal total_races, total_new
            async with httpx.AsyncClient(
                headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30
            ) as client:
                for tc in track_codes:
                    try:
                        races = await scrape_card(tc, race_date_val, client)
                    except Exception as e:
                        logger.error("Card scrape error %s %s: %s", tc, race_date_val, e)
                        tracks_failed.append(tc)
                        await _asyncio.sleep(1.0)
                        continue

                    if not races:
                        await _asyncio.sleep(1.0)
                        continue

                    tracks_with_races.append(tc)

                    if include_form_detail:
                        for r in races:
                            try:
                                form = await scrape_card_form(
                                    tc, race_date_val, r["race_number"], client
                                )
                                if form:
                                    merge_card_form_into_race(r, form)
                            except Exception as e:
                                logger.error(
                                    "Form scrape %s R%s failed: %s",
                                    tc, r.get("race_number"), e,
                                )
                            await _asyncio.sleep(0.5)

                    db_local = SessionLocal()
                    try:
                        stats = upsert_race_results(db_local, races, scrape_log_id=log_id)
                        total_races += stats["races_new"] + stats["races_updated"]
                        total_new += stats["races_new"]
                    except Exception as e:
                        logger.error("DB upsert failed for %s: %s", tc, e)
                        db_local.rollback()
                    finally:
                        db_local.close()

                    # progress update
                    db_log = SessionLocal()
                    try:
                        log_entry = db_log.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
                        if log_entry:
                            log_entry.records_scraped = total_races
                            log_entry.records_new = total_new
                            log_entry.heartbeat_at = datetime.utcnow()
                            db_log.commit()
                    finally:
                        db_log.close()

                    await _asyncio.sleep(1.0)

        try:
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            loop.run_until_complete(_run())
            loop.close()
            db_final = SessionLocal()
            try:
                log_final = (
                    db_final.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
                )
                if log_final:
                    log_final.status = "success" if not tracks_failed else "partial"
                    log_final.records_scraped = total_races
                    log_final.records_new = total_new
                    log_final.completed_at = datetime.utcnow()
                    parts = [
                        f"tracks_with_races={','.join(tracks_with_races) or 'none'}",
                    ]
                    if tracks_failed:
                        parts.append(
                            "failed (track, date): "
                            + ", ".join(f"{tc} {race_date_val}" for tc in tracks_failed)
                        )
                    log_final.error_message = "; ".join(parts) if tracks_failed else None
                    db_final.commit()
            finally:
                db_final.close()
        except Exception as e:
            logger.error("Scrape-date thread crashed: %s\n%s", e, traceback.format_exc())
            db_err = SessionLocal()
            try:
                log_err = db_err.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
                if log_err:
                    log_err.status = "failed"
                    log_err.error_message = f"{type(e).__name__}: {e}"
                    log_err.completed_at = datetime.utcnow()
                    db_err.commit()
            finally:
                db_err.close()

    Thread(target=_run_sync, daemon=True).start()


@router.post("/scrape-date", response_model=TriggerResponse)
def trigger_scrape_date(req: ScrapeDateRequest, db: Session = Depends(get_db)):
    """Brute-force scrape upcoming race cards for every active track on a date.

    Use `include_form_detail=true` to also pull per-race form pages (trainer,
    sire, dam, owner, best track time). Costs ~8-12 extra requests per meeting.
    """
    race_date_val = date.fromisoformat(req.race_date)

    if req.track_codes:
        tracks = db.query(Track).filter(Track.code.in_(req.track_codes)).all()
    else:
        tracks = db.query(Track).filter(Track.active.is_(True)).all()

    if not tracks:
        return TriggerResponse(message="No tracks found")

    track_codes = [t.code for t in tracks]

    suffix = " (with form detail)" if req.include_form_detail else ""
    log = ScrapeLog(
        spider_name="gri-card",
        source=f"scrape-date {race_date_val} {len(track_codes)} tracks{suffix}",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    _run_scrape_date_in_thread(
        race_date_val, track_codes, req.include_form_detail, log.id
    )

    return TriggerResponse(
        message=(
            f"Card scrape started for {race_date_val} across "
            f"{len(track_codes)} tracks{suffix}"
        ),
        log_id=log.id,
    )


@router.post("/discover-tracks")
def discover_tracks():
    """Return known GRI track codes."""
    from scraping.gri_scraper import GRI_TRACK_CODES
    return {"tracks": [{"code": k, "name": v} for k, v in GRI_TRACK_CODES.items()]}


@router.get("/test-track-scrape")
async def test_track_scrape(
    track_code: str = "TRL",
    date_str: str = "04-Apr-2026",
):
    """
    Test scraping a single track+date and show detailed diagnostics.
    Helps debug why a track might be failing.
    """
    from scraping.gri_scraper import VIEW_RESULTS_URL, DEFAULT_HEADERS, parse_results_page
    from datetime import datetime as dt

    race_date = dt.strptime(date_str, "%d-%b-%Y").date()
    url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
        try:
            resp = await client.get(url)
        except Exception as e:
            return {"error": f"HTTP failed: {e}", "url": url}

    if resp.status_code != 200:
        return {"error": f"Status {resp.status_code}", "url": url}

    races = parse_results_page(resp.text, track_code, race_date)

    return {
        "url": url,
        "status_code": resp.status_code,
        "html_length": len(resp.text),
        "has_race_data": "Race 1" in resp.text,
        "races_parsed": len(races),
        "entries_total": sum(len(r.get("entries", [])) for r in races),
    }


@router.get("/coverage-calendar")
def coverage_calendar(
    start_date: str | None = None,
    end_date: str | None = None,
    track_code: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Per-day race coverage for a GitHub-style calendar heatmap.

    Returns one entry per date with at least one race in the range. When
    `track_code` is given, counts only races at that track; otherwise counts
    races across all tracks and reports which tracks raced that day.

    Defaults to the last 365 days ending today.
    """
    from sqlalchemy import func as sqlfunc

    today = date.today()
    ed = date.fromisoformat(end_date) if end_date else today
    sd = date.fromisoformat(start_date) if start_date else (ed - timedelta(days=365))

    q = (
        db.query(Race.race_date, Track.code, sqlfunc.count(Race.id).label("races"))
        .join(Track, Track.id == Race.track_id)
        .filter(Race.race_date >= sd, Race.race_date <= ed)
    )
    if track_code:
        q = q.filter(Track.code == track_code)
    rows = q.group_by(Race.race_date, Track.code).all()

    by_day: dict[str, dict[str, Any]] = {}
    for rd, code, races in rows:
        key = str(rd)
        bucket = by_day.setdefault(
            key, {"date": key, "tracks": set(), "race_count": 0}
        )
        bucket["tracks"].add(code)
        bucket["race_count"] += races

    days = [
        {
            "date": d["date"],
            "race_count": d["race_count"],
            "track_count": len(d["tracks"]),
            "tracks": sorted(d["tracks"]),
        }
        for d in by_day.values()
    ]
    days.sort(key=lambda x: x["date"])

    return {
        "start_date": str(sd),
        "end_date": str(ed),
        "track_code": track_code,
        "days": days,
    }


@router.get("/data-summary")
def data_summary(db: Session = Depends(get_db)):
    """Show how much data we have per track — helps identify gaps."""
    from sqlalchemy import func

    rows = (
        db.query(
            Track.code,
            Track.name,
            func.count(Race.id).label("race_count"),
            func.min(Race.race_date).label("earliest"),
            func.max(Race.race_date).label("latest"),
        )
        .outerjoin(Race, Track.id == Race.track_id)
        .group_by(Track.code, Track.name)
        .order_by(func.count(Race.id).desc())
        .all()
    )

    tracks = []
    for row in rows:
        tracks.append({
            "code": row.code,
            "name": row.name,
            "race_count": row.race_count,
            "earliest": str(row.earliest) if row.earliest else None,
            "latest": str(row.latest) if row.latest else None,
        })

    total_races = db.query(Race).count()
    total_entries = db.query(RaceEntry).count()
    total_dogs = db.query(Dog).count()

    return {
        "total_races": total_races,
        "total_entries": total_entries,
        "total_dogs": total_dogs,
        "tracks": tracks,
    }


@router.get("/verify-coverage")
def verify_coverage(
    start_date: str | None = None,
    end_date: str | None = None,
    max_gap_days: int = 14,
    db: Session = Depends(get_db),
):
    """
    Verify that scraping is complete across all active tracks.

    Returns a per-track breakdown showing:
    - Whether the track has data at all
    - Date range covered
    - Any suspicious gaps (>max_gap_days with no races)
    - An overall verdict: "complete", "gaps_found", or "missing_tracks"

    Use this before materializing features to confirm all track data is present.
    """
    from sqlalchemy import func as sqlfunc
    from ml.data_integrity import find_coverage_gaps, get_track_date_coverage

    # Determine date range
    if start_date and end_date:
        sd = date.fromisoformat(start_date)
        ed = date.fromisoformat(end_date)
    else:
        date_range = (
            db.query(sqlfunc.min(Race.race_date), sqlfunc.max(Race.race_date))
            .filter(Race.status == "resulted")
            .first()
        )
        if not date_range or not date_range[0]:
            return {"verdict": "no_data", "message": "No resulted races in database"}
        sd, ed = date_range

    active_tracks = (
        db.query(Track.code, Track.name, Track.id)
        .filter(Track.active.is_(True))
        .order_by(Track.name)
        .all()
    )
    coverage = get_track_date_coverage(db, sd, ed)
    gaps = find_coverage_gaps(db, sd, ed, max_gap_days)

    # Build gap lookup by track code
    gaps_by_track: dict[str, list] = {}
    for g in gaps:
        gaps_by_track.setdefault(g["track_code"], []).append({
            "from": str(g["gap_start"]),
            "to": str(g["gap_end"]),
            "days": g["gap_days"],
        })

    # Per-track report
    track_reports = []
    missing_tracks = []
    tracks_with_gaps = []

    for code, name, tid in active_tracks:
        dates = coverage.get(code, [])
        race_count = (
            db.query(sqlfunc.count(Race.id))
            .filter(Race.track_id == tid, Race.status == "resulted",
                    Race.race_date >= sd, Race.race_date <= ed)
            .scalar() or 0
        )

        if not dates:
            missing_tracks.append(code)
            track_reports.append({
                "code": code,
                "name": name,
                "status": "no_data",
                "race_count": 0,
                "earliest": None,
                "latest": None,
                "gaps": [],
            })
        else:
            track_gaps = gaps_by_track.get(code, [])
            status = "gaps" if track_gaps else "ok"
            if track_gaps:
                tracks_with_gaps.append(code)
            track_reports.append({
                "code": code,
                "name": name,
                "status": status,
                "race_count": race_count,
                "earliest": str(dates[0]),
                "latest": str(dates[-1]),
                "gaps": track_gaps,
            })

    # Overall verdict
    if missing_tracks:
        verdict = "missing_tracks"
        message = (
            f"{len(missing_tracks)} track(s) have NO data in "
            f"{sd} to {ed}: {', '.join(missing_tracks)}. "
            "Run a backfill for these tracks before materializing features."
        )
    elif tracks_with_gaps:
        verdict = "gaps_found"
        total_gaps = sum(len(g) for g in gaps_by_track.values())
        message = (
            f"{total_gaps} gap(s) found across {len(tracks_with_gaps)} track(s). "
            "Some dogs' histories may be incomplete. Consider backfilling "
            f"these tracks: {', '.join(tracks_with_gaps)}"
        )
    else:
        verdict = "complete"
        message = (
            f"All {len(active_tracks)} active tracks have continuous coverage "
            f"from {sd} to {ed} (no gaps > {max_gap_days} days)."
        )

    return {
        "verdict": verdict,
        "message": message,
        "date_range": {"start": str(sd), "end": str(ed)},
        "max_gap_days": max_gap_days,
        "summary": {
            "total_tracks": len(active_tracks),
            "tracks_ok": len(active_tracks) - len(missing_tracks) - len(tracks_with_gaps),
            "tracks_with_gaps": len(tracks_with_gaps),
            "tracks_missing": len(missing_tracks),
        },
        "tracks": track_reports,
    }

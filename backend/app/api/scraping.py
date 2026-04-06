"""Scraping management API endpoints — uses httpx (no Playwright)."""

import asyncio
import logging
import traceback
from datetime import date, datetime, timedelta
from threading import Thread
from typing import Any

import httpx
from fastapi import APIRouter, Depends
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
    completed_at: datetime | None
    model_config = {"from_attributes": True}


class ScrapingStatusResponse(BaseModel):
    total_races: int
    total_entries: int
    total_dogs: int
    total_tracks: int
    last_scrape: ScrapeLogResponse | None
    recent_logs: list[ScrapeLogResponse]


class TriggerRequest(BaseModel):
    track_code: str
    date_from: str
    date_to: str | None = None


class BackfillRequest(BaseModel):
    start_date: str
    end_date: str
    track_codes: list[str] | None = None


class TriggerResponse(BaseModel):
    message: str
    log_id: int | None = None


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

        async def _run():
            nonlocal total_races, total_new
            async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
                current = date_from
                while current <= date_to:
                    try:
                        races = await scrape_results(track_code, current, client)
                        if races:
                            stats = upsert_race_results(db, races)
                            total_races += stats["races_new"] + stats["races_updated"]
                            total_new += stats["races_new"]
                    except Exception as e:
                        logger.error("Error on %s %s: %s", track_code, current, e)
                    current += timedelta(days=1)
                    if current <= date_to:
                        await _asyncio.sleep(1.0)

        try:
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            loop.run_until_complete(_run())
            loop.close()
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
    start = date.fromisoformat(req.start_date)
    end = date.fromisoformat(req.end_date)

    if req.track_codes:
        tracks = db.query(Track).filter(Track.code.in_(req.track_codes)).all()
    else:
        tracks = db.query(Track).filter(Track.active.is_(True)).all()

    if not tracks:
        return TriggerResponse(message="No tracks found")

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
                                    stats = upsert_race_results(db_track, races)
                                    track_races += stats["races_new"] + stats["races_updated"]
                                    track_new += stats["races_new"]
                            except Exception as e:
                                logger.error("Error %s %s: %s", tc, current, e)

                            # Commit + update log every 50 days
                            if day_count % 50 == 0:
                                db_track.commit()
                                _update_log(total_races + track_races, total_new + track_new)

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
        log_final = db_final.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
        if log_final:
            log_final.status = "success" if not failed_tracks else "partial"
            log_final.records_scraped = total_races
            log_final.records_new = total_new
            log_final.completed_at = datetime.utcnow()
            if failed_tracks:
                log_final.error_message = f"Failed tracks: {', '.join(failed_tracks)}"
            db_final.commit()
        db_final.close()
        logger.info("Backfill complete: %d races, %d new. Failed: %s", total_races, total_new, failed_tracks or "none")

    def _update_log(scraped: int, new: int):
        """Update the scrape log with current progress."""
        try:
            db_log = SessionLocal()
            log_entry = db_log.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_entry:
                log_entry.records_scraped = scraped
                log_entry.records_new = new
                db_log.commit()
            db_log.close()
        except Exception:
            pass
            db2.close()

    Thread(target=_run_backfill, daemon=True).start()

    return TriggerResponse(
        message=f"Backfill started for {len(tracks)} tracks from {start} to {end}",
        log_id=log.id,
    )


@router.post("/discover-tracks")
def discover_tracks():
    """Return known GRI track codes."""
    from scraping.gri_scraper import GRI_TRACK_CODES
    return {"tracks": [{"code": k, "name": v} for k, v in GRI_TRACK_CODES.items()]}


@router.get("/start-backfill")
def start_backfill_get(
    start_date: str = "2021-04-05",
    end_date: str = "2026-04-05",
    db: Session = Depends(get_db),
):
    """GET endpoint to start a full backfill — use from browser URL bar."""
    req = BackfillRequest(start_date=start_date, end_date=end_date, track_codes=None)
    return trigger_backfill(req, db)

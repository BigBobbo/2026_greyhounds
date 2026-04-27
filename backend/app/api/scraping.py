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
        try:
            log_final = db_final.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_final:
                log_final.status = "success" if not failed_tracks else "partial"
                log_final.records_scraped = total_races
                log_final.records_new = total_new
                log_final.completed_at = datetime.utcnow()
                if failed_tracks:
                    log_final.error_message = f"Failed tracks: {', '.join(failed_tracks)}"
                db_final.commit()
        finally:
            db_final.close()
        logger.info("Backfill complete: %d races, %d new. Failed: %s", total_races, total_new, failed_tracks or "none")

    def _update_log(scraped: int, new: int):
        """Update the scrape log with current progress."""
        db_log = SessionLocal()
        try:
            log_entry = db_log.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
            if log_entry:
                log_entry.records_scraped = scraped
                log_entry.records_new = new
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


@router.post("/discover-tracks")
def discover_tracks():
    """Return known GRI track codes."""
    from scraping.gri_scraper import GRI_TRACK_CODES
    return {"tracks": [{"code": k, "name": v} for k, v in GRI_TRACK_CODES.items()]}


class UpcomingCardRequest(BaseModel):
    track_code: str
    race_date: str
    save: bool = True  # if False, only preview the parse without writing to DB


class UpcomingCardEntry(BaseModel):
    trap: int
    dog_name: str
    trainer_name: str | None = None
    sire_name: str | None = None
    dam_name: str | None = None
    weight_kg: float | None = None


class UpcomingCardRace(BaseModel):
    race_number: int | None = None
    race_time: str | None = None
    distance_m: int | None = None
    grade: str | None = None
    race_type: str = "flat"
    entries: list[UpcomingCardEntry] = []


class UpcomingCardResponse(BaseModel):
    track_code: str
    race_date: str
    url_used: str | None
    races_found: int
    races: list[UpcomingCardRace]
    saved: bool
    db_stats: dict[str, int] | None = None
    message: str | None = None


@router.post("/upcoming", response_model=UpcomingCardResponse)
async def scrape_upcoming_card(req: UpcomingCardRequest, db: Session = Depends(get_db)):
    """
    Scrape a published race card (declarations) for a future date and
    optionally write it to the database with status="scheduled".

    Tries multiple URL patterns to find the card (see
    scraping/gri_racecard_scraper.py). If the GRI URL is non-standard,
    set the GRI_RACECARD_URL env var.

    Use save=False to preview the parse without writing to the DB —
    useful when verifying that the card was scraped correctly before
    committing.
    """
    from scraping.gri_racecard_scraper import scrape_race_card
    from scraping.db_pipeline import upsert_race_results

    track = db.query(Track).filter(Track.code == req.track_code).first()
    if not track:
        return UpcomingCardResponse(
            track_code=req.track_code,
            race_date=req.race_date,
            url_used=None,
            races_found=0,
            races=[],
            saved=False,
            message=f"Unknown track code: {req.track_code}",
        )

    try:
        race_date_val = date.fromisoformat(req.race_date)
    except ValueError:
        return UpcomingCardResponse(
            track_code=req.track_code,
            race_date=req.race_date,
            url_used=None,
            races_found=0,
            races=[],
            saved=False,
            message="race_date must be ISO format (YYYY-MM-DD)",
        )

    races, url_used = await scrape_race_card(req.track_code, race_date_val)

    races_payload = [
        UpcomingCardRace(
            race_number=r.get("race_number"),
            race_time=r.get("race_time"),
            distance_m=r.get("distance_m"),
            grade=r.get("grade"),
            race_type=r.get("race_type", "flat"),
            entries=[
                UpcomingCardEntry(
                    trap=e["trap"],
                    dog_name=e.get("dog_name", ""),
                    trainer_name=e.get("trainer_name"),
                    sire_name=e.get("sire_name"),
                    dam_name=e.get("dam_name"),
                    weight_kg=e.get("weight_kg"),
                )
                for e in r.get("entries", [])
            ],
        )
        for r in races
    ]

    if not races:
        return UpcomingCardResponse(
            track_code=req.track_code,
            race_date=req.race_date,
            url_used=url_used,
            races_found=0,
            races=[],
            saved=False,
            message=(
                "No race card found. The GRI URL may have changed; set "
                "GRI_RACECARD_URL or use the Manual Race Entry page."
            ),
        )

    if not req.save:
        return UpcomingCardResponse(
            track_code=req.track_code,
            race_date=req.race_date,
            url_used=url_used,
            races_found=len(races),
            races=races_payload,
            saved=False,
            message="Preview only — no rows written to the database.",
        )

    # Tag with track_code so upsert_race_results can resolve the track
    for r in races:
        r["track_code"] = req.track_code

    log = ScrapeLog(
        spider_name="gri-racecard",
        source=f"upcoming {req.track_code} {race_date_val}",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    try:
        stats = upsert_race_results(db, races)
        log.status = "success"
        log.records_scraped = stats["races_new"] + stats["races_updated"]
        log.records_new = stats["races_new"]
        log.records_updated = stats["races_updated"]
        log.completed_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        db.rollback()
        log.status = "failed"
        log.error_message = f"{type(e).__name__}: {e}"
        log.completed_at = datetime.utcnow()
        db.commit()
        return UpcomingCardResponse(
            track_code=req.track_code,
            race_date=req.race_date,
            url_used=url_used,
            races_found=len(races),
            races=races_payload,
            saved=False,
            message=f"Parse succeeded but DB write failed: {e}",
        )

    return UpcomingCardResponse(
        track_code=req.track_code,
        race_date=req.race_date,
        url_used=url_used,
        races_found=len(races),
        races=races_payload,
        saved=True,
        db_stats=stats,
        message=(
            f"Saved {stats['races_new']} new race(s), "
            f"{stats['entries_new']} new entries, "
            f"{stats['dogs_new']} new dogs."
        ),
    )


@router.get("/start-backfill")
def start_backfill_get(
    start_date: str = "2021-04-05",
    end_date: str = "2026-04-05",
    tracks: str | None = None,
    db: Session = Depends(get_db),
):
    """
    GET endpoint to start a backfill from browser URL bar.
    Use tracks param for specific tracks: ?tracks=TRL,SPK,CRK
    """
    track_codes = tracks.split(",") if tracks else None
    req = BackfillRequest(start_date=start_date, end_date=end_date, track_codes=track_codes)
    return trigger_backfill(req, db)


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

"""Scraping management API endpoints."""

import asyncio
import logging
from datetime import date, datetime
from threading import Thread
from typing import Any

from fastapi import APIRouter, Depends, Query
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
    date_from: str  # YYYY-MM-DD
    date_to: str | None = None  # defaults to date_from


class BackfillRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    track_codes: list[str] | None = None


class TriggerResponse(BaseModel):
    message: str
    log_id: int | None = None


@router.get("/status", response_model=ScrapingStatusResponse)
def get_scraping_status(db: Session = Depends(get_db)):
    """Get overall scraping status and stats."""
    total_races = db.query(Race).count()
    total_entries = db.query(RaceEntry).count()
    total_dogs = db.query(Dog).count()
    total_tracks = db.query(Track).filter(Track.active.is_(True)).count()

    last_scrape = db.query(ScrapeLog).order_by(ScrapeLog.id.desc()).first()
    recent_logs = (
        db.query(ScrapeLog)
        .order_by(ScrapeLog.id.desc())
        .limit(20)
        .all()
    )

    return ScrapingStatusResponse(
        total_races=total_races,
        total_entries=total_entries,
        total_dogs=total_dogs,
        total_tracks=total_tracks,
        last_scrape=ScrapeLogResponse.model_validate(last_scrape) if last_scrape else None,
        recent_logs=[ScrapeLogResponse.model_validate(log) for log in recent_logs],
    )


def _run_scrape_in_thread(track_code: str, date_from: date, date_to: date, log_id: int):
    """Run scraping in a background thread."""
    from scraping.gri_scraper import scrape_results
    from scraping.db_pipeline import upsert_race_results

    async def _scrape():
        db = SessionLocal()
        log = db.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
        total_races = 0
        total_new = 0

        try:
            current = date_from
            while current <= date_to:
                races = await scrape_results(track_code, current)
                if races:
                    stats = upsert_race_results(db, races)
                    total_races += stats["races_new"] + stats["races_updated"]
                    total_new += stats["races_new"]
                current += __import__("datetime").timedelta(days=1)
                await asyncio.sleep(2.0)

            log.status = "success"
            log.records_scraped = total_races
            log.records_new = total_new
        except Exception as e:
            logger.error("Scrape failed: %s", e)
            log.status = "failed"
            log.error_message = str(e)
        finally:
            log.completed_at = datetime.utcnow()
            db.commit()
            db.close()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_scrape())
    loop.close()


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

    # Run in background thread
    thread = Thread(
        target=_run_scrape_in_thread,
        args=(req.track_code, date_from, date_to, log.id),
        daemon=True,
    )
    thread.start()

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

    def _run_backfill():
        from scraping.backfill import run_backfill
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_backfill(start, end, req.track_codes))
        loop.close()

        db2 = SessionLocal()
        log2 = db2.query(ScrapeLog).filter(ScrapeLog.id == log.id).first()
        if log2:
            log2.status = "success"
            log2.completed_at = datetime.utcnow()
            db2.commit()
        db2.close()

    thread = Thread(target=_run_backfill, daemon=True)
    thread.start()

    return TriggerResponse(
        message=f"Backfill started for {len(tracks)} tracks from {start} to {end}",
        log_id=log.id,
    )


@router.post("/discover-tracks")
def discover_tracks():
    """Trigger track code discovery from GRI website."""
    from scraping.backfill import run_discover_tracks

    def _run():
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_discover_tracks())
        loop.close()

    thread = Thread(target=_run, daemon=True)
    thread.start()

    return {"message": "Track discovery started in background"}


@router.get("/debug-fetch")
async def debug_fetch(track_code: str = "SHP", date_str: str = "04-Apr-2026"):
    """Fetch a GRI page with Playwright (JS rendering) and return debug info."""
    from scraping.gri_scraper import (
        VIEW_RESULTS_URL, fetch_page_playwright, parse_results_page
    )
    from datetime import datetime as dt

    url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"

    try:
        html = await fetch_page_playwright(url, wait_selector="table")
    except Exception as e:
        return {"error": str(e), "url": url}

    # Try to parse the date
    try:
        race_date = dt.strptime(date_str, "%d-%b-%Y").date()
    except ValueError:
        race_date = None

    # Try parsing
    races = []
    if race_date:
        races = parse_results_page(html, track_code, race_date)

    # Analyze the rendered HTML
    from bs4 import BeautifulSoup
    import re
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style", "link", "meta", "noscript"]):
        tag.decompose()

    body = soup.find("body")
    body_text = body.get_text(" ", strip=True)[:5000] if body else ""

    # Find tables
    tables_info = []
    for i, table in enumerate(soup.find_all("table")):
        rows = table.find_all("tr")
        rows_text = []
        for row in rows[:5]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            rows_text.append(cells)
        tables_info.append({
            "table_index": i,
            "num_rows": len(rows),
            "classes": table.get("class", []),
            "first_rows": rows_text,
        })

    # Find race-related elements
    race_elements = []
    for elem in soup.find_all(string=re.compile(r"Race\s+\d+|Trap|525m|480m|550m|\d{2}\.\d{2}", re.IGNORECASE)):
        parent = elem.find_parent()
        if parent:
            race_elements.append({
                "tag": parent.name,
                "class": parent.get("class", []),
                "text": parent.get_text(" ", strip=True)[:300],
            })

    return {
        "url": url,
        "html_length": len(html),
        "body_text_preview": body_text,
        "tables_found": tables_info,
        "race_elements": race_elements[:30],
        "races_parsed": len(races),
        "races": races,
    }

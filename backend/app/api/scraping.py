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


@router.get("/debug-trap-column")
async def debug_trap_column(track_code: str = "SPK", date_str: str = "04-Apr-2026"):
    """Inspect the Trap column HTML to see how trap numbers are represented."""
    from playwright.async_api import async_playwright
    from scraping.gri_scraper import RESULTS_URL, _dismiss_cookie_banners, _navigate_and_load_results
    from bs4 import BeautifulSoup
    from datetime import datetime as dt

    race_date = dt.strptime(date_str, "%d-%b-%Y").date()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        page = await context.new_page()
        html = await _navigate_and_load_results(page, track_code, race_date)
        await browser.close()

    soup = BeautifulSoup(html, "html.parser")

    # Find the first result table — try multiple matching strategies
    trap_info = []
    target_table = None
    for table in soup.find_all("table"):
        header_text = table.get_text(" ", strip=True)[:200].lower()
        if "greyhound" in header_text or "pos" in header_text or "trap" in header_text:
            target_table = table
            break

    if not target_table:
        # Fallback: just get the table with most rows
        tables = soup.find_all("table")
        if tables:
            target_table = max(tables, key=lambda t: len(t.find_all("tr")))

    if not target_table:
        return {"trap_column_analysis": [], "error": "No table found"}

    rows = target_table.find_all("tr")
    # Find which column index is "Trap"
    trap_col_idx = 1  # default
    if rows:
        header_cells = [c.get_text(strip=True).lower() for c in rows[0].find_all(["td", "th"])]
        trap_info.append({"row": -1, "header_cells": header_cells})
        for idx, h in enumerate(header_cells):
            if "trap" in h:
                trap_col_idx = idx
                break

    for i, row in enumerate(rows[:8]):
        cells = row.find_all(["td", "th"])
        if len(cells) <= trap_col_idx:
            continue

        trap_cell = cells[trap_col_idx]
            trap_html = str(trap_cell)
            trap_text = trap_cell.get_text(strip=True)

            # Check for images
            images = trap_cell.find_all("img")
            img_info = []
            for img in images:
                img_info.append({
                    "src": img.get("src", ""),
                    "alt": img.get("alt", ""),
                    "title": img.get("title", ""),
                    "class": img.get("class", []),
                    "width": img.get("width", ""),
                    "height": img.get("height", ""),
                })

            # Check for spans, divs with classes
            spans = trap_cell.find_all(["span", "div"])
            span_info = [{"class": s.get("class", []), "text": s.get_text(strip=True), "style": s.get("style", "")} for s in spans]

            # Check for background colors or styles
            cell_style = trap_cell.get("style", "")
            cell_class = trap_cell.get("class", [])

            trap_info.append({
                "row": i,
                "is_header": i == 0,
                "trap_text": trap_text,
                "trap_html": trap_html[:500],
                "images": img_info,
                "spans": span_info,
                "cell_style": cell_style,
                "cell_class": cell_class,
            })

        break  # Only inspect first table

    return {"trap_column_analysis": trap_info}


@router.get("/debug-fetch")
async def debug_fetch(track_code: str = "Shelbourne Park", date_str: str = "04-Apr-2026"):
    """
    Fetch a GRI page with Playwright. Interacts with the form to load results.
    Use full track names from the dropdown (e.g. 'Shelbourne Park', 'Curraheen Park').
    """
    from playwright.async_api import async_playwright
    from scraping.gri_scraper import RESULTS_URL, _dismiss_cookie_banners, parse_results_page
    from datetime import datetime as dt
    import base64

    debug_info = {"steps": []}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            page = await context.new_page()

            # Block cookie script
            await page.route("**/consent.cookiebot.com/**", lambda route: route.abort())

            # Go to the results page
            await page.goto(RESULTS_URL, timeout=30000, wait_until="networkidle")
            await _dismiss_cookie_banners(page)
            await page.wait_for_timeout(2000)
            debug_info["steps"].append("Loaded results page")

            # Find all select elements and their options
            selects_info = []
            selects = await page.query_selector_all("select")
            for sel in selects:
                sel_id = await sel.get_attribute("id") or ""
                sel_name = await sel.get_attribute("name") or ""
                options = await sel.query_selector_all("option")
                opts = []
                for opt in options[:30]:
                    val = await opt.get_attribute("value") or ""
                    txt = await opt.text_content() or ""
                    opts.append({"value": val, "text": txt.strip()})
                selects_info.append({"id": sel_id, "name": sel_name, "options": opts})
            debug_info["selects"] = selects_info

            # Find all input elements (date pickers, etc.)
            inputs_info = []
            inputs = await page.query_selector_all("input")
            for inp in inputs:
                inp_type = await inp.get_attribute("type") or ""
                inp_id = await inp.get_attribute("id") or ""
                inp_name = await inp.get_attribute("name") or ""
                inp_val = await inp.get_attribute("value") or ""
                inp_placeholder = await inp.get_attribute("placeholder") or ""
                inputs_info.append({
                    "type": inp_type, "id": inp_id, "name": inp_name,
                    "value": inp_val, "placeholder": inp_placeholder,
                })
            debug_info["inputs"] = inputs_info

            # Find all buttons/links
            buttons_info = []
            for btn in await page.query_selector_all("button, input[type='submit'], a.btn"):
                txt = await btn.text_content() or ""
                href = await btn.get_attribute("href") or ""
                cls = await btn.get_attribute("class") or ""
                btn_id = await btn.get_attribute("id") or ""
                buttons_info.append({"text": txt.strip(), "href": href, "class": cls, "id": btn_id})
            debug_info["buttons"] = buttons_info[:20]

            # Try to select the stadium in dropdown
            stadium_selected = False
            for sel in selects:
                options = await sel.query_selector_all("option")
                for opt in options:
                    txt = (await opt.text_content() or "").strip()
                    if txt == track_code or track_code.lower() in txt.lower():
                        val = await opt.get_attribute("value") or ""
                        await sel.select_option(value=val)
                        stadium_selected = True
                        debug_info["steps"].append(f"Selected stadium: {txt} (value={val})")
                        break
                if stadium_selected:
                    break

            # Try to set the date
            date_set = False
            for inp in inputs:
                inp_type = await inp.get_attribute("type") or ""
                inp_id = (await inp.get_attribute("id") or "").lower()
                inp_name = (await inp.get_attribute("name") or "").lower()
                if "date" in inp_id or "date" in inp_name or inp_type == "date":
                    await inp.fill(date_str)
                    date_set = True
                    debug_info["steps"].append(f"Set date input to: {date_str}")
                    break

            # Wait a moment then try to submit
            await page.wait_for_timeout(1000)

            # Try clicking submit/go buttons
            submitted = False
            for btn in await page.query_selector_all("button, input[type='submit'], a"):
                txt = (await btn.text_content() or "").strip().lower()
                if txt in ("go", "search", "go to meeting", "go to meeting search", "submit", "view results"):
                    try:
                        await btn.click()
                        submitted = True
                        debug_info["steps"].append(f"Clicked button: {txt}")
                        break
                    except Exception:
                        continue

            # Wait for results to load
            await page.wait_for_timeout(5000)
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Take screenshot
            screenshot = await page.screenshot(full_page=False)
            screenshot_b64 = base64.b64encode(screenshot).decode()

            html = await page.content()
            await browser.close()

    except Exception as e:
        return {"error": str(e), "steps": debug_info.get("steps", [])}

    # Parse
    try:
        race_date = dt.strptime(date_str, "%d-%b-%Y").date()
    except ValueError:
        race_date = None

    races = []
    if race_date:
        races = parse_results_page(html, track_code, race_date)

    # Analyze HTML
    from bs4 import BeautifulSoup
    import re
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "link", "meta", "noscript"]):
        tag.decompose()
    body = soup.find("body")
    body_text = body.get_text(" ", strip=True)[:5000] if body else ""

    tables_info = []
    for i, table in enumerate(soup.find_all("table")):
        rows = table.find_all("tr")
        rows_text = []
        for row in rows[:5]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            rows_text.append(cells)
        tables_info.append({"table_index": i, "num_rows": len(rows), "first_rows": rows_text})

    race_elements = []
    for elem in soup.find_all(string=re.compile(r"Race\s+\d+|Trap|525m|480m|550m", re.IGNORECASE)):
        parent = elem.find_parent()
        if parent:
            race_elements.append({
                "tag": parent.name,
                "text": parent.get_text(" ", strip=True)[:300],
            })

    return {
        "url": RESULTS_URL,
        "track_code": track_code,
        "date_str": date_str,
        "html_length": len(html),
        "debug_steps": debug_info["steps"],
        "selects": debug_info.get("selects", []),
        "inputs": debug_info.get("inputs", []),
        "buttons": debug_info.get("buttons", []),
        "body_text_preview": body_text,
        "tables_found": tables_info,
        "race_elements": race_elements[:30],
        "races_parsed": len(races),
        "races": races,
        "screenshot_base64": screenshot_b64[:100] + "..." if screenshot_b64 else None,
        "screenshot_url": "/api/scraping/debug-screenshot",
    }

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
    """Run scraping in a background thread with full error capture."""
    import traceback

    def _scrape_sync():
        import asyncio as _asyncio
        from scraping.gri_scraper import _navigate_and_load_results, parse_results_page
        from scraping.db_pipeline import upsert_race_results
        from datetime import timedelta
        from playwright.async_api import async_playwright

        db = SessionLocal()
        log = db.query(ScrapeLog).filter(ScrapeLog.id == log_id).first()
        total_races = 0
        total_new = 0

        async def _run():
            nonlocal total_races, total_new

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                )
                page = await context.new_page()
                await page.route("**/consent.cookiebot.com/**", lambda route: route.abort())

                try:
                    current = date_from
                    while current <= date_to:
                        try:
                            html = await _navigate_and_load_results(page, track_code, current)
                            races = parse_results_page(html, track_code, current)
                            logger.info("Parsed %d races for %s on %s", len(races), track_code, current)

                            if races:
                                stats = upsert_race_results(db, races)
                                total_races += stats["races_new"] + stats["races_updated"]
                                total_new += stats["races_new"]
                                logger.info("Saved: %s", stats)
                            else:
                                logger.info("No races found for %s on %s", track_code, current)

                        except Exception as e:
                            logger.error("Error on %s %s: %s\n%s", track_code, current, e, traceback.format_exc())

                        current += timedelta(days=1)
                        if current <= date_to:
                            await _asyncio.sleep(3.0)
                finally:
                    await browser.close()

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

    # Run in background thread
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


@router.get("/test-scrape")
async def test_scrape(track_code: str = "SPK", date_str: str = "04-Apr-2026"):
    """
    Scrape one date synchronously and save to DB. Returns results directly
    with full diagnostics.
    """
    from scraping.gri_scraper import (
        RESULTS_URL, _dismiss_cookie_banners, parse_results_page, format_date,
    )
    from scraping.db_pipeline import upsert_race_results
    from playwright.async_api import async_playwright
    from datetime import datetime as dt
    from bs4 import BeautifulSoup
    import re

    race_date = dt.strptime(date_str, "%d-%b-%Y").date()
    date_formatted = format_date(race_date)
    steps = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            page = await context.new_page()
            await page.route("**/consent.cookiebot.com/**", lambda route: route.abort())

            # Navigate to results page
            await page.goto(RESULTS_URL, timeout=30000, wait_until="networkidle")
            await _dismiss_cookie_banners(page)
            await page.wait_for_timeout(1000)
            steps.append("Loaded results page")

            # Select stadium
            stadium_select = await page.query_selector("#stadium")
            if stadium_select:
                await stadium_select.select_option(value=track_code)
                steps.append(f"Selected stadium: {track_code}")
            else:
                steps.append("ERROR: #stadium dropdown not found")

            # Set date
            date_input = await page.query_selector("#FromDate")
            if date_input:
                await date_input.click()
                await page.wait_for_timeout(300)
                # Clear and type the date
                await date_input.press("Control+a")
                await date_input.type(date_formatted, delay=50)
                steps.append(f"Typed date: {date_formatted}")
            else:
                steps.append("ERROR: #FromDate input not found")

            await page.wait_for_timeout(500)

            # Find and click the submit button
            clicked = False
            for btn_text in ["Show Meetings", "view results", "View Results"]:
                try:
                    btn = await page.query_selector(f"button:has-text('{btn_text}')")
                    if btn and await btn.is_visible():
                        await btn.click()
                        steps.append(f"Clicked: {btn_text}")
                        clicked = True
                        break
                except Exception:
                    continue

            if not clicked:
                # Try any visible button with class btn
                for btn in await page.query_selector_all("button.btn, a.btn"):
                    txt = (await btn.text_content() or "").strip()
                    if txt and "menu" not in txt.lower():
                        try:
                            await btn.click()
                            steps.append(f"Clicked fallback btn: {txt}")
                            clicked = True
                            break
                        except Exception:
                            continue

            if not clicked:
                steps.append("ERROR: No submit button found/clicked")

            # Wait for meetings list to load
            await page.wait_for_timeout(3000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            # Step 2: The form loads a meeting list — we need to click on
            # the specific meeting to see results. Look for links/buttons
            # with "View Race" or the track name or meeting links.
            meeting_clicked = False

            # Try clicking "View Race" links or meeting links
            for selector in [
                "a:has-text('View Race')",
                "a:has-text('View Results')",
                "a:has-text('Race 1')",
                f"a:has-text('{track_code}')",
                ".meeting-link",
                "a[href*='race']",
                "a[href*='result']",
            ]:
                try:
                    link = await page.query_selector(selector)
                    if link and await link.is_visible():
                        await link.click()
                        meeting_clicked = True
                        steps.append(f"Clicked meeting link: {selector}")
                        break
                except Exception:
                    continue

            if not meeting_clicked:
                # List all visible links to find the right one
                visible_links = []
                for link in await page.query_selector_all("a"):
                    try:
                        if await link.is_visible():
                            txt = (await link.text_content() or "").strip()
                            href = (await link.get_attribute("href") or "")
                            if txt and len(txt) < 100 and txt not in ("", "Home Page", "Results"):
                                visible_links.append({"text": txt[:80], "href": href[:100]})
                    except Exception:
                        continue
                steps.append(f"No meeting link found. Visible links: {visible_links[:15]}")

            # Wait for results to load after clicking meeting
            if meeting_clicked:
                await page.wait_for_timeout(5000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass

            html = await page.content()
            await browser.close()

    except Exception as e:
        return {"error": f"Playwright failed: {e}", "steps": steps}

    # Check what we got
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    body_text = soup.get_text(" ", strip=True)[:1000]

    # Count tables and race headers
    tables = soup.find_all("table")
    race_headers = soup.find_all("h4", string=re.compile(r"Race\s+\d+", re.IGNORECASE))

    races = parse_results_page(html, track_code, race_date)

    result = {
        "steps": steps,
        "html_length": len(html),
        "tables_count": len(tables),
        "race_headers_count": len(race_headers),
        "races_parsed": len(races),
        "body_preview": body_text,
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
            "first_dog": races[0]["entries"][0]["dog_name"] if races[0]["entries"] else None,
        }
    except Exception as e:
        db.rollback()
        result["db_error"] = str(e)
    finally:
        db.close()

    return result


@router.get("/debug-trap-column")
async def debug_trap_column(track_code: str = "SPK", date_str: str = "04-Apr-2026"):
    """
    Inspect the Trap column HTML using a lighter approach.
    Reuses the same page fetch as debug-fetch but only returns trap cell details.
    """
    from scraping.gri_scraper import _navigate_and_load_results
    from playwright.async_api import async_playwright
    from bs4 import BeautifulSoup
    from datetime import datetime as dt
    import re

    race_date = dt.strptime(date_str, "%d-%b-%Y").date()

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            page = await context.new_page()
            html = await _navigate_and_load_results(page, track_code, race_date)
            await browser.close()
    except Exception as e:
        return {"error": str(e)}

    soup = BeautifulSoup(html, "html.parser")

    # Find a result table
    target_table = None
    for table in soup.find_all("table"):
        if "Greyhound" in table.get_text()[:300]:
            target_table = table
            break

    if not target_table:
        return {"error": "No result table found", "table_count": len(soup.find_all("table"))}

    rows = target_table.find_all("tr")
    header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])]

    trap_col_idx = None
    for idx, h in enumerate(header_cells):
        if "trap" in h.lower():
            trap_col_idx = idx
            break

    result = {
        "header_cells": header_cells,
        "trap_col_idx": trap_col_idx,
        "rows_analyzed": [],
    }

    for i, row in enumerate(rows[:4]):  # header + 3 data rows
        cells = row.find_all(["td", "th"])
        if trap_col_idx is not None and trap_col_idx < len(cells):
            cell = cells[trap_col_idx]
            raw_html = str(cell)

            # Check all children elements
            children = []
            for child in cell.descendants:
                if hasattr(child, 'name') and child.name:
                    children.append({
                        "tag": child.name,
                        "attrs": dict(child.attrs) if hasattr(child, 'attrs') else {},
                        "text": child.get_text(strip=True),
                    })

            result["rows_analyzed"].append({
                "row_idx": i,
                "raw_html": raw_html[:500],
                "text_content": cell.get_text(strip=True),
                "children": children,
            })

    return result


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

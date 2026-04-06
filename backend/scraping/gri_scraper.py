"""
GRI (Greyhound Racing Ireland) scraper using Playwright + BeautifulSoup.

The GRI results page loads race data dynamically via JavaScript,
so we MUST use Playwright to render the page before parsing.

URL pattern: https://www.grireland.ie/results/view-results/?track={CODE}&date={DD-Mon-YYYY}
"""

import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

GRI_BASE_URL = "https://www.grireland.ie"
RESULTS_URL = f"{GRI_BASE_URL}/results/"
VIEW_RESULTS_URL = f"{GRI_BASE_URL}/results/view-results/"


def format_date(d: date) -> str:
    """Format date as DD-Mon-YYYY for GRI URL (e.g. '05-Apr-2026')."""
    return d.strftime("%d-%b-%Y")


async def fetch_page_playwright(url: str, wait_selector: str | None = None) -> str:
    """Fetch a page using Playwright, waiting for JS to render content."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            # Block cookie consent script to prevent overlay
            await page.route("**/consent.cookiebot.com/**", lambda route: route.abort())

            await page.goto(url, timeout=30000, wait_until="networkidle")

            # Dismiss any cookie banners that still appear
            await _dismiss_cookie_banners(page)

            # Wait for dynamic content to load
            if wait_selector:
                try:
                    await page.wait_for_selector(wait_selector, timeout=10000)
                except Exception:
                    logger.debug("Selector '%s' not found, continuing", wait_selector)

            # Wait for race data to render
            await page.wait_for_timeout(3000)

            # Try clicking the Go/Search button if a form exists
            await _submit_search_form(page)

            html = await page.content()
            return html
        finally:
            await browser.close()


async def _dismiss_cookie_banners(page) -> None:
    """Try to dismiss cookie consent banners."""
    # Common cookie banner button selectors
    selectors = [
        "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
        "#CybotCookiebotDialogBodyButtonAccept",
        "button[data-cookieconsent='accept']",
        ".cookie-accept",
        "button:has-text('Accept')",
        "button:has-text('Allow all')",
        "button:has-text('Accept all')",
        "a:has-text('Accept')",
    ]
    for selector in selectors:
        try:
            btn = await page.query_selector(selector)
            if btn:
                await btn.click()
                logger.debug("Dismissed cookie banner with selector: %s", selector)
                await page.wait_for_timeout(1000)
                return
        except Exception:
            continue

    # Try removing the overlay via JavaScript
    try:
        await page.evaluate("""
            document.querySelectorAll('[id*="Cookiebot"], [id*="cookie"], .cookie-banner, .cookie-overlay, #CybotCookiebotDialog').forEach(el => el.remove());
            document.body.style.overflow = 'auto';
        """)
    except Exception:
        pass


async def _submit_search_form(page) -> None:
    """Try to find and click a Go/Search/Submit button on the results form."""
    selectors = [
        "input[type='submit']",
        "button[type='submit']",
        "a:has-text('Go to Meeting')",
        "button:has-text('Go')",
        "button:has-text('Search')",
        ".btn-search",
        "a.btn:has-text('Go')",
    ]
    for selector in selectors:
        try:
            btn = await page.query_selector(selector)
            if btn and await btn.is_visible():
                await btn.click()
                await page.wait_for_timeout(3000)
                logger.debug("Clicked submit button: %s", selector)
                return
        except Exception:
            continue


async def fetch_page_playwright_reuse(page, url: str) -> str:
    """Fetch a page reusing an existing Playwright page (for batch scraping)."""
    await page.goto(url, timeout=30000, wait_until="networkidle")
    await _dismiss_cookie_banners(page)
    await page.wait_for_timeout(3000)
    await _submit_search_form(page)
    return await page.content()


async def discover_track_codes() -> list[dict[str, str]]:
    """
    Navigate to the GRI results page with Playwright and extract track options
    from the stadium dropdown.
    """
    html = await fetch_page_playwright(RESULTS_URL, wait_selector="select")
    soup = BeautifulSoup(html, "html.parser")

    tracks: list[dict[str, str]] = []

    # Look for <select> elements — the stadium dropdown
    for select in soup.find_all("select"):
        for option in select.find_all("option"):
            value = option.get("value", "").strip()
            text = option.get_text(strip=True)
            # Skip "All Stadia" and empty options
            if not value or not text or text == "All Stadia":
                continue
            # Track codes might be full names or codes
            tracks.append({"code": value, "name": text})

    if tracks:
        logger.info("Discovered %d tracks from GRI dropdown", len(tracks))
        return tracks

    # Fallback: look for links with track parameter
    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.search(r"track=([^&]+)", href)
        if match:
            code = match.group(1)
            name = link.get_text(strip=True)
            if code and name and code != "All":
                tracks.append({"code": code, "name": name})

    if tracks:
        seen = set()
        unique = [t for t in tracks if t["code"] not in seen and not seen.add(t["code"])]
        logger.info("Discovered %d tracks from links", len(unique))
        return unique

    logger.warning("Could not discover tracks, using fallback")
    return _fallback_track_codes()


def _fallback_track_codes() -> list[dict[str, str]]:
    """Fallback track codes based on known GRI/IGB codes."""
    return [
        {"code": "Clonmel", "name": "Clonmel"},
        {"code": "Curraheen Park", "name": "Curraheen Park"},
        {"code": "Derry", "name": "Derry"},
        {"code": "Drumbo Park", "name": "Drumbo Park"},
        {"code": "Dundalk", "name": "Dundalk"},
        {"code": "Enniscorthy", "name": "Enniscorthy"},
        {"code": "Galway", "name": "Galway"},
        {"code": "Kilkenny", "name": "Kilkenny"},
        {"code": "Lifford", "name": "Lifford"},
        {"code": "Limerick", "name": "Limerick"},
        {"code": "Longford", "name": "Longford"},
        {"code": "Mullingar", "name": "Mullingar"},
        {"code": "Newbridge", "name": "Newbridge"},
        {"code": "Shelbourne Park", "name": "Shelbourne Park"},
        {"code": "Thurles Park", "name": "Thurles Park"},
        {"code": "Tralee", "name": "Tralee"},
        {"code": "Waterford", "name": "Waterford"},
        {"code": "Youghal", "name": "Youghal"},
    ]


def parse_results_page(html: str, track_code: str, race_date: date) -> list[dict[str, Any]]:
    """
    Parse a GRI results page (JS-rendered HTML) and extract race + entry data.
    Returns a list of race dicts, each containing an 'entries' list.
    """
    soup = BeautifulSoup(html, "html.parser")
    races: list[dict[str, Any]] = []

    # Strategy 1: Find race containers by looking for headers like "Race 1", "Race 2"
    race_headers = soup.find_all(string=re.compile(r"Race\s+\d+", re.IGNORECASE))

    if race_headers:
        # Group content by race headers
        for header_text in race_headers:
            header_elem = header_text.find_parent()
            if not header_elem:
                continue

            # Find the race container (parent div/section)
            container = header_elem
            for _ in range(5):  # Walk up to find a meaningful container
                parent = container.parent
                if parent and parent.name in ["div", "section", "article"]:
                    # Check if this parent contains tables or result data
                    if parent.find("table") or parent.find(class_=re.compile(r"result|race|card", re.IGNORECASE)):
                        container = parent
                        break
                    container = parent
                else:
                    break

            race_data = _parse_race_section(container, track_code, race_date)
            if race_data and race_data.get("entries"):
                races.append(race_data)

    # Strategy 2: Look for tables with race data
    if not races:
        for table in soup.find_all("table"):
            table_text = table.get_text(" ", strip=True)
            if re.search(r"Trap|Position|Time|SP|Trainer", table_text, re.IGNORECASE):
                race_data = _parse_race_table_generic(table, len(races) + 1, track_code, race_date)
                if race_data and race_data.get("entries"):
                    races.append(race_data)

    # Strategy 3: Look for divs with class patterns containing race/result
    if not races:
        for div in soup.find_all(class_=re.compile(r"race-result|raceResult|race_result|result-card", re.IGNORECASE)):
            race_data = _parse_race_section(div, track_code, race_date)
            if race_data and race_data.get("entries"):
                races.append(race_data)

    # Strategy 4: Look for repeated structural patterns (rows of dog data)
    if not races:
        races = _parse_flat_structure(soup, track_code, race_date)

    logger.info("Parsed %d races from %s on %s", len(races), track_code, race_date)
    return races


def _parse_race_section(container, track_code: str, race_date: date) -> dict[str, Any] | None:
    """Parse a container element that holds one race."""
    text = container.get_text(" ", strip=True)

    # Extract race number
    race_num_match = re.search(r"Race\s+(\d+)", text, re.IGNORECASE)
    race_number = int(race_num_match.group(1)) if race_num_match else None

    # Extract distance
    distance = None
    dist_match = re.search(r"(\d{3,4})\s*(?:m\b|metres?|Metres?|yds)", text, re.IGNORECASE)
    if dist_match:
        distance = int(dist_match.group(1))

    # Extract grade
    grade = None
    grade_match = re.search(r"\b([A-S]\d|OR|ON|D\d|Nov|OP|Novice|Puppy)\b", text, re.IGNORECASE)
    if grade_match:
        grade = grade_match.group(1).upper()

    # Extract prize
    prize = None
    prize_match = re.search(r"[€£]\s*([\d,]+)", text)
    if prize_match:
        prize = float(prize_match.group(1).replace(",", ""))

    # Extract going
    going = None
    going_match = re.search(r"Going[:\s]+([A-Za-z\s\+\-\d.]+?)(?:\s*[|,]|\s*$)", text)
    if going_match:
        going = going_match.group(1).strip()[:30]

    # Extract race type
    race_type = "hurdle" if re.search(r"hurdle|hurdles", text, re.IGNORECASE) else "flat"

    # Parse entries from tables within this container
    entries = []
    for table in container.find_all("table"):
        entries.extend(_parse_result_table(table))

    # If no tables, try parsing rows/divs
    if not entries:
        entries = _parse_entry_divs(container)

    if not entries:
        return None

    return {
        "race_number": race_number,
        "race_date": race_date,
        "track_code": track_code,
        "distance_m": distance,
        "grade": grade,
        "race_type": race_type,
        "going": going,
        "prize_money": prize,
        "entries": entries,
    }


def _parse_result_table(table) -> list[dict[str, Any]]:
    """Parse a <table> element to extract race entries."""
    entries = []
    rows = table.find_all("tr")

    # Detect header row to understand column mapping
    header_map = {}
    if rows:
        header_cells = rows[0].find_all(["th", "td"])
        for i, cell in enumerate(header_cells):
            text = cell.get_text(strip=True).lower()
            if "trap" in text:
                header_map["trap"] = i
            elif "pos" in text or "position" in text:
                header_map["position"] = i
            elif "greyhound" in text or "dog" in text or "name" in text:
                header_map["dog_name"] = i
            elif "time" in text and "sectional" not in text:
                header_map["time"] = i
            elif "sectional" in text or "sec" in text:
                header_map["sectional"] = i
            elif "sp" in text or "price" in text:
                header_map["sp"] = i
            elif "weight" in text or "wt" in text:
                header_map["weight"] = i
            elif "trainer" in text:
                header_map["trainer"] = i
            elif "beaten" in text or "btn" in text:
                header_map["beaten"] = i
            elif "comment" in text or "remarks" in text:
                header_map["comment"] = i

    # Parse data rows
    data_rows = rows[1:] if header_map else rows
    for row in data_rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        entry = _extract_entry_from_cells(cells, header_map)
        if entry and entry.get("dog_name"):
            entries.append(entry)

    return entries


def _extract_entry_from_cells(cells, header_map: dict) -> dict[str, Any] | None:
    """Extract entry data from table cells using header mapping or heuristics."""
    entry: dict[str, Any] = {}
    texts = [c.get_text(strip=True) for c in cells]

    if header_map:
        # Use header mapping
        if "trap" in header_map and header_map["trap"] < len(texts):
            trap_text = re.search(r"\d+", texts[header_map["trap"]])
            if trap_text:
                entry["trap"] = int(trap_text.group())
        if "position" in header_map and header_map["position"] < len(texts):
            pos_text = re.search(r"\d+", texts[header_map["position"]])
            if pos_text:
                entry["finish_position"] = int(pos_text.group())
        if "dog_name" in header_map and header_map["dog_name"] < len(texts):
            entry["dog_name"] = texts[header_map["dog_name"]]
            # Also check for link
            link = cells[header_map["dog_name"]].find("a")
            if link:
                entry["dog_name"] = link.get_text(strip=True)
        if "time" in header_map and header_map["time"] < len(texts):
            time_match = re.search(r"(\d{2}\.\d{2})", texts[header_map["time"]])
            if time_match:
                entry["finish_time"] = float(time_match.group(1))
        if "sectional" in header_map and header_map["sectional"] < len(texts):
            sec_match = re.search(r"(\d+\.\d{2})", texts[header_map["sectional"]])
            if sec_match:
                entry["sectional_time"] = float(sec_match.group(1))
        if "sp" in header_map and header_map["sp"] < len(texts):
            sp_text = texts[header_map["sp"]]
            entry["starting_price"] = sp_text
            entry["sp_decimal"] = _parse_sp_decimal(sp_text)
        if "weight" in header_map and header_map["weight"] < len(texts):
            wt_match = re.search(r"(\d{2}\.?\d?)", texts[header_map["weight"]])
            if wt_match:
                entry["weight_kg"] = float(wt_match.group(1))
        if "trainer" in header_map and header_map["trainer"] < len(texts):
            entry["trainer_name"] = texts[header_map["trainer"]]
        if "beaten" in header_map and header_map["beaten"] < len(texts):
            btn_match = re.search(r"([\d.]+)", texts[header_map["beaten"]])
            if btn_match:
                entry["beaten_distance"] = float(btn_match.group(1))
        if "comment" in header_map and header_map["comment"] < len(texts):
            entry["comment"] = texts[header_map["comment"]]
    else:
        # Heuristic parsing — try to identify columns by content
        entry = _heuristic_parse_row(texts, cells)

    return entry if entry.get("dog_name") else None


def _heuristic_parse_row(texts: list[str], cells) -> dict[str, Any]:
    """Parse a table row by guessing columns from content patterns."""
    entry: dict[str, Any] = {}

    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue

        # Trap (single digit 1-8)
        if re.match(r"^[1-8]$", text) and "trap" not in entry:
            entry["trap"] = int(text)
        # Finish position
        elif re.match(r"^\d+(st|nd|rd|th)?$", text) and "finish_position" not in entry and entry.get("trap"):
            entry["finish_position"] = int(re.match(r"^(\d+)", text).group(1))
        # Finish time (28.xx, 29.xx, 30.xx etc)
        elif re.match(r"^\d{2}\.\d{2}$", text) and "finish_time" not in entry:
            entry["finish_time"] = float(text)
        # SP fraction
        elif re.match(r"^\d+/\d+$", text) or text.lower() in ("evens", "evs"):
            entry["starting_price"] = text
            entry["sp_decimal"] = _parse_sp_decimal(text)
        # Weight (25-45 range)
        elif re.match(r"^\d{2}\.\d$", text) and 20 <= float(text) <= 45:
            entry["weight_kg"] = float(text)
        # Dog name (check for link first)
        elif "dog_name" not in entry:
            link = cells[i].find("a") if i < len(cells) else None
            if link:
                name = link.get_text(strip=True)
                if len(name) > 2 and not name.isdigit():
                    entry["dog_name"] = name
            elif re.match(r"^[A-Za-z][A-Za-z\s'\-\.]{2,}$", text) and len(text) > 3:
                entry["dog_name"] = text

    return entry


def _parse_entry_divs(container) -> list[dict[str, Any]]:
    """Parse non-table entry elements (divs, list items, etc.)."""
    entries = []

    # Look for elements with trap/runner classes
    for elem in container.find_all(class_=re.compile(r"trap|runner|entry|dog|row", re.IGNORECASE)):
        text = elem.get_text(" ", strip=True)

        entry: dict[str, Any] = {}

        # Trap
        trap_match = re.search(r"Trap\s*(\d)", text, re.IGNORECASE)
        if trap_match:
            entry["trap"] = int(trap_match.group(1))

        # Dog name from link
        link = elem.find("a")
        if link:
            name = link.get_text(strip=True)
            if len(name) > 2 and not name.isdigit():
                entry["dog_name"] = name

        # Position
        pos_match = re.search(r"(\d+)(st|nd|rd|th)", text)
        if pos_match:
            entry["finish_position"] = int(pos_match.group(1))

        # Time
        time_match = re.search(r"(\d{2}\.\d{2})", text)
        if time_match:
            entry["finish_time"] = float(time_match.group(1))

        # SP
        sp_match = re.search(r"(\d+/\d+|evens|evs)", text, re.IGNORECASE)
        if sp_match:
            entry["starting_price"] = sp_match.group(1)
            entry["sp_decimal"] = _parse_sp_decimal(sp_match.group(1))

        if entry.get("dog_name") and entry.get("trap"):
            entries.append(entry)

    return entries


def _parse_flat_structure(soup, track_code: str, race_date: date) -> list[dict[str, Any]]:
    """Try to parse race data from a flat page structure (no clear containers)."""
    races = []
    text = soup.get_text(" ", strip=True)

    # Check if there's any race data at all
    if not re.search(r"Race\s+\d+", text, re.IGNORECASE):
        return []

    # Find all tables on the page
    tables = soup.find_all("table")
    for i, table in enumerate(tables):
        entries = _parse_result_table(table)
        if entries:
            races.append({
                "race_number": i + 1,
                "race_date": race_date,
                "track_code": track_code,
                "distance_m": None,
                "grade": None,
                "race_type": "flat",
                "going": None,
                "prize_money": None,
                "entries": entries,
            })

    return races


def _parse_race_table_generic(table, race_num: int, track_code: str, race_date: date) -> dict[str, Any] | None:
    """Parse a standalone table as a race result."""
    entries = _parse_result_table(table)
    if not entries:
        return None

    return {
        "race_number": race_num,
        "race_date": race_date,
        "track_code": track_code,
        "distance_m": None,
        "grade": None,
        "race_type": "flat",
        "going": None,
        "prize_money": None,
        "entries": entries,
    }


def _parse_sp_decimal(sp_text: str) -> float | None:
    """Convert SP text to decimal odds."""
    sp_text = sp_text.strip().lower()
    if sp_text in ("evens", "evs"):
        return 2.0
    match = re.match(r"(\d+)/(\d+)", sp_text)
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        if den > 0:
            return round(num / den + 1, 2)
    return None


async def scrape_results(
    track_code: str,
    race_date: date,
) -> list[dict[str, Any]]:
    """Scrape race results for a specific track and date using Playwright."""
    date_str = format_date(race_date)
    url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"

    logger.info("Scraping %s %s -> %s", track_code, race_date, url)

    try:
        html = await fetch_page_playwright(url, wait_selector="table")
    except Exception as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return []

    races = parse_results_page(html, track_code, race_date)
    logger.info("Found %d races for %s on %s", len(races), track_code, race_date)

    return races


async def scrape_date_range(
    track_code: str,
    start_date: date,
    end_date: date,
    delay: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Scrape results for a track across a date range.
    Reuses Playwright browser for efficiency.
    """
    from playwright.async_api import async_playwright

    all_races: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            current = start_date
            total_days = (end_date - start_date).days + 1
            day_num = 0

            while current <= end_date:
                day_num += 1
                date_str = format_date(current)
                url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"

                try:
                    html = await fetch_page_playwright_reuse(page, url)
                    races = parse_results_page(html, track_code, current)
                    all_races.extend(races)
                except Exception as e:
                    logger.error("Error scraping %s %s: %s", track_code, current, e)

                if day_num % 10 == 0:
                    logger.info(
                        "Progress: %s %d/%d days, %d races found",
                        track_code, day_num, total_days, len(all_races),
                    )

                current += timedelta(days=1)
                if current <= end_date:
                    await asyncio.sleep(delay)

        finally:
            await browser.close()

    logger.info(
        "Completed %s: %d days scraped, %d races found",
        track_code, total_days, len(all_races),
    )
    return all_races

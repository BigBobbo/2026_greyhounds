"""
GRI (Greyhound Racing Ireland) scraper using httpx + BeautifulSoup.
Falls back to Playwright for JS-rendered pages if needed.

URL pattern: https://www.grireland.ie/results/view-results/?track={CODE}&date={DD-Mon-YYYY}

Track codes are discovered from the results page on first run.
"""

import asyncio
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

GRI_BASE_URL = "https://www.grireland.ie"
RESULTS_URL = f"{GRI_BASE_URL}/results/"
VIEW_RESULTS_URL = f"{GRI_BASE_URL}/results/view-results/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def format_date(d: date) -> str:
    """Format date as DD-Mon-YYYY for GRI URL (e.g. '05-Apr-2026')."""
    return d.strftime("%d-%b-%Y")


async def fetch_page(url: str, client: httpx.AsyncClient | None = None) -> str:
    """Fetch a page with httpx. Falls back to Playwright if 403."""
    close_client = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30)
        close_client = True

    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.text

        if resp.status_code == 403:
            logger.info("httpx got 403, trying Playwright for %s", url)
            return await _fetch_with_playwright(url)

        resp.raise_for_status()
        return resp.text
    finally:
        if close_client:
            await client.aclose()


async def _fetch_with_playwright(url: str) -> str:
    """Fetch page using headless Playwright browser."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install chromium")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)
            return await page.content()
        finally:
            await browser.close()


async def discover_track_codes(client: httpx.AsyncClient | None = None) -> list[dict[str, str]]:
    """
    Navigate to the GRI results page and extract track codes from
    dropdowns, links, or other selectors.

    Returns list of {"code": "SHP", "name": "Shelbourne Park"}.
    """
    html = await fetch_page(RESULTS_URL, client)
    soup = BeautifulSoup(html, "html.parser")

    tracks: list[dict[str, str]] = []

    # Strategy 1: Look for <select> with track options
    for select in soup.find_all("select"):
        for option in select.find_all("option"):
            value = option.get("value", "").strip()
            text = option.get_text(strip=True)
            if value and text and len(value) <= 5 and value.isalpha():
                tracks.append({"code": value, "name": text})

    if tracks:
        logger.info("Found %d tracks from <select> dropdown", len(tracks))
        return tracks

    # Strategy 2: Look for links with track= parameter
    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.search(r"track=([A-Z]{2,5})", href)
        if match:
            code = match.group(1)
            name = link.get_text(strip=True)
            if code and name:
                tracks.append({"code": code, "name": name})

    if tracks:
        # Deduplicate
        seen = set()
        unique = []
        for t in tracks:
            if t["code"] not in seen:
                seen.add(t["code"])
                unique.append(t)
        logger.info("Found %d tracks from links", len(unique))
        return unique

    # Strategy 3: Look for any element with data attributes for tracks
    for elem in soup.find_all(attrs={"data-track": True}):
        code = elem.get("data-track", "").strip()
        name = elem.get_text(strip=True)
        if code:
            tracks.append({"code": code, "name": name or code})

    if tracks:
        logger.info("Found %d tracks from data attributes", len(tracks))
        return tracks

    logger.warning("Could not discover tracks. Using fallback codes.")
    return _fallback_track_codes()


def _fallback_track_codes() -> list[dict[str, str]]:
    """Fallback track codes based on known GRI/IGB codes."""
    # These are the codes from the old IGB scraper + best guesses for GRI
    return [
        {"code": "CLM", "name": "Clonmel"},
        {"code": "CRK", "name": "Curraheen Park"},
        {"code": "DRY", "name": "Derry"},
        {"code": "DBP", "name": "Drumbo Park"},
        {"code": "DLK", "name": "Dundalk"},
        {"code": "ENN", "name": "Enniscorthy"},
        {"code": "GAL", "name": "Galway"},
        {"code": "KKY", "name": "Kilkenny"},
        {"code": "LMK", "name": "Limerick"},
        {"code": "LFD", "name": "Longford"},
        {"code": "MUL", "name": "Mullingar"},
        {"code": "NBR", "name": "Newbridge"},
        {"code": "SHP", "name": "Shelbourne Park"},
        {"code": "THL", "name": "Thurles"},
        {"code": "TRA", "name": "Tralee"},
        {"code": "WFD", "name": "Waterford"},
        {"code": "YGL", "name": "Youghal"},
    ]


def parse_results_page(html: str, track_code: str, race_date: date) -> list[dict[str, Any]]:
    """
    Parse a GRI results page HTML and extract race + entry data.

    Returns a list of race dicts, each containing an 'entries' list.
    The parser uses multiple strategies to handle different HTML structures.
    """
    soup = BeautifulSoup(html, "html.parser")
    races: list[dict[str, Any]] = []

    # Log HTML structure for debugging on first runs
    if not soup.find(string=re.compile(r"Race\s+\d+", re.IGNORECASE)):
        logger.debug("No 'Race N' text found on page for %s %s", track_code, race_date)
        return []

    # Strategy 1: Look for race containers (divs/sections with Race N headers)
    race_containers = _find_race_containers(soup)
    if race_containers:
        for i, container in enumerate(race_containers, 1):
            race_data = _parse_race_container(container, i, track_code, race_date)
            if race_data:
                races.append(race_data)
        return races

    # Strategy 2: Look for tables with result data
    tables = soup.find_all("table")
    for i, table in enumerate(tables, 1):
        race_data = _parse_race_table(table, i, track_code, race_date)
        if race_data:
            races.append(race_data)

    return races


def _find_race_containers(soup: BeautifulSoup) -> list:
    """Find HTML containers for individual races."""
    # Try common patterns
    containers = []

    # Pattern: div/section with class containing 'race'
    for cls_pattern in ["race", "result", "racecard"]:
        elements = soup.find_all(class_=re.compile(cls_pattern, re.IGNORECASE))
        if elements:
            containers = elements
            break

    if not containers:
        # Pattern: elements containing "Race N" headers
        headers = soup.find_all(string=re.compile(r"Race\s+\d+", re.IGNORECASE))
        for header in headers:
            parent = header.find_parent(["div", "section", "article"])
            if parent and parent not in containers:
                containers.append(parent)

    return containers


def _parse_race_container(container, race_num: int, track_code: str, race_date: date) -> dict[str, Any] | None:
    """Parse a single race container element."""
    race_info = _extract_race_info(container, race_num, race_date)
    entries = _extract_entries(container)

    if not entries:
        return None

    race_info["entries"] = entries
    race_info["track_code"] = track_code
    return race_info


def _parse_race_table(table, race_num: int, track_code: str, race_date: date) -> dict[str, Any] | None:
    """Parse a table as race results."""
    rows = table.find_all("tr")
    if len(rows) < 2:
        return None

    entries = []
    for row in rows[1:]:  # skip header
        cells = row.find_all(["td", "th"])
        if len(cells) >= 3:
            entry = _parse_table_row(cells)
            if entry:
                entries.append(entry)

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


def _extract_race_info(container, race_num: int, race_date: date) -> dict[str, Any]:
    """Extract race-level info (distance, grade, going, etc.) from container."""
    text = container.get_text(" ", strip=True)

    # Extract distance (e.g. "525m", "480 Metres")
    distance = None
    dist_match = re.search(r"(\d{3,4})\s*(?:m|metres?|yds)", text, re.IGNORECASE)
    if dist_match:
        distance = int(dist_match.group(1))

    # Extract grade (e.g. "A1", "S3", "OR", "ON")
    grade = None
    grade_match = re.search(r"\b([A-S]\d|OR|ON|D\d|Nov)\b", text, re.IGNORECASE)
    if grade_match:
        grade = grade_match.group(1).upper()

    # Extract prize money (e.g. "€300", "Prize €1,000")
    prize = None
    prize_match = re.search(r"€\s*([\d,]+)", text)
    if prize_match:
        prize = float(prize_match.group(1).replace(",", ""))

    # Extract race number from text if possible
    num_match = re.search(r"Race\s+(\d+)", text, re.IGNORECASE)
    actual_num = int(num_match.group(1)) if num_match else race_num

    # Extract going
    going = None
    going_match = re.search(r"(?:Going|going)[:\s]*([A-Za-z\s\+\-\d.]+?)(?:\s*\||$)", text)
    if going_match:
        going = going_match.group(1).strip()

    # Extract race type
    race_type = "flat"
    if re.search(r"hurdle|hurdles", text, re.IGNORECASE):
        race_type = "hurdle"

    return {
        "race_number": actual_num,
        "race_date": race_date,
        "distance_m": distance,
        "grade": grade,
        "race_type": race_type,
        "going": going,
        "prize_money": prize,
    }


def _extract_entries(container) -> list[dict[str, Any]]:
    """Extract race entries (dogs) from a race container."""
    entries = []

    # Look for table rows within the container
    tables = container.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                entry = _parse_table_row(cells)
                if entry:
                    entries.append(entry)

    if entries:
        return entries

    # Look for list items or divs with dog info
    items = container.find_all(class_=re.compile(r"dog|runner|entry|trap", re.IGNORECASE))
    for item in items:
        entry = _parse_entry_element(item)
        if entry:
            entries.append(entry)

    return entries


def _parse_table_row(cells) -> dict[str, Any] | None:
    """Parse a table row into a race entry dict."""
    texts = [c.get_text(strip=True) for c in cells]

    # Skip header rows
    if any(t.lower() in ("trap", "pos", "position", "dog", "greyhound") for t in texts[:3]):
        return None

    # Try to identify columns
    entry: dict[str, Any] = {}

    for i, text in enumerate(texts):
        # Trap number (1-8)
        if re.match(r"^[1-8]$", text) and "trap" not in entry:
            entry["trap"] = int(text)

        # Finish position (1st, 2nd, etc or just digits)
        elif re.match(r"^\d+(st|nd|rd|th)?$", text) and "finish_position" not in entry and i > 0:
            entry["finish_position"] = int(re.match(r"^(\d+)", text).group(1))

        # Finish time (e.g., 28.93, 29.14)
        elif re.match(r"^\d{2}\.\d{2}$", text) and "finish_time" not in entry:
            entry["finish_time"] = float(text)

        # Dog name (alphabetic, possibly with spaces/apostrophes)
        elif re.match(r"^[A-Za-z][A-Za-z\s'\-\.]{2,}$", text) and "dog_name" not in entry:
            entry["dog_name"] = text.strip()

        # Weight (e.g., "32.5", "28.0" — typically in kg range 25-40)
        elif re.match(r"^\d{2}\.\d$", text) and "weight_kg" not in entry:
            weight = float(text)
            if 20 <= weight <= 45:
                entry["weight_kg"] = weight

        # Starting price (e.g., "3/1", "evens", "5/2")
        elif re.match(r"^\d+/\d+$", text) or text.lower() == "evens":
            entry["starting_price"] = text
            if text.lower() == "evens":
                entry["sp_decimal"] = 2.0
            else:
                parts = text.split("/")
                entry["sp_decimal"] = round(int(parts[0]) / int(parts[1]) + 1, 2)

    # Also check for links containing dog names
    for cell in cells:
        link = cell.find("a")
        if link and "dog_name" not in entry:
            name = link.get_text(strip=True)
            if len(name) > 2 and not name.isdigit():
                entry["dog_name"] = name

    # Must have at least dog_name and trap to be valid
    if "dog_name" in entry and "trap" in entry:
        return entry

    return None


def _parse_entry_element(elem) -> dict[str, Any] | None:
    """Parse a non-table element into a race entry."""
    text = elem.get_text(" ", strip=True)
    entry: dict[str, Any] = {}

    # Try to extract trap
    trap_match = re.search(r"Trap\s*(\d)", text, re.IGNORECASE)
    if trap_match:
        entry["trap"] = int(trap_match.group(1))

    # Dog name from link
    link = elem.find("a")
    if link:
        entry["dog_name"] = link.get_text(strip=True)

    # Position
    pos_match = re.search(r"(\d+)(st|nd|rd|th)", text)
    if pos_match:
        entry["finish_position"] = int(pos_match.group(1))

    # Time
    time_match = re.search(r"(\d{2}\.\d{2})", text)
    if time_match:
        entry["finish_time"] = float(time_match.group(1))

    if "dog_name" in entry and "trap" in entry:
        return entry

    return None


def _parse_sp(text: str) -> tuple[str, float | None]:
    """Parse starting price string to (fractional, decimal)."""
    text = text.strip()
    if text.lower() == "evens":
        return ("evens", 2.0)
    match = re.match(r"(\d+)/(\d+)", text)
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        return (text, round(num / den + 1, 2))
    return (text, None)


async def scrape_results(
    track_code: str,
    race_date: date,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """
    Scrape race results for a specific track and date.

    Returns list of race dicts with entries.
    """
    date_str = format_date(race_date)
    url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"

    logger.info("Scraping %s %s -> %s", track_code, race_date, url)

    try:
        html = await fetch_page(url, client)
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

    Reuses HTTP client for efficiency. Rate-limits with configurable delay.
    """
    all_races: list[dict[str, Any]] = []

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
        current = start_date
        total_days = (end_date - start_date).days + 1
        day_num = 0

        while current <= end_date:
            day_num += 1
            races = await scrape_results(track_code, current, client)
            all_races.extend(races)

            if day_num % 10 == 0:
                logger.info(
                    "Progress: %s %d/%d days, %d races found so far",
                    track_code, day_num, total_days, len(all_races),
                )

            current += timedelta(days=1)
            if current <= end_date:
                await asyncio.sleep(delay)

    logger.info(
        "Completed %s: %d days scraped, %d races found",
        track_code, total_days, len(all_races),
    )
    return all_races

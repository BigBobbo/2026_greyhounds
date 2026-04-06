"""
GRI (Greyhound Racing Ireland) scraper using Playwright + BeautifulSoup.

The GRI results page loads race data dynamically via JavaScript.
We must use Playwright, navigate to /results/, select stadium + date from the
form, click "view results", then parse the rendered HTML.

Known GRI track codes (from dropdown):
  CML=Clonmel, CRK=Curraheen Park, DRY=Derry, DBP=Drumbo Park,
  DLK=Dundalk, ECY=Enniscorthy, GLY=Galway, HRX=Harolds Cross,
  KKY=Kilkenny, KWE=Kilkenny Wed Evening, LFD=Lifford, LMK=Limerick,
  LGD=Longford, MGR=Mullingar, NWB=Newbridge, SPK=Shelbourne Park,
  THR=Thurles Park, TRL=Tralee, TRS=Tralee Sat Evening, WFD=Waterford,
  WFE=Waterford Thursday Morning, YGL=Youghal

Table columns:
  Pos. | Trap | Greyhound | SIRE NAME | DAM NAME | Prize | Wt. |
  WinTime | By | Going | EstTime | SP. | Grade | Comm.
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

# Confirmed GRI track codes from the stadium dropdown
GRI_TRACK_CODES = {
    "CML": "Clonmel",
    "CRK": "Curraheen Park",
    "DRY": "Derry",
    "DBP": "Drumbo Park",
    "DLK": "Dundalk",
    "ECY": "Enniscorthy",
    "GLY": "Galway",
    "HRX": "Harolds Cross",
    "KKY": "Kilkenny",
    "KWE": "Kilkenny Wed Evening",
    "LFD": "Lifford",
    "LMK": "Limerick",
    "LGD": "Longford",
    "MGR": "Mullingar",
    "NWB": "Newbridge",
    "SPK": "Shelbourne Park",
    "THR": "Thurles Park",
    "TRL": "Tralee",
    "TRS": "Tralee Sat Evening",
    "WFD": "Waterford",
    "WFE": "Waterford Thursday Morning",
    "YGL": "Youghal",
}


def format_date(d: date) -> str:
    """Format date as DD-Mon-YYYY for GRI (e.g. '05-Apr-2026')."""
    return d.strftime("%d-%b-%Y")


async def _dismiss_cookie_banners(page) -> None:
    """Try to dismiss cookie consent banners."""
    selectors = [
        "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
        "#CybotCookiebotDialogBodyButtonAccept",
        "button:has-text('Accept')",
        "button:has-text('Allow all')",
    ]
    for selector in selectors:
        try:
            btn = await page.query_selector(selector)
            if btn:
                await btn.click()
                await page.wait_for_timeout(1000)
                return
        except Exception:
            continue
    try:
        await page.evaluate("""
            document.querySelectorAll('[id*="Cookiebot"], [id*="cookie"], #CybotCookiebotDialog').forEach(el => el.remove());
            document.body.style.overflow = 'auto';
        """)
    except Exception:
        pass


async def _navigate_and_load_results(page, track_code: str, race_date: date) -> str:
    """
    Navigate to GRI results page, select stadium + date, submit form,
    and return the rendered HTML with race data.
    """
    date_str = format_date(race_date)

    # Block cookie script
    await page.route("**/consent.cookiebot.com/**", lambda route: route.abort())

    await page.goto(RESULTS_URL, timeout=30000, wait_until="networkidle")
    await _dismiss_cookie_banners(page)
    await page.wait_for_timeout(1000)

    # Select stadium from dropdown
    stadium_select = await page.query_selector("#stadium")
    if stadium_select:
        await stadium_select.select_option(value=track_code)
        logger.debug("Selected stadium: %s", track_code)
    else:
        logger.warning("Stadium dropdown not found")

    # Set the date
    date_input = await page.query_selector("#FromDate")
    if date_input:
        await date_input.fill("")
        await date_input.fill(date_str)
        logger.debug("Set date to: %s", date_str)
    else:
        logger.warning("Date input not found")

    await page.wait_for_timeout(500)

    # Click "view results" or "Show Meetings" button
    for selector in [
        "button:has-text('view results')",
        "button:has-text('View Results')",
        "button:has-text('Show Meetings')",
        "input[type='submit']",
    ]:
        try:
            btn = await page.query_selector(selector)
            if btn and await btn.is_visible():
                await btn.click()
                logger.debug("Clicked: %s", selector)
                break
        except Exception:
            continue

    # Wait for results to load
    await page.wait_for_timeout(5000)
    try:
        await page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass

    return await page.content()


def parse_results_page(html: str, track_code: str, race_date: date) -> list[dict[str, Any]]:
    """
    Parse GRI results HTML. The page contains one table per race, each preceded
    by an h4 header like "Race 1 - The Welcome To Clonmel Track A6 / A7 525".

    Table columns (confirmed):
      Pos. | Trap | Greyhound | SIRE NAME | DAM NAME | Prize | Wt. |
      WinTime | By | Going | EstTime | SP. | Grade | Comm.
    """
    soup = BeautifulSoup(html, "html.parser")
    races: list[dict[str, Any]] = []

    # Find all race header elements (h4 tags with "Race N" text)
    race_headers = soup.find_all("h4", string=re.compile(r"Race\s+\d+", re.IGNORECASE))

    if not race_headers:
        # Fallback: find all tables with the expected header structure
        for table in soup.find_all("table"):
            header_row = table.find("tr")
            if header_row and "Pos." in header_row.get_text():
                race_data = _parse_gri_table(table, len(races) + 1, track_code, race_date, None)
                if race_data:
                    races.append(race_data)
        return races

    # Process each race header + its corresponding table
    for header in race_headers:
        header_text = header.get_text(strip=True)

        # Extract race info from header like:
        # "Race 1 - The Welcome To Clonmel Track A6 / A7 525   (Grade : A6/7) Flat 525"
        race_info = _parse_race_header(header_text)

        # Find the next table after this header
        table = header.find_next("table")
        if not table:
            continue

        race_data = _parse_gri_table(table, race_info["race_number"], track_code, race_date, race_info)
        if race_data:
            races.append(race_data)

    # Deduplicate by race_number
    seen_nums = set()
    unique_races = []
    for race in races:
        rn = race.get("race_number")
        if rn not in seen_nums:
            seen_nums.add(rn)
            unique_races.append(race)

    logger.info("Parsed %d races from %s on %s", len(unique_races), track_code, race_date)
    return unique_races


def _parse_race_header(text: str) -> dict[str, Any]:
    """Parse race header text to extract race number, distance, grade."""
    info: dict[str, Any] = {"race_number": None, "distance_m": None, "grade": None, "race_type": "flat"}

    # Race number
    num_match = re.search(r"Race\s+(\d+)", text, re.IGNORECASE)
    if num_match:
        info["race_number"] = int(num_match.group(1))

    # Grade from "(Grade : A6/7)" pattern
    grade_match = re.search(r"Grade\s*:\s*([A-Za-z0-9/]+)", text)
    if grade_match:
        info["grade"] = grade_match.group(1).strip()
    else:
        # Try simpler grade pattern
        grade_match2 = re.search(r"\b([A-S]\d(?:/\d)?|OR|ON|OP|Nov|Novice|Puppy)\b", text, re.IGNORECASE)
        if grade_match2:
            info["grade"] = grade_match2.group(1)

    # Distance — look for 3-4 digit number that's a typical distance
    dist_match = re.search(r"\b(\d{3,4})\b", text)
    if dist_match:
        dist = int(dist_match.group(1))
        if 200 <= dist <= 1000:  # valid greyhound distance range
            info["distance_m"] = dist

    # Race type
    if re.search(r"hurdle|hurdles", text, re.IGNORECASE):
        info["race_type"] = "hurdle"

    return info


def _parse_gri_table(
    table, race_number: int, track_code: str, race_date: date, race_info: dict | None
) -> dict[str, Any] | None:
    """
    Parse a GRI result table with known column order:
    Pos. | Trap | Greyhound | SIRE NAME | DAM NAME | Prize | Wt. |
    WinTime | By | Going | EstTime | SP. | Grade | Comm.
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return None

    # Detect column indices from header row
    header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    col_map = _build_column_map(header_cells)

    if not col_map:
        return None

    entries = []
    winner_time = None
    trap_num = 0

    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        texts = [c.get_text(strip=True) for c in cells]

        if len(texts) < 5:
            continue

        trap_num += 1
        entry = _extract_entry_mapped(texts, cells, col_map)
        if entry and entry.get("dog_name"):
            # Assign trap number from row order if not already set
            # In GRI results, rows are ordered by trap (1-6/8)
            if not entry.get("trap"):
                entry["trap"] = trap_num
            # Track winner time for calculating estimated times
            if entry.get("finish_position") == 1 and entry.get("finish_time"):
                winner_time = entry["finish_time"]
            entries.append(entry)

    if not entries:
        return None

    # Extract race-level info
    distance = race_info.get("distance_m") if race_info else None
    grade = race_info.get("grade") if race_info else None
    race_type = race_info.get("race_type", "flat") if race_info else "flat"

    # Try to get going from entries
    going = None
    for e in entries:
        if e.get("going"):
            going = e["going"]
            break

    # Get prize from first entry
    prize = None
    for e in entries:
        if e.get("prize_money"):
            prize = e["prize_money"]
            break

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


def _build_column_map(headers: list[str]) -> dict[str, int]:
    """Map column names to indices from the header row."""
    col_map = {}
    for i, h in enumerate(headers):
        h_lower = h.lower().strip().rstrip(".")

        if h_lower in ("pos", "position"):
            col_map["position"] = i
        elif h_lower == "trap":
            col_map["trap"] = i
        elif h_lower in ("greyhound", "dog", "name"):
            col_map["greyhound"] = i
        elif h_lower == "sire name" or h_lower == "sire":
            col_map["sire"] = i
        elif h_lower == "dam name" or h_lower == "dam":
            col_map["dam"] = i
        elif h_lower == "prize":
            col_map["prize"] = i
        elif h_lower in ("wt", "weight"):
            col_map["weight"] = i
        elif h_lower in ("wintime", "win time", "time"):
            col_map["wintime"] = i
        elif h_lower == "by":
            col_map["by"] = i
        elif h_lower == "going":
            col_map["going"] = i
        elif h_lower in ("esttime", "est time", "est.time"):
            col_map["esttime"] = i
        elif h_lower in ("sp", "sp."):
            col_map["sp"] = i
        elif h_lower == "grade":
            col_map["grade_col"] = i
        elif h_lower in ("comm", "comm.", "comment", "remarks"):
            col_map["comment"] = i

    # Must at least have greyhound column to be useful
    if "greyhound" not in col_map:
        return {}

    return col_map


def _extract_entry_mapped(texts: list[str], cells, col_map: dict) -> dict[str, Any]:
    """Extract entry data using confirmed column mapping."""
    entry: dict[str, Any] = {}

    def get(key: str) -> str:
        idx = col_map.get(key)
        if idx is not None and idx < len(texts):
            return texts[idx].strip()
        return ""

    # Position
    pos_text = get("position")
    pos_match = re.match(r"(\d+)", pos_text)
    if pos_match:
        entry["finish_position"] = int(pos_match.group(1))

    # Trap
    trap_text = get("trap")
    trap_match = re.match(r"(\d+)", trap_text)
    if trap_match:
        entry["trap"] = int(trap_match.group(1))

    # Greyhound name (this is the DOG, not sire/dam)
    dog_name = get("greyhound")
    # Also check for link text
    greyhound_idx = col_map.get("greyhound")
    if greyhound_idx is not None and greyhound_idx < len(cells):
        link = cells[greyhound_idx].find("a")
        if link:
            dog_name = link.get_text(strip=True)
    if dog_name:
        entry["dog_name"] = dog_name.upper()

    # Sire and Dam
    sire = get("sire")
    dam = get("dam")
    if sire:
        entry["sire_name"] = sire
    if dam:
        entry["dam_name"] = dam

    # Prize
    prize_text = get("prize")
    prize_match = re.search(r"[€£]?([\d,]+(?:\.\d+)?)", prize_text)
    if prize_match:
        entry["prize_money"] = float(prize_match.group(1).replace(",", ""))

    # Weight
    wt_text = get("weight")
    wt_match = re.match(r"(\d+\.?\d*)", wt_text)
    if wt_match:
        entry["weight_kg"] = float(wt_match.group(1))

    # Win Time (the winner's time, same for all entries in a race)
    wintime_text = get("wintime")
    wt_time_match = re.match(r"(\d+\.\d+)", wintime_text)
    if wt_time_match:
        entry["win_time"] = float(wt_time_match.group(1))

    # Estimated Time (individual dog's time)
    esttime_text = get("esttime")
    est_match = re.match(r"(\d+\.\d+)", esttime_text)
    if est_match:
        entry["finish_time"] = float(est_match.group(1))

    # Beaten by
    by_text = get("by")
    if by_text and by_text not in ("", "-"):
        # Could be "3.75L", "nk", "hd", "sh", "dnf"
        dist_match = re.match(r"([\d.]+)L?", by_text)
        if dist_match:
            entry["beaten_distance"] = float(dist_match.group(1))

    # Going
    going_text = get("going")
    if going_text:
        entry["going"] = going_text

    # SP
    sp_text = get("sp")
    if sp_text:
        entry["starting_price"] = sp_text
        entry["sp_decimal"] = _parse_sp_decimal(sp_text)

    # Comment
    comment_text = get("comment")
    if comment_text:
        entry["comment"] = comment_text

    # Grade at entry
    grade_text = get("grade_col")
    if grade_text:
        entry["grade_at_entry"] = grade_text

    return entry


def _parse_sp_decimal(sp_text: str) -> float | None:
    """Convert SP text to decimal odds."""
    sp_text = sp_text.strip().lower()
    # Remove F (favourite) or JF (joint favourite) suffixes
    sp_clean = re.sub(r"[fj]+$", "", sp_text, flags=re.IGNORECASE).strip()

    if sp_clean in ("evens", "evs"):
        return 2.0
    match = re.match(r"(\d+)/(\d+)", sp_clean)
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        if den > 0:
            return round(num / den + 1, 2)
    return None


async def scrape_results(track_code: str, race_date: date) -> list[dict[str, Any]]:
    """Scrape race results for a specific track and date using Playwright."""
    from playwright.async_api import async_playwright

    logger.info("Scraping %s %s", track_code, race_date)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                )
                page = await context.new_page()
                html = await _navigate_and_load_results(page, track_code, race_date)
                races = parse_results_page(html, track_code, race_date)
                return races
            finally:
                await browser.close()
    except Exception as e:
        logger.error("Failed to scrape %s %s: %s", track_code, race_date, e)
        return []


async def scrape_date_range(
    track_code: str,
    start_date: date,
    end_date: date,
    delay: float = 3.0,
) -> list[dict[str, Any]]:
    """Scrape results for a track across a date range, reusing one browser."""
    from playwright.async_api import async_playwright

    all_races: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        page = await context.new_page()
        # Block cookies once for the session
        await page.route("**/consent.cookiebot.com/**", lambda route: route.abort())

        try:
            current = start_date
            total_days = (end_date - start_date).days + 1
            day_num = 0

            while current <= end_date:
                day_num += 1
                try:
                    html = await _navigate_and_load_results(page, track_code, current)
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

    logger.info("Completed %s: %d days, %d races", track_code, total_days, len(all_races))
    return all_races


async def discover_track_codes() -> list[dict[str, str]]:
    """Return the confirmed GRI track codes."""
    return [{"code": code, "name": name} for code, name in GRI_TRACK_CODES.items()]

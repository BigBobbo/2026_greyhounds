"""
GRI (Greyhound Racing Ireland) scraper using httpx + BeautifulSoup.

NO Playwright needed — the race data is server-rendered in the HTML.

URL pattern: https://www.grireland.ie/results/view-results/?track={CODE}&date={DD-Mon-YYYY}

Trap numbers are encoded as <img alt="Trap N"> tags.

Known GRI track codes (from dropdown):
  CML=Clonmel, CRK=Curraheen Park, DRY=Derry, DBP=Drumbo Park,
  DLK=Dundalk, ECY=Enniscorthy, GLY=Galway, HRX=Harolds Cross,
  KKY=Kilkenny, KWE=Kilkenny Wed Evening, LFD=Lifford, LMK=Limerick,
  LGD=Longford, MGR=Mullingar, NWB=Newbridge, SPK=Shelbourne Park,
  THR=Thurles Park, TRL=Tralee, TRS=Tralee Sat Evening, WFD=Waterford,
  WFE=Waterford Thursday Morning, YGL=Youghal
"""

import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

GRI_BASE_URL = "https://www.grireland.ie"
VIEW_RESULTS_URL = f"{GRI_BASE_URL}/results/view-results/"

DEFAULT_HEADERS = {
    "User-Agent": "Greyhound-Research-Bot/1.0 (race prediction research)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

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


def parse_results_page(html: str, track_code: str, race_date: date) -> list[dict[str, Any]]:
    """
    Parse a GRI results page. Race data is server-rendered HTML with tables.

    Each race has:
    - An <h4> header: "Race 1 - Name 525 (Grade : A6/7) Flat 525"
    - A <table class='igb-tbl'> with rows for each dog

    Trap numbers are <img alt="Trap N"> tags.
    """
    soup = BeautifulSoup(html, "html.parser")
    races: list[dict[str, Any]] = []

    # Find all race header elements
    race_headers = soup.find_all("h4", string=re.compile(r"Race\s+\d+", re.IGNORECASE))

    if not race_headers:
        logger.debug("No race headers found for %s %s", track_code, race_date)
        return []

    for header in race_headers:
        header_text = header.get_text(strip=True)
        race_info = _parse_race_header(header_text)

        # Find the next table after this header
        table = header.find_next("table", class_="igb-tbl")
        if not table:
            table = header.find_next("table")
        if not table:
            continue

        entries = _parse_result_table(table)
        if not entries:
            continue

        # Get going from first entry
        going = next((e.get("going") for e in entries if e.get("going")), None)

        races.append({
            "race_number": race_info["race_number"],
            "race_date": race_date,
            "track_code": track_code,
            "distance_m": race_info["distance_m"],
            "grade": race_info["grade"],
            "race_type": race_info["race_type"],
            "going": going,
            "prize_money": max((e.get("prize_money") or 0 for e in entries), default=None),
            "entries": entries,
        })

    logger.info("Parsed %d races from %s on %s", len(races), track_code, race_date)
    return races


def _parse_race_header(text: str) -> dict[str, Any]:
    """Parse header like 'Race 1 - Fáilte Go Dtí 525 (Grade : A8/9) Flat 525'."""
    info: dict[str, Any] = {
        "race_number": None, "distance_m": None, "grade": None, "race_type": "flat",
    }

    num_match = re.search(r"Race\s+(\d+)", text, re.IGNORECASE)
    if num_match:
        info["race_number"] = int(num_match.group(1))

    grade_match = re.search(r"Grade\s*:\s*([A-Za-z0-9/]+)", text)
    if grade_match:
        info["grade"] = grade_match.group(1).strip()

    # Distance: last 3-4 digit number in the header (appears twice, take any)
    distances = re.findall(r"\b(\d{3,4})\b", text)
    for d in distances:
        val = int(d)
        if 200 <= val <= 1000:
            info["distance_m"] = val
            break

    if re.search(r"hurdle|hurdles", text, re.IGNORECASE):
        info["race_type"] = "hurdle"

    return info


def _parse_result_table(table) -> list[dict[str, Any]]:
    """
    Parse a GRI result table. Columns:
    Pos. | Trap | Greyhound | SIRE NAME | DAM NAME | Prize | Wt. |
    Win Time | By | Going | Est Time | SP. | Grade | Comm.

    Trap is an <img alt="Trap N"> tag.
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    entries = []
    for row in rows[1:]:  # skip header
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        entry = _parse_row(cells)
        if entry and entry.get("dog_name"):
            entries.append(entry)

    return entries


def _parse_row(cells) -> dict[str, Any]:
    """Parse a single result row from known column positions."""
    entry: dict[str, Any] = {}

    # Col 0: Position (e.g. "1.", "2.")
    pos_text = cells[0].get_text(strip=True)
    pos_match = re.match(r"(\d+)", pos_text)
    if pos_match:
        entry["finish_position"] = int(pos_match.group(1))

    # Col 1: Trap — encoded as <img alt="Trap 4">
    trap_img = cells[1].find("img")
    if trap_img:
        alt = trap_img.get("alt", "")
        trap_match = re.search(r"Trap\s*(\d+)", alt)
        if trap_match:
            entry["trap"] = int(trap_match.group(1))

    # Col 2: Greyhound name (contains <a> link)
    dog_link = cells[2].find("a")
    if dog_link:
        entry["dog_name"] = dog_link.get_text(strip=True).upper()

    # Col 3: Sire name (in next <td> with pedigree-sire span)
    sire_cell = cells[2].find_next("td")
    if sire_cell:
        sire_span = sire_cell.find(class_="viewresults-pedigree-sire")
        if sire_span:
            sire_link = sire_span.find("a")
            if sire_link:
                entry["sire_name"] = sire_link.get_text(strip=True)

    # Col 4: Dam name
    dam_span = cells[2].find_next(class_="viewresults-pedigree-dam")
    if dam_span:
        dam_link = dam_span.find("a")
        if dam_link:
            entry["dam_name"] = dam_link.get_text(strip=True)

    # Now handle the remaining columns — but the sire/dam cells are embedded
    # oddly in the HTML (missing closing </td> on greyhound cell).
    # The actual separate <td> cells after the pedigree section are:
    # Prize | Wt. | WinTime | By | Going | EstTime | SP. | Grade | Comm.
    #
    # Because of the broken HTML, let's find all <td> in the row and count
    # from the end, since the last columns are consistent.
    all_tds = cells
    num_tds = len(all_tds)

    # Work backwards from the known tail columns
    # Last column = Comm (index -1)
    # SP = -2, Grade = -3, EstTime = -4, Going = -5, By = -6
    # WinTime = -7, Wt = -8, Prize = -9

    def get_td_text(idx: int) -> str:
        if idx < 0:
            idx = num_tds + idx
        if 0 <= idx < num_tds:
            return all_tds[idx].get_text(strip=True)
        return ""

    # Comment
    comment = get_td_text(-1)
    if comment:
        entry["comment"] = comment

    # Grade at entry
    grade = get_td_text(-2)
    if grade:
        entry["grade_at_entry"] = grade

    # SP
    sp_text = get_td_text(-3)
    if sp_text:
        entry["starting_price"] = sp_text
        entry["sp_decimal"] = _parse_sp_decimal(sp_text)

    # Est Time (individual dog's time)
    est_text = get_td_text(-4)
    est_match = re.match(r"([\d.]+)", est_text)
    if est_match:
        entry["finish_time"] = float(est_match.group(1))

    # Going
    going_text = get_td_text(-5)
    if going_text:
        entry["going"] = going_text

    # By (beaten distance)
    by_text = get_td_text(-6)
    if by_text and by_text.strip() not in ("", "&nbsp;"):
        dist_match = re.match(r"([\d.]+)", by_text)
        if dist_match:
            entry["beaten_distance"] = float(dist_match.group(1))

    # Win Time
    wt_text = get_td_text(-7)
    wt_match = re.match(r"([\d.]+)", wt_text)
    if wt_match:
        entry["win_time"] = float(wt_match.group(1))

    # Weight
    weight_text = get_td_text(-8)
    weight_match = re.match(r"(\d+\.?\d*)", weight_text)
    if weight_match:
        entry["weight_kg"] = float(weight_match.group(1))

    # Prize
    prize_text = get_td_text(-9)
    prize_match = re.search(r"([\d,]+\.?\d*)", prize_text)
    if prize_match:
        entry["prize_money"] = float(prize_match.group(1).replace(",", ""))

    return entry


def _parse_sp_decimal(sp_text: str) -> float | None:
    """Convert SP text to decimal odds."""
    sp_clean = re.sub(r"[FfJj]+$", "", sp_text).strip()
    if sp_clean.lower() in ("evens", "evs"):
        return 2.0
    match = re.match(r"(\d+)/(\d+)", sp_clean)
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        if den > 0:
            return round(num / den + 1, 2)
    return None


async def scrape_results(
    track_code: str,
    race_date: date,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Scrape race results for a specific track and date."""
    date_str = format_date(race_date)
    url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"

    close_client = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30)
        close_client = True

    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            logger.warning("Got %d for %s", resp.status_code, url)
            return []
        html = resp.text
    except Exception as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return []
    finally:
        if close_client:
            await client.aclose()

    return parse_results_page(html, track_code, race_date)


async def scrape_date_range(
    track_code: str,
    start_date: date,
    end_date: date,
    delay: float = 1.0,
) -> list[dict[str, Any]]:
    """Scrape results for a track across a date range. Fast — no browser needed."""
    all_races: list[dict[str, Any]] = []

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
        current = start_date
        total_days = (end_date - start_date).days + 1
        day_num = 0

        while current <= end_date:
            day_num += 1
            races = await scrape_results(track_code, current, client)
            all_races.extend(races)

            if day_num % 50 == 0:
                logger.info(
                    "Progress: %s %d/%d days, %d races found",
                    track_code, day_num, total_days, len(all_races),
                )

            current += timedelta(days=1)
            if current <= end_date:
                await asyncio.sleep(delay)

    logger.info("Completed %s: %d days, %d races", track_code, total_days, len(all_races))
    return all_races


async def discover_track_codes() -> list[dict[str, str]]:
    """Return the confirmed GRI track codes."""
    return [{"code": code, "name": name} for code, name in GRI_TRACK_CODES.items()]

"""
GRI (Greyhound Racing Ireland) scraper using httpx + BeautifulSoup.

NO Playwright needed — the race data is server-rendered in the HTML.

URL patterns:
  Results:        /results/view-results/?track={CODE}&date={DD-Mon-YYYY}
  Card summary:   /racing/upcoming-race-cards/upcoming-race-card-summary/?track={CODE}&date={DD-Mon-YYYY}
  Card form:      /racing/upcoming-race-cards/upcoming-race-card-summary/view-race-form/?Track={CODE}&Date={DD-Mon-YYYY}&RaceNumber={N}

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
from datetime import date, time, timedelta
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ScrapeFetchError(Exception):
    """Raised when a GRI page cannot be fetched (network error or non-200)."""


class ParseStructureError(Exception):
    """Raised when a fetched GRI page no longer matches the expected markup.

    This is deliberately loud: a silent `return []` here looks identical to a
    quiet no-racing day and lets scrape jobs report success while storing
    nothing.
    """


GRI_BASE_URL = "https://www.grireland.ie"
VIEW_RESULTS_URL = f"{GRI_BASE_URL}/results/view-results/"
VIEW_CARD_URL = f"{GRI_BASE_URL}/racing/upcoming-race-cards/upcoming-race-card-summary/"
VIEW_CARD_FORM_URL = f"{GRI_BASE_URL}/racing/upcoming-race-cards/upcoming-race-card-summary/view-race-form/"

DEFAULT_HEADERS = {
    "User-Agent": "Greyhound-Research-Bot/1.0 (race prediction research)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Transient-failure retry policy (audit task E7): network/transport errors and
# 5xx responses are retried with exponential backoff. 4xx responses and
# ParseStructureError are NOT retried — repeating a wrong request (or
# re-parsing the same broken markup) cannot help. Module-level constants so
# tests can shrink the delays.
MAX_FETCH_ATTEMPTS = 3
RETRY_BACKOFF_S: tuple[float, ...] = (2.0, 4.0, 8.0)


async def _fetch_page(url: str, client: httpx.AsyncClient | None = None) -> str:
    """GET a GRI page, retrying transient failures with exponential backoff.

    Transient = network error or 5xx response: up to MAX_FETCH_ATTEMPTS
    attempts, sleeping RETRY_BACKOFF_S[attempt] between them. Any other
    non-200 response (4xx etc.) raises immediately without retrying.

    Raises ScrapeFetchError when the page cannot be fetched.
    """
    close_client = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30)
        close_client = True

    try:
        last_error: Exception | None = None
        reason = "unknown"
        for attempt in range(MAX_FETCH_ATTEMPTS):
            try:
                resp = await client.get(url)
            except Exception as e:
                last_error = e
                reason = f"network error: {e}"
            else:
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code < 500:
                    # 4xx (or a 3xx that survived follow_redirects): the
                    # request itself is wrong — retrying cannot help.
                    raise ScrapeFetchError(f"Got {resp.status_code} for {url}")
                last_error = None
                reason = f"HTTP {resp.status_code}"
            if attempt + 1 < MAX_FETCH_ATTEMPTS:
                backoff = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
                logger.warning(
                    "Transient fetch failure for %s (%s) — attempt %d/%d, "
                    "retrying in %.0fs",
                    url, reason, attempt + 1, MAX_FETCH_ATTEMPTS, backoff,
                )
                await asyncio.sleep(backoff)
        msg = f"Failed to fetch {url} after {MAX_FETCH_ATTEMPTS} attempts ({reason})"
        if last_error is not None:
            raise ScrapeFetchError(msg) from last_error
        raise ScrapeFetchError(msg)
    finally:
        if close_client:
            await client.aclose()

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


_GRI_DOG_ID_RE = re.compile(r"[?&]gid=(\d+)", re.IGNORECASE)


def _extract_gri_id(href: str | None) -> str | None:
    """Pull the GRI dog id out of a greyhound-details href querystring.

    GRI dog-detail links look like
    `/results/greyhound-search/greyhound-details/?gid=12345` — the gid is a
    stable per-dog identifier, unlike the name (two dogs can share a name).
    Returns None when the href carries no gid.
    """
    if not href:
        return None
    m = _GRI_DOG_ID_RE.search(href)
    return m.group(1) if m else None


def _has_gri_page_anchor(soup: BeautifulSoup) -> bool:
    """True if the page carries GRI structural anchors.

    A genuine GRI results/card page always renders the search form with its
    track dropdown (ASP.NET select whose id/name contains 'track') and/or
    `igb-tbl` tables. A 200 page with neither is not a GRI racing page at
    all (error page, redirect target, site redesign) and must not be treated
    as a quiet "no racing today".
    """
    if soup.find("select", attrs={"name": re.compile("track", re.IGNORECASE)}):
        return True
    if soup.find("select", id=re.compile("track", re.IGNORECASE)):
        return True
    if soup.find("table", class_="igb-tbl"):
        return True
    return False


def _has_race_like_text(html: str) -> bool:
    """True if the raw HTML mentions race content ('Race 1' etc.)."""
    return re.search(r"Race\s+\d+", html, re.IGNORECASE) is not None


def parse_results_page(html: str, track_code: str, race_date: date) -> list[dict[str, Any]]:
    """
    Parse a GRI results page. Race data is server-rendered HTML with tables.

    Each race has:
    - An <h4> header: "Race 1 - Name 525 (Grade : A6/7) Flat 525"
    - A <table class='igb-tbl'> with rows for each dog

    Trap numbers are <img alt="Trap N"> tags.

    Raises ParseStructureError when the page contains race-like content (or
    lacks GRI page anchors entirely) but no races could be parsed — i.e. the
    markup changed and the parser is broken, which must be loud rather than
    look like a quiet no-racing day.
    """
    soup = BeautifulSoup(html, "html.parser")
    races: list[dict[str, Any]] = []

    # Find all race header elements
    race_headers = soup.find_all("h4", string=re.compile(r"Race\s+\d+", re.IGNORECASE))

    if not race_headers:
        if _has_race_like_text(html):
            raise ParseStructureError(
                f"Results page for {track_code} {race_date} contains race-like "
                "text but no <h4> race headers were found — markup has changed"
            )
        if not _has_gri_page_anchor(soup):
            raise ParseStructureError(
                f"Page for {track_code} {race_date} lacks GRI structural "
                "anchors (track dropdown / igb-tbl) — not a GRI results page"
            )
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

    if not races:
        raise ParseStructureError(
            f"Results page for {track_code} {race_date} has "
            f"{len(race_headers)} race header(s) but 0 races could be parsed "
            "— markup has changed"
        )

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

    # Distance: the true distance repeats at the END of the header, while
    # sponsor/race names can inject bogus in-range numbers earlier (e.g.
    # "The 600 Final" run over 525) — so prefer the LAST in-range match.
    # Range extends to 1100: Irish marathons run 1010/1035 yards.
    distances = re.findall(r"\b(\d{3,4})\b", text)
    for d in reversed(distances):
        val = int(d)
        if 200 <= val <= 1100:
            info["distance_m"] = val
            break

    if re.search(r"hurdle|hurdles", text, re.IGNORECASE):
        info["race_type"] = "hurdle"

    return info


# Sanity ranges applied at parse time — out-of-range values are dropped (None)
# rather than stored as garbage.
WEIGHT_RANGE_KG = (20.0, 40.0)
TIME_RANGE_S = (15.0, 70.0)

# Legacy positional mapping for header-less tables, counted from the END of
# the row because the malformed greyhound cell (missing </td>) makes leading
# indices unreliable. Tail order per the documented columns:
# ... Prize | Wt. | Win Time | By | Going | Est Time | SP. | Grade | Comm.
_LEGACY_TAIL_OFFSETS = {
    "prize": -9,
    "weight": -8,
    "win_time": -7,
    "by": -6,
    "going": -5,
    "est_time": -4,
    "sp": -3,
    "grade": -2,
    "comment": -1,
}

_REQUIRED_RESULT_COLUMNS = {"position", "trap", "greyhound", "weight", "sp", "grade", "comment"}


def _sanity_check(value: float | None, lo: float, hi: float, label: str) -> float | None:
    """Return value if within [lo, hi], else None with a warning."""
    if value is None:
        return None
    if lo <= value <= hi:
        return value
    logger.warning("Discarding out-of-range %s: %s (expected %s-%s)", label, value, lo, hi)
    return None


def _match_result_header(text: str) -> str | None:
    """Map a header cell's text onto a canonical column name."""
    norm = re.sub(r"[.\s]+", " ", text).strip().lower()
    if not norm:
        return None
    if norm.startswith("pos"):
        return "position"
    if "trap" in norm:
        return "trap"
    if "greyhound" in norm or norm == "dog":
        return "greyhound"
    if "sire" in norm:
        return "sire"
    if "dam" in norm:
        return "dam"
    if "prize" in norm:
        return "prize"
    if norm == "wt" or "weight" in norm:
        return "weight"
    if "win time" in norm or "wintime" in norm:
        return "win_time"
    if norm == "by":
        return "by"
    if "going" in norm:
        return "going"
    if norm.startswith("est"):
        return "est_time"
    if norm == "sp":
        return "sp"
    if "grade" in norm:
        return "grade"
    if norm.startswith("comm"):
        return "comment"
    return None


def _build_result_column_map(header_cells) -> dict[str, int] | None:
    """Build a column-name -> index map from a results table header row.

    Returns None when the required columns can't all be located.
    """
    col_map: dict[str, int] = {}
    for idx, cell in enumerate(header_cells):
        name = _match_result_header(cell.get_text(" ", strip=True))
        if name and name not in col_map:
            col_map[name] = idx
    if not _REQUIRED_RESULT_COLUMNS.issubset(col_map):
        return None
    return col_map


def _parse_result_table(table) -> list[dict[str, Any]]:
    """
    Parse a GRI result table. Columns:
    Pos. | Trap | Greyhound | SIRE NAME | DAM NAME | Prize | Wt. |
    Win Time | By | Going | Est Time | SP. | Grade | Comm.

    Columns are addressed by header text (name -> index map built from the
    header row), so an inserted/reordered column cannot silently shift
    values. If the table has a header row but the expected columns are
    missing, ParseStructureError is raised. Only when no header row exists
    at all do we fall back to the legacy positional mapping.

    Trap is an <img alt="Trap N"> tag.
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    th_cells = rows[0].find_all("th")
    header_cells = th_cells or rows[0].find_all("td")
    col_map = _build_result_column_map(header_cells)
    if col_map is None:
        if th_cells:
            raise ParseStructureError(
                "Results table header row found but expected columns "
                f"({', '.join(sorted(_REQUIRED_RESULT_COLUMNS))}) could not "
                f"be matched in: {[c.get_text(strip=True) for c in th_cells]}"
            )
        logger.warning(
            "Results table has no header row — falling back to legacy "
            "positional column mapping"
        )

    entries = []
    for row in rows[1:]:  # skip header
        cells = row.find_all("td")
        if col_map is None and len(cells) < 10:
            continue
        if not cells:
            continue

        entry = _parse_row(row, cells, col_map)
        if entry and entry.get("dog_name"):
            entries.append(entry)

    return entries


def _parse_row(row, cells, col_map: dict[str, int] | None) -> dict[str, Any]:
    """Parse a single result row.

    With a header-derived `col_map` (the normal path) columns are addressed
    by name. NOTE: GRI emits malformed HTML — the greyhound cell is missing
    its closing </td>, so the pedigree cells end up NESTED inside it — but
    `find_all("td")` still returns every cell in document order, so the
    header indices line up.

    With `col_map=None` (legacy header-less fallback) the head columns are
    Pos=0 / Trap=1 / Greyhound=2 and tail columns are counted from the end
    of the row per `_LEGACY_TAIL_OFFSETS`.
    """
    entry: dict[str, Any] = {}
    num_tds = len(cells)

    def get_cell(field: str):
        if col_map is not None:
            idx = col_map.get(field)
        else:
            idx = {"position": 0, "trap": 1, "greyhound": 2}.get(field)
            if idx is None:
                offset = _LEGACY_TAIL_OFFSETS.get(field)
                idx = num_tds + offset if offset is not None else None
        if idx is None or not (0 <= idx < num_tds):
            return None
        return cells[idx]

    def get_text(field: str) -> str:
        cell = get_cell(field)
        return cell.get_text(strip=True) if cell is not None else ""

    # Position (e.g. "1.", "2."). Non-finishers have no numeric position.
    pos_match = re.match(r"(\d+)", get_text("position"))
    if pos_match:
        entry["finish_position"] = int(pos_match.group(1))

    # Trap — encoded as <img alt="Trap 4">
    trap_cell = get_cell("trap")
    trap_img = trap_cell.find("img") if trap_cell is not None else None
    if trap_img:
        trap_match = re.search(r"Trap\s*(\d+)", trap_img.get("alt", ""))
        if trap_match:
            entry["trap"] = int(trap_match.group(1))

    # Greyhound name (contains <a> link). With the malformed markup the
    # pedigree cells are nested inside this cell, but the dog's own link
    # comes first in document order.
    dog_cell = get_cell("greyhound")
    dog_link = dog_cell.find("a") if dog_cell is not None else None
    if dog_link:
        entry["dog_name"] = dog_link.get_text(strip=True).upper()
        gri_id = _extract_gri_id(dog_link.get("href"))
        if gri_id:
            entry["gri_id"] = gri_id

    # Sire / Dam — search within THIS ROW's subtree only. (The old code used
    # cells[2].find_next(...), which walks the whole document forward: a row
    # missing its pedigree span silently grabbed the NEXT dog's pedigree.)
    # Names are uppercased to match how dog names are stored.
    sire_span = row.find(class_="viewresults-pedigree-sire")
    if sire_span:
        sire_link = sire_span.find("a")
        if sire_link:
            sire_name = sire_link.get_text(strip=True).upper()
            if sire_name:
                entry["sire_name"] = sire_name

    dam_span = row.find(class_="viewresults-pedigree-dam")
    if dam_span:
        dam_link = dam_span.find("a")
        if dam_link:
            dam_name = dam_link.get_text(strip=True).upper()
            if dam_name:
                entry["dam_name"] = dam_name

    # Comment
    comment = get_text("comment")
    if comment:
        entry["comment"] = comment

    # Grade at entry
    grade = get_text("grade")
    if grade:
        entry["grade_at_entry"] = grade

    # SP
    sp_text = get_text("sp")
    if sp_text:
        entry["starting_price"] = sp_text
        entry["sp_decimal"] = _parse_sp_decimal(sp_text)

    # Est Time (individual dog's time)
    est_match = re.match(r"([\d.]+)", get_text("est_time"))
    if est_match:
        entry["finish_time"] = _sanity_check(
            float(est_match.group(1)), *TIME_RANGE_S, "finish_time (s)"
        )

    # Going
    going_text = get_text("going")
    if going_text:
        entry["going"] = going_text

    # By (beaten distance)
    by_text = get_text("by")
    if by_text and by_text.strip() not in ("", "&nbsp;"):
        dist_match = re.match(r"([\d.]+)", by_text)
        if dist_match:
            entry["beaten_distance"] = float(dist_match.group(1))

    # Win Time
    wt_match = re.match(r"([\d.]+)", get_text("win_time"))
    if wt_match:
        entry["win_time"] = _sanity_check(
            float(wt_match.group(1)), *TIME_RANGE_S, "win_time (s)"
        )

    # Weight
    weight_match = re.match(r"(\d+\.?\d*)", get_text("weight"))
    if weight_match:
        entry["weight_kg"] = _sanity_check(
            float(weight_match.group(1)), *WEIGHT_RANGE_KG, "weight (kg)"
        )

    # Prize
    prize_match = re.search(r"([\d,]+\.?\d*)", get_text("prize"))
    if prize_match:
        entry["prize_money"] = float(prize_match.group(1).replace(",", ""))

    return entry


def _parse_sp_decimal(sp_text: str) -> float | None:
    """Convert SP text to decimal odds (e.g. 'evens' -> 2.0, '5/2F' -> 3.5)."""
    sp_clean = re.sub(r"[FfJj]+$", "", sp_text).strip()
    if sp_clean.lower() in ("evens", "evs"):
        return 2.0
    match = re.match(r"(\d+)/(\d+)", sp_clean)
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        if den > 0:
            decimal = round(num / den + 1, 2)
            if decimal <= 1.0:
                logger.warning("Discarding non-sensical SP %r -> %s", sp_text, decimal)
                return None
            return decimal
    return None


async def scrape_results(
    track_code: str,
    race_date: date,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Scrape race results for a specific track and date.

    Raises ScrapeFetchError on network failure or non-200 response (with
    retry/backoff for transient failures — see `_fetch_page`), and
    propagates ParseStructureError from the parser — callers record these
    as failed (track, date) pairs instead of silently logging success.
    """
    date_str = format_date(race_date)
    url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"
    html = await _fetch_page(url, client)
    return parse_results_page(html, track_code, race_date)


async def scrape_date_range(
    track_code: str,
    start_date: date,
    end_date: date,
    delay: float = 1.0,
) -> list[dict[str, Any]]:
    """Scrape results for a track across a date range. Fast — no browser needed.

    NOTE: ScrapeFetchError/ParseStructureError from any single day propagate
    and abort the range — callers needing per-day fault tolerance should loop
    over `scrape_results` themselves (as the API/scheduler jobs do).
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


# ---------------------------------------------------------------------------
# Upcoming race-card scrapers (Tier 1: summary; Tier 2: per-race form detail)
# ---------------------------------------------------------------------------


def _parse_card_header(text: str) -> dict[str, Any]:
    """Parse a card summary race header.

    Example: "Race 1 WELCOME TO ENNISCORTHY GREYHOUND STADIUM 20:00 Approx.
              (525 Yds. Flat) (Race Grade : A3)"
    """
    info: dict[str, Any] = {
        "race_number": None,
        "race_time": None,
        "distance_m": None,
        "grade": None,
        "race_type": "flat",
    }

    num_match = re.search(r"Race\s+(\d+)", text, re.IGNORECASE)
    if num_match:
        info["race_number"] = int(num_match.group(1))

    time_match = re.search(r"(\d{1,2}):(\d{2})\s*Approx", text, re.IGNORECASE)
    if time_match:
        hh, mm = int(time_match.group(1)), int(time_match.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            info["race_time"] = time(hh, mm)

    grade_match = re.search(r"Grade\s*:\s*([A-Za-z0-9/]+)", text)
    if grade_match:
        info["grade"] = grade_match.group(1).strip()

    # Distance: stored in yards on Irish cards but convention here is the raw number.
    # Range extends to 1100: Irish marathons run 1010/1035 yards.
    dist_match = re.search(r"\(\s*(\d{3,4})\s*(?:Yds|Yards|m)\b", text, re.IGNORECASE)
    if dist_match:
        val = int(dist_match.group(1))
        if 200 <= val <= 1100:
            info["distance_m"] = val
    if info["distance_m"] is None:
        # Fallback: prefer the LAST in-range 3-4 digit number — sponsor/race
        # names can inject bogus numbers earlier (e.g. "The 600 Final").
        for d in reversed(re.findall(r"\b(\d{3,4})\b", text)):
            v = int(d)
            if 200 <= v <= 1100:
                info["distance_m"] = v
                break

    if re.search(r"hurdle", text, re.IGNORECASE):
        info["race_type"] = "hurdle"

    return info


def parse_card_page(html: str, track_code: str, race_date: date) -> list[dict[str, Any]]:
    """Parse an upcoming-race-card summary page.

    Each race is a `<table class="igb-tbl">` whose first row's `<th>` carries
    the race header. Subsequent rows contain trap-img + dog-link pairs.

    Returns the same dict shape as `parse_results_page` but with `entries`
    containing only `trap` and `dog_name` (no finishing/SP data).
    """
    soup = BeautifulSoup(html, "html.parser")
    races: list[dict[str, Any]] = []

    for table in soup.find_all("table", class_="igb-tbl"):
        header_th = table.find("th")
        if not header_th:
            continue
        header_text = header_th.get_text(" ", strip=True)
        if not re.search(r"Race\s+\d+", header_text, re.IGNORECASE):
            continue

        race_info = _parse_card_header(header_text)
        if race_info["race_number"] is None:
            continue

        entries: list[dict[str, Any]] = []
        for row in table.find_all("tr"):
            trap_img = row.find("img", alt=re.compile(r"^Trap\s*\d+$", re.IGNORECASE))
            if not trap_img:
                continue
            trap_match = re.search(r"Trap\s*(\d+)", trap_img.get("alt", ""))
            if not trap_match:
                continue
            trap = int(trap_match.group(1))

            dog_link = row.find(
                "a", href=re.compile(r"greyhound-search/greyhound-details", re.IGNORECASE)
            )
            if not dog_link:
                continue
            dog_name = dog_link.get_text(strip=True).upper()
            if not dog_name:
                continue

            entry: dict[str, Any] = {"trap": trap, "dog_name": dog_name}
            gri_id = _extract_gri_id(dog_link.get("href"))
            if gri_id:
                entry["gri_id"] = gri_id
            entries.append(entry)

        if not entries:
            continue

        races.append({
            "race_number": race_info["race_number"],
            "race_date": race_date,
            "race_time": race_info["race_time"],
            "track_code": track_code,
            "distance_m": race_info["distance_m"],
            "grade": race_info["grade"],
            "race_type": race_info["race_type"],
            "going": None,
            "prize_money": None,
            "status": "scheduled",
            "entries": entries,
        })

    if not races:
        if _has_race_like_text(html):
            raise ParseStructureError(
                f"Card page for {track_code} {race_date} contains race-like "
                "text but 0 races could be parsed — markup has changed"
            )
        if not _has_gri_page_anchor(soup):
            raise ParseStructureError(
                f"Page for {track_code} {race_date} lacks GRI structural "
                "anchors (track dropdown / igb-tbl) — not a GRI card page"
            )
        return []

    logger.info("Parsed %d card races from %s on %s", len(races), track_code, race_date)
    return races


async def scrape_card(
    track_code: str,
    race_date: date,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Scrape the card summary (Tier 1) for a track + future date.

    Raises ScrapeFetchError on network failure or non-200 response (with
    retry/backoff for transient failures — see `_fetch_page`).
    """
    date_str = format_date(race_date)
    url = f"{VIEW_CARD_URL}?track={track_code}&date={date_str}"
    html = await _fetch_page(url, client)
    return parse_card_page(html, track_code, race_date)


def parse_card_form_page(html: str) -> dict[int, dict[str, Any]]:
    """Parse a per-race form-detail page.

    Returns `{trap_number: {trainer_name, owner, sire_name, dam_name, best_time}}`.
    Used to enrich Dog records — the page does NOT carry an intended weight for
    the upcoming race (dogs are weighed-in on the night).
    """
    soup = BeautifulSoup(html, "html.parser")
    by_trap: dict[int, dict[str, Any]] = {}

    # Each dog block is anchored by an <img alt="Trap N"> with rowspan=8 in the
    # outer form table. The relevant <a> tags appear as siblings/descendants of
    # the next ~3 <tr> elements.
    trap_imgs = soup.find_all("img", alt=re.compile(r"^Trap\s*\d+$", re.IGNORECASE))
    if not trap_imgs:
        return {}

    # Find the bounding tr for each trap, plus the following 3 tr siblings
    # which carry owner/trainer/dog/breeding info.
    for trap_img in trap_imgs:
        trap_match = re.search(r"Trap\s*(\d+)", trap_img.get("alt", ""))
        if not trap_match:
            continue
        trap = int(trap_match.group(1))
        if trap in by_trap:
            continue  # already captured (some pages render the trap twice)

        anchor_tr = trap_img.find_parent("tr")
        if not anchor_tr:
            continue

        # Walk forward over the next 3 sibling rows (header, dog, breeding/late notice)
        block_rows = [anchor_tr]
        sib = anchor_tr
        for _ in range(3):
            sib = sib.find_next_sibling("tr")
            if not sib:
                break
            block_rows.append(sib)

        info: dict[str, Any] = {
            "trainer_name": None,
            "owner_name": None,
            "sire_name": None,
            "dam_name": None,
            "best_time": None,
            "gri_id": None,
        }

        for row in block_rows:
            # Owner: <a href=/results/greyhound-search/owners-page/?oid=...>Name</a>
            if info["owner_name"] is None:
                owner_a = row.find("a", href=re.compile(r"owners-page", re.IGNORECASE))
                if owner_a:
                    info["owner_name"] = owner_a.get_text(strip=True) or None

            # Trainer: <a href=/results/trainers-page/?tid=...>Name</a>  (often empty)
            if info["trainer_name"] is None:
                trainer_a = row.find("a", href=re.compile(r"trainers-page", re.IGNORECASE))
                if trainer_a:
                    name = trainer_a.get_text(strip=True)
                    if name:
                        info["trainer_name"] = name

            row_text = row.get_text(" ", strip=True)
            is_breeding_row = bool(
                "/" in row_text and re.search(r"\.\w{3}-\d{2}", row_text)
            )

            # The running dog's own detail link (carries its GRI id) appears
            # in a NON-breeding row — breeding rows hold sire/dam links.
            if info["gri_id"] is None and not is_breeding_row:
                own_link = row.find(
                    "a",
                    href=re.compile(
                        r"greyhound-search/greyhound-details", re.IGNORECASE
                    ),
                )
                if own_link:
                    info["gri_id"] = _extract_gri_id(own_link.get("href"))

            # Sire/Dam: two greyhound-details links inside breeding cell
            if info["sire_name"] is None or info["dam_name"] is None:
                if is_breeding_row:
                    dog_links = row.find_all(
                        "a",
                        href=re.compile(
                            r"greyhound-search/greyhound-details", re.IGNORECASE
                        ),
                    )
                    # First link in the breeding row is the running dog name itself
                    # when it appears in the SAME row; sire/dam are the trailing two.
                    # Uppercased for consistency with the results parser and how
                    # dog names are stored.
                    if len(dog_links) >= 2:
                        info["sire_name"] = dog_links[-2].get_text(strip=True).upper()
                        info["dam_name"] = dog_links[-1].get_text(strip=True).upper()

            # Best time: bracketed [29.01-ECY] notation
            if info["best_time"] is None:
                bt_match = re.search(r"\[(\d{2}\.\d{2})-([A-Z]{2,4})\]", row.get_text())
                if bt_match:
                    info["best_time"] = float(bt_match.group(1))

        by_trap[trap] = info

    return by_trap


async def scrape_card_form(
    track_code: str,
    race_date: date,
    race_number: int,
    client: httpx.AsyncClient | None = None,
) -> dict[int, dict[str, Any]]:
    """Scrape per-race form detail (Tier 2) and return enrichment by trap.

    Raises ScrapeFetchError on network failure or non-200 response (with
    retry/backoff for transient failures — see `_fetch_page`).
    """
    date_str = format_date(race_date)
    url = (
        f"{VIEW_CARD_FORM_URL}?Track={track_code}&Date={date_str}"
        f"&RaceNumber={race_number}"
    )
    html = await _fetch_page(url, client)
    return parse_card_form_page(html)


def merge_card_form_into_race(
    race: dict[str, Any], form_by_trap: dict[int, dict[str, Any]]
) -> None:
    """Mutate `race` in place, copying trainer/sire/dam/owner onto each entry."""
    for entry in race.get("entries", []):
        trap = entry.get("trap")
        if trap is None or trap not in form_by_trap:
            continue
        form = form_by_trap[trap]
        for key in (
            "trainer_name", "owner_name", "sire_name", "dam_name",
            "best_time", "gri_id",
        ):
            if form.get(key) and not entry.get(key):
                entry[key] = form[key]

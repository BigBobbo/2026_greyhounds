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

GRI_BASE_URL = "https://www.grireland.ie"
VIEW_RESULTS_URL = f"{GRI_BASE_URL}/results/view-results/"
VIEW_CARD_URL = f"{GRI_BASE_URL}/racing/upcoming-race-cards/upcoming-race-card-summary/"
VIEW_CARD_FORM_URL = f"{GRI_BASE_URL}/racing/upcoming-race-cards/upcoming-race-card-summary/view-race-form/"

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

    # Find all race header elements. Match on the element's full text, not
    # bs4's `string=` — `string=` only matches when the tag has a single
    # string child, so any nested markup GRI adds inside the <h4> (a span,
    # a link) would silently drop every race site-wide.
    race_headers = [
        h for h in soup.find_all("h4")
        if re.search(r"Race\s+\d+", h.get_text(" ", strip=True), re.IGNORECASE)
    ]

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
        margin = parse_beaten_margin(by_text)
        if margin is not None:
            entry["beaten_distance"] = margin

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


# Named margins in lengths — the racing conventions for close finishes.
# Close finishes carry the MOST information about relative ability, and the
# old numeric-only parser threw every one of them away (SH/HD/NK -> NULL)
# and truncated vulgar fractions ("1½" -> 1.0).
_NAMED_MARGINS = {
    "DH": 0.0,     # dead heat
    "NSE": 0.02,   # nose (rare on GRI but appears)
    "SH": 0.05,    # short head
    "SHD": 0.05,
    "HD": 0.1,     # head
    "NK": 0.25,    # neck
    "DIS": None,   # distance — unquantified blowout; leave NULL
    "DIST": None,
}
_VULGAR = {"¼": 0.25, "½": 0.5, "¾": 0.75}


def parse_beaten_margin(text: str) -> float | None:
    """Parse a beaten-distance cell into lengths.

    Handles: "3", "3L", "2.5", named margins ("SH", "HD", "NK", "DH"),
    vulgar fractions ("½", "1½", "2¾"), and ASCII fractions ("1 1/2").
    Returns None when the margin is genuinely unquantifiable ("DIS").
    """
    t = text.strip().upper().rstrip("L").strip()
    if not t:
        return None

    if t in _NAMED_MARGINS:
        return _NAMED_MARGINS[t]

    # Vulgar fraction, optionally after a whole number: "½", "1½", "2¾"
    m = re.match(r"^(\d+)?\s*([¼½¾])$", t)
    if m:
        whole = int(m.group(1)) if m.group(1) else 0
        return whole + _VULGAR[m.group(2)]

    # ASCII fraction: "1 1/2", "1/2"
    m = re.match(r"^(?:(\d+)\s+)?(\d+)/(\d+)$", t)
    if m:
        whole = int(m.group(1)) if m.group(1) else 0
        num, den = int(m.group(2)), int(m.group(3))
        if den > 0:
            return whole + num / den

    # Plain decimal / integer
    m = re.match(r"^(\d+(?:\.\d+)?)$", t)
    if m:
        return float(m.group(1))

    return None


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


class ScrapeError(Exception):
    """A fetch failed after retries. Distinct from an empty page: an empty
    list from scrape_results means GRI genuinely lists no races for that
    track/date; a ScrapeError means we DON'T KNOW what GRI lists. Callers
    that treated both as "quiet day" were silently losing whole meetings."""


_RETRY_DELAYS = (2.0, 4.0, 8.0)


async def scrape_results(
    track_code: str,
    race_date: date,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Scrape race results for a specific track and date.

    Retries transient failures with backoff; raises ScrapeError when the
    page cannot be fetched at all, so callers can distinguish failure from
    a day with no racing.
    """
    date_str = format_date(race_date)
    url = f"{VIEW_RESULTS_URL}?track={track_code}&date={date_str}"

    close_client = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30)
        close_client = True

    try:
        last_err: str | None = None
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return parse_results_page(resp.text, track_code, race_date)
                last_err = f"HTTP {resp.status_code}"
                logger.warning(
                    "Got %d for %s (attempt %d)", resp.status_code, url, attempt + 1,
                )
            except Exception as e:
                last_err = str(e)
                logger.warning(
                    "Fetch failed for %s (attempt %d): %s", url, attempt + 1, e,
                )
            if delay is None:
                break
            await asyncio.sleep(delay)

        raise ScrapeError(f"{url}: {last_err}")
    finally:
        if close_client:
            await client.aclose()


async def scrape_date_range(
    track_code: str,
    start_date: date,
    end_date: date,
    delay: float = 1.0,
) -> list[dict[str, Any]]:
    """Scrape results for a track across a date range. Fast — no browser needed."""
    all_races: list[dict[str, Any]] = []

    failed_days: list[date] = []
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
        current = start_date
        total_days = (end_date - start_date).days + 1
        day_num = 0

        while current <= end_date:
            day_num += 1
            try:
                races = await scrape_results(track_code, current, client)
                all_races.extend(races)
            except ScrapeError as e:
                failed_days.append(current)
                logger.error("Day failed for %s: %s", track_code, e)

            if day_num % 50 == 0:
                logger.info(
                    "Progress: %s %d/%d days, %d races found",
                    track_code, day_num, total_days, len(all_races),
                )

            current += timedelta(days=1)
            if current <= end_date:
                await asyncio.sleep(delay)

    if failed_days:
        logger.error(
            "Completed %s with %d FAILED days (data missing, retry these): %s",
            track_code, len(failed_days),
            ", ".join(str(d) for d in failed_days[:20]),
        )
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
    dist_match = re.search(r"\(\s*(\d{3,4})\s*(?:Yds|Yards|m)\b", text, re.IGNORECASE)
    if dist_match:
        val = int(dist_match.group(1))
        if 200 <= val <= 1000:
            info["distance_m"] = val
    if info["distance_m"] is None:
        # Fallback: first 3-4 digit number
        for d in re.findall(r"\b(\d{3,4})\b", text):
            v = int(d)
            if 200 <= v <= 1000:
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

            entries.append({"trap": trap, "dog_name": dog_name})

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

    logger.info("Parsed %d card races from %s on %s", len(races), track_code, race_date)
    return races


async def scrape_card(
    track_code: str,
    race_date: date,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Scrape the card summary (Tier 1) for a track + future date."""
    date_str = format_date(race_date)
    url = f"{VIEW_CARD_URL}?track={track_code}&date={date_str}"

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

            # Sire/Dam: two greyhound-details links inside breeding cell
            if info["sire_name"] is None or info["dam_name"] is None:
                breeding_text = row.get_text(" ", strip=True)
                if "/" in breeding_text and re.search(r"\.\w{3}-\d{2}", breeding_text):
                    dog_links = row.find_all(
                        "a",
                        href=re.compile(
                            r"greyhound-search/greyhound-details", re.IGNORECASE
                        ),
                    )
                    # First link in the breeding row is the running dog name itself
                    # when it appears in the SAME row; sire/dam are the trailing two.
                    if len(dog_links) >= 2:
                        info["sire_name"] = dog_links[-2].get_text(strip=True).title()
                        info["dam_name"] = dog_links[-1].get_text(strip=True).title()

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
    """Scrape per-race form detail (Tier 2) and return enrichment by trap."""
    date_str = format_date(race_date)
    url = (
        f"{VIEW_CARD_FORM_URL}?Track={track_code}&Date={date_str}"
        f"&RaceNumber={race_number}"
    )

    close_client = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30)
        close_client = True

    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            logger.warning("Got %d for %s", resp.status_code, url)
            return {}
        html = resp.text
    except Exception as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return {}
    finally:
        if close_client:
            await client.aclose()

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
        for key in ("trainer_name", "owner_name", "sire_name", "dam_name", "best_time"):
            if form.get(key) and not entry.get(key):
                entry[key] = form[key]

"""
GRI (Greyhound Racing Ireland) UPCOMING race-card scraper.

Companion to gri_scraper.py (which only handles historical RESULTS).
This module fetches a published race card (declarations) for a future date
so we can predict upcoming meetings.

Key differences vs the results scraper:
- Race cards have NO finishing positions, win times, or SP. The parser
  must work without those columns.
- The HTML layout for the declarations page may differ from results.
  We try multiple candidate URL patterns and a flexible table parser
  that only insists on (trap, dog name).
- Races scraped here are saved with status="scheduled".

URL candidates (tried in order):
  1. /racing/race-card/?track={CODE}&date={DD-Mon-YYYY}
  2. /racing/cards/?track={CODE}&date={DD-Mon-YYYY}
  3. /racing/declarations/?track={CODE}&date={DD-Mon-YYYY}
  4. /results/view-results/?track={CODE}&date={DD-Mon-YYYY}  (some sites
     republish the same page for future dates with declarations only)

If GRI changes its URL, set GRI_RACECARD_URL env var to the correct path
(without query string) — e.g. "https://www.grireland.ie/racing/race-card/".
"""

import logging
import os
import re
from datetime import date
from typing import Any

import httpx
from bs4 import BeautifulSoup

from scraping.gri_scraper import (
    DEFAULT_HEADERS,
    GRI_BASE_URL,
    GRI_TRACK_CODES,
    VIEW_RESULTS_URL,
    _parse_race_header,
    format_date,
)

logger = logging.getLogger(__name__)


def _candidate_urls(track_code: str, race_date: date) -> list[str]:
    date_str = format_date(race_date)

    override = os.environ.get("GRI_RACECARD_URL")
    base_paths: list[str] = []
    if override:
        base_paths.append(override.rstrip("/") + "/")
    base_paths.extend([
        f"{GRI_BASE_URL}/racing/race-card/",
        f"{GRI_BASE_URL}/racing/cards/",
        f"{GRI_BASE_URL}/racing/declarations/",
        VIEW_RESULTS_URL,
    ])

    seen: set[str] = set()
    urls: list[str] = []
    for base in base_paths:
        url = f"{base}?track={track_code}&date={date_str}"
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def parse_racecard_page(
    html: str, track_code: str, race_date: date
) -> list[dict[str, Any]]:
    """
    Parse a race-card / declarations page.

    Looks for <h4> "Race N" headers (same as results page) and the next
    table after each. For each table row we try to extract:
      - trap number  (from <img alt="Trap N"> or first numeric cell)
      - dog name     (from <a> text, falling back to plain cell text)
      - trainer name (best-effort: look for "Trainer:" prefix or known col)
      - weight_kg    (any cell matching r"\\d+\\.\\d+\\s*kg" or 28-40 range)
      - sire / dam   (from pedigree spans if present, like results parser)

    A row is kept only if it has a trap AND a dog name.
    """
    soup = BeautifulSoup(html, "html.parser")
    races: list[dict[str, Any]] = []

    race_headers = soup.find_all("h4", string=re.compile(r"Race\s+\d+", re.IGNORECASE))
    if not race_headers:
        logger.debug("No race headers found in card page for %s %s", track_code, race_date)
        return []

    for header in race_headers:
        header_text = header.get_text(strip=True)
        race_info = _parse_race_header(header_text)

        table = header.find_next("table", class_="igb-tbl") or header.find_next("table")
        if not table:
            continue

        # Race time often appears in a sibling element; best-effort extraction
        race_time = _extract_race_time(header)

        entries = _parse_card_table(table)
        if not entries:
            continue

        races.append({
            "race_number": race_info["race_number"],
            "race_date": race_date,
            "race_time": race_time,
            "track_code": track_code,
            "distance_m": race_info["distance_m"],
            "grade": race_info["grade"],
            "race_type": race_info["race_type"],
            "going": None,
            "prize_money": None,
            "entries": entries,
        })

    logger.info(
        "Parsed %d upcoming races from %s on %s", len(races), track_code, race_date
    )
    return races


_TIME_RE = re.compile(r"\b(\d{1,2}[:.]\d{2})\b")


def _extract_race_time(header) -> str | None:
    """Try to find a HH:MM time near the race header."""
    text = header.get_text(" ", strip=True)
    m = _TIME_RE.search(text)
    if m:
        return m.group(1).replace(".", ":")
    # Sometimes time is in an adjacent element
    sibling = header.find_next(["p", "span", "div"])
    if sibling:
        m = _TIME_RE.search(sibling.get_text(" ", strip=True))
        if m:
            return m.group(1).replace(".", ":")
    return None


_WEIGHT_RE = re.compile(r"(\d{2}\.\d{1,2})\s*(kg)?", re.IGNORECASE)


def _parse_card_table(table) -> list[dict[str, Any]]:
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    entries: list[dict[str, Any]] = []
    for row in rows[1:]:  # skip header
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        entry: dict[str, Any] = {}

        # Trap: prefer <img alt="Trap N"> anywhere in the row
        trap = None
        for c in cells:
            img = c.find("img")
            if img:
                alt = img.get("alt", "")
                m = re.search(r"Trap\s*(\d+)", alt, re.IGNORECASE)
                if m:
                    trap = int(m.group(1))
                    break
        # Fallback: first cell's plain text "1." / "1"
        if trap is None:
            text0 = cells[0].get_text(strip=True)
            m = re.match(r"^(\d+)[.\s]*$", text0)
            if m:
                v = int(m.group(1))
                if 1 <= v <= 8:
                    trap = v
        if trap is None:
            continue
        entry["trap"] = trap

        # Dog name: first <a> in the row whose text isn't "Trap N"
        dog_link = None
        for c in cells:
            for a in c.find_all("a"):
                text = a.get_text(strip=True)
                if text and not re.match(r"Trap\s*\d+", text, re.IGNORECASE):
                    dog_link = a
                    break
            if dog_link:
                break
        if dog_link:
            entry["dog_name"] = dog_link.get_text(strip=True).upper()
        else:
            # Fallback: pick the longest non-numeric cell text
            candidates = [
                c.get_text(strip=True) for c in cells if c.get_text(strip=True)
            ]
            candidates = [t for t in candidates if not re.match(r"^[\d.]+$", t)]
            if candidates:
                entry["dog_name"] = max(candidates, key=len).upper()

        if not entry.get("dog_name"):
            continue

        # Sire / dam from pedigree spans (results-style markup)
        sire_span = row.find(class_="viewresults-pedigree-sire")
        if sire_span:
            sa = sire_span.find("a")
            if sa:
                entry["sire_name"] = sa.get_text(strip=True)
        dam_span = row.find(class_="viewresults-pedigree-dam")
        if dam_span:
            da = dam_span.find("a")
            if da:
                entry["dam_name"] = da.get_text(strip=True)

        # Weight: scan cells for a "NN.N" pattern in the kg range
        for c in cells:
            t = c.get_text(" ", strip=True)
            for m in _WEIGHT_RE.finditer(t):
                val = float(m.group(1))
                if 24.0 <= val <= 42.0:
                    entry["weight_kg"] = val
                    break
            if "weight_kg" in entry:
                break

        # Trainer: look for a cell with "Trainer:" or a column header hint.
        # Best-effort only — many declarations pages put trainer in a
        # separate cell adjacent to the dog name.
        for c in cells:
            t = c.get_text(" ", strip=True)
            m = re.search(r"Trainer\s*[:\-]\s*([A-Z][A-Za-z .'\-]+)", t)
            if m:
                entry["trainer_name"] = m.group(1).strip()
                break

        entries.append(entry)

    return entries


async def scrape_race_card(
    track_code: str,
    race_date: date,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Try candidate URLs in order until one returns a parseable race card.

    Returns (races, url_used). url_used is None if nothing worked.
    """
    close_client = False
    if client is None:
        client = httpx.AsyncClient(
            headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30
        )
        close_client = True

    try:
        for url in _candidate_urls(track_code, race_date):
            try:
                resp = await client.get(url)
            except Exception as e:
                logger.warning("Race-card fetch failed for %s: %s", url, e)
                continue

            if resp.status_code != 200:
                logger.debug("Race-card %s -> HTTP %d", url, resp.status_code)
                continue

            races = parse_racecard_page(resp.text, track_code, race_date)
            if races:
                logger.info("Race-card hit: %s (%d races)", url, len(races))
                return races, url

        logger.info(
            "No race card found for %s on %s (tried %d URL patterns)",
            track_code, race_date, len(_candidate_urls(track_code, race_date)),
        )
        return [], None
    finally:
        if close_client:
            await client.aclose()


def known_track_codes() -> list[dict[str, str]]:
    return [{"code": code, "name": name} for code, name in GRI_TRACK_CODES.items()]

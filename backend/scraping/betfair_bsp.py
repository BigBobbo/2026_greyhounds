"""
Betfair BSP (Betfair Starting Price) historical data importer.

Downloads and parses CSV files from Betfair's free BSP data archive:
  https://promo.betfair.com/betfairsp/prices

File naming: dwbfgreyhoundwin{DDMMYYYY}.csv
URL pattern: https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin{DDMMYYYY}.csv

CSV columns (greyhound win markets):
  EVENT_ID, MENU_HINT, EVENT_NAME, EVENT_DT, SELECTION_ID, SELECTION_NAME,
  WIN_LOSE, BSP, PPWAP, MOWAP, PPMAX, PPMIN, IPMAX, IPMIN, MORNINGWAP,
  PPTRADEDVOL, IPTRADEDVOL

MENU_HINT contains the track name (e.g. "Greyhounds/Shelbourne/20:28")
EVENT_NAME contains the meeting info (e.g. "Shel 20:28 525m A3")
SELECTION_NAME is the dog name + trap (e.g. "1. Ballymac Flash")

BSP = Betfair Starting Price (decimal odds)
"""

import asyncio
import csv
import io
import logging
import re
from datetime import date, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BSP_BASE_URL = "https://promo.betfair.com/betfairsp/prices"

# Mapping of Betfair track names to GRI track codes
# Betfair uses abbreviated names in MENU_HINT and EVENT_NAME
BETFAIR_TO_GRI_TRACK = {
    "shelbourne": "SPK",
    "shel": "SPK",
    "shelbourne park": "SPK",
    "curraheen": "CRK",
    "curraheen park": "CRK",
    "tralee": "TRL",
    "limerick": "LMK",
    "clonmel": "CML",
    "galway": "GLY",
    "dundalk": "DLK",
    "kilkenny": "KKY",
    "newbridge": "NWB",
    "mullingar": "MGR",
    "waterford": "WFD",
    "thurles": "THR",
    "thurles park": "THR",
    "enniscorthy": "ECY",
    "youghal": "YGL",
    "longford": "LGD",
    "lifford": "LFD",
    "derry": "DRY",
    "drumbo": "DBP",
    "drumbo park": "DBP",
    "harolds cross": "HRX",
    "harold's cross": "HRX",
}

# Irish track names for filtering (Betfair data includes UK races too)
IRISH_TRACK_NAMES = set(BETFAIR_TO_GRI_TRACK.keys())


def _format_bsp_date(d: date) -> str:
    """Format date for BSP filename: DDMMYYYY."""
    return d.strftime("%d%m%Y")


def bsp_csv_url(d: date) -> str:
    """Build the download URL for a given date's BSP CSV."""
    return f"{BSP_BASE_URL}/dwbfgreyhoundwin{_format_bsp_date(d)}.csv"


async def download_bsp_csv(
    target_date: date,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Download BSP CSV for a given date. Returns CSV text or None."""
    url = bsp_csv_url(target_date)
    close_client = False
    if client is None:
        client = httpx.AsyncClient(follow_redirects=True, timeout=30)
        close_client = True

    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            logger.debug("No BSP data for %s (404)", target_date)
            return None
        if resp.status_code != 200:
            logger.warning("BSP download failed for %s: HTTP %d", target_date, resp.status_code)
            return None
        return resp.text
    except Exception as e:
        logger.error("BSP download error for %s: %s", target_date, e)
        return None
    finally:
        if close_client:
            await client.aclose()


def _extract_track_from_menu_hint(menu_hint: str) -> str | None:
    """
    Extract track name from MENU_HINT like 'Greyhounds/Shelbourne/20:28'.
    Returns lowercase track name or None.
    """
    if not menu_hint:
        return None
    parts = menu_hint.split("/")
    if len(parts) >= 2:
        return parts[1].strip().lower()
    return None


def _extract_trap_and_name(selection_name: str) -> tuple[int | None, str]:
    """
    Parse SELECTION_NAME like '1. Ballymac Flash' into (trap_number, dog_name).
    """
    match = re.match(r"(\d+)\.\s*(.+)", selection_name.strip())
    if match:
        return int(match.group(1)), match.group(2).strip().upper()
    return None, selection_name.strip().upper()


def _extract_distance_from_event(event_name: str) -> int | None:
    """Extract distance in meters from EVENT_NAME like 'Shel 20:28 525m A3'."""
    match = re.search(r"(\d{3,4})m", event_name)
    if match:
        return int(match.group(1))
    return None


def _extract_race_time(event_name: str) -> str | None:
    """Extract race time from EVENT_NAME like 'Shel 20:28 525m A3'."""
    match = re.search(r"(\d{1,2}:\d{2})", event_name)
    if match:
        return match.group(1)
    return None


def parse_bsp_csv(csv_text: str, irish_only: bool = True) -> list[dict[str, Any]]:
    """
    Parse Betfair BSP CSV into a list of odds records.

    Each record contains:
      - event_id, event_date, track_name, gri_track_code (if Irish)
      - dog_name, trap, bsp_decimal, implied_prob
      - win_lose (W/L), event_name, race_time, distance_m
      - ppwap, ipmax, ipmin (pre-play and in-play price data)

    If irish_only=True, filters to only Irish tracks.
    """
    records = []
    reader = csv.DictReader(io.StringIO(csv_text))

    for row in reader:
        menu_hint = row.get("MENU_HINT", "")
        track_name = _extract_track_from_menu_hint(menu_hint)

        if not track_name:
            continue

        gri_code = BETFAIR_TO_GRI_TRACK.get(track_name)

        if irish_only and not gri_code:
            continue

        event_name = row.get("EVENT_NAME", "")
        selection_name = row.get("SELECTION_NAME", "")
        trap, dog_name = _extract_trap_and_name(selection_name)

        bsp_str = row.get("BSP", "").strip()
        try:
            bsp = float(bsp_str) if bsp_str else None
        except ValueError:
            bsp = None

        if bsp is None or bsp <= 0:
            continue

        implied_prob = round(1.0 / bsp, 4) if bsp > 0 else None

        # Parse additional price columns
        def _safe_float(key: str) -> float | None:
            val = row.get(key, "").strip()
            try:
                return float(val) if val else None
            except ValueError:
                return None

        record = {
            "event_id": row.get("EVENT_ID", "").strip(),
            "event_date": row.get("EVENT_DT", "").strip(),
            "track_name": track_name,
            "gri_track_code": gri_code,
            "event_name": event_name.strip(),
            "race_time": _extract_race_time(event_name),
            "distance_m": _extract_distance_from_event(event_name),
            "selection_id": row.get("SELECTION_ID", "").strip(),
            "dog_name": dog_name,
            "trap": trap,
            "bsp_decimal": bsp,
            "implied_prob": implied_prob,
            "win_lose": row.get("WIN_LOSE", "").strip(),
            "ppwap": _safe_float("PPWAP"),
            "ipmax": _safe_float("IPMAX"),
            "ipmin": _safe_float("IPMIN"),
            "pp_traded_vol": _safe_float("PPTRADEDVOL"),
            "ip_traded_vol": _safe_float("IPTRADEDVOL"),
        }
        records.append(record)

    logger.info("Parsed %d BSP records (%s Irish filtering)", len(records), "with" if irish_only else "without")
    return records


async def fetch_bsp_date_range(
    start_date: date,
    end_date: date,
    irish_only: bool = True,
    delay: float = 1.0,
) -> list[dict[str, Any]]:
    """Download and parse BSP data for a date range."""
    all_records = []

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        current = start_date
        total_days = (end_date - start_date).days + 1
        day_num = 0

        while current <= end_date:
            day_num += 1
            csv_text = await download_bsp_csv(current, client)

            if csv_text:
                records = parse_bsp_csv(csv_text, irish_only=irish_only)
                all_records.extend(records)

            if day_num % 30 == 0:
                logger.info(
                    "BSP progress: %d/%d days, %d records so far",
                    day_num, total_days, len(all_records),
                )

            current += timedelta(days=1)
            if current <= end_date:
                await asyncio.sleep(delay)

    logger.info("BSP fetch complete: %d days, %d total records", total_days, len(all_records))
    return all_records

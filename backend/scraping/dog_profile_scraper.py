"""GRI dog-profile scraper — fills the dead columns.

Each greyhound has a profile page at
``/results/greyhound-search/greyhound-details/?gid=<NAME>`` carrying:

  * header: whelp (birth) date, sex, colour, trainer, owner, sire/dam,
    current seeding;
  * a form-lines table covering the dog's FULL career (verified: a
    163-run dog shows 167 lines, no pagination), each line with the
    sectional time, sectional running positions (e.g. "1222"), weight,
    beaten margin, going allowance, remark and SP.

Early pace (sectional time) is the single most predictive variable in
greyhound racing; the schema always had ``sectional_time`` /
``adjusted_time`` / ``birth_date`` / ``sex`` columns but nothing wrote
them. This module is the writer.

Going-allowance convention: the printed value ("+.20", "-.30") is stored
as-is on the race and applied as ``adjusted_time = finish_time +
allowance``. Whichever direction GRI intends, applying the printed value
uniformly yields internally consistent normalised times, which is what the
speed-figure layer needs.
"""

import asyncio
import logging
import re
from datetime import date, datetime
from typing import Any
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from scraping.gri_scraper import DEFAULT_HEADERS, GRI_BASE_URL, ScrapeError, _RETRY_DELAYS

logger = logging.getLogger(__name__)

DOG_DETAILS_URL = f"{GRI_BASE_URL}/results/greyhound-search/greyhound-details/"


def _parse_whelp_date(text: str) -> date | None:
    """'01-Jun-24' -> date(2024, 6, 1). Two-digit years pivot at 50."""
    m = re.search(r"(\d{1,2})-([A-Za-z]{3})-(\d{2,4})", text)
    if not m:
        return None
    try:
        day, mon, yr = int(m.group(1)), m.group(2), int(m.group(3))
        if yr < 100:
            yr += 2000 if yr < 50 else 1900
        return datetime.strptime(f"{day}-{mon}-{yr}", "%d-%b-%Y").date()
    except ValueError:
        return None


def parse_profile_header(soup: BeautifulSoup) -> dict[str, Any]:
    """Parse the key/value header table (Whelp Date, Trainer, ...)."""
    out: dict[str, Any] = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        first = rows[0].get_text(" ", strip=True)
        if "Whelp Date" not in first:
            continue
        for row in rows:
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if len(cells) < 2:
                continue
            key = cells[0].rstrip(":").strip().lower()
            val = cells[1].strip()
            if not val:
                continue
            if key == "whelp date":
                out["birth_date"] = _parse_whelp_date(val)
            elif key == "owner(s)":
                out["owner_name"] = val
            elif key == "trainer":
                out["trainer_name"] = val
            elif key == "sire / dam":
                parts = [p.strip() for p in val.split("/")]
                if len(parts) == 2:
                    out["sire"], out["dam"] = parts[0], parts[1]
            elif key == "color / sex":
                parts = [p.strip() for p in val.split("/")]
                if len(parts) == 2:
                    out["colour"], out["sex"] = parts[0], parts[1]
            elif key == "last race seeding":
                out["seeding"] = val
        break
    return out


def _parse_going_allowance(text: str) -> float | None:
    """'+.20 Fast' -> 0.20; '-.30 Slow' -> -0.30; 'N/A' -> None."""
    m = re.search(r"([+-])\s*(\d*\.?\d+)", text)
    if not m:
        return None
    val = float(m.group(2))
    return val if m.group(1) == "+" else -val


def parse_form_lines(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Parse the career form table (header: Date | Wt. | Dist. | Trap |
    Sct. T. | Sct. P. | Place | By | Winner/Second | Dogs | Win Tm |
    Going | Rem | SP)."""
    lines: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        hdr = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        if "Sct. T." not in hdr:
            continue
        col = {name: i for i, name in enumerate(hdr)}

        def cell(cells, name) -> str:
            i = col.get(name)
            if i is None or i >= len(cells):
                return ""
            return cells[i].get_text(" ", strip=True)

        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            run_date = _parse_whelp_date(cell(cells, "Date"))
            if run_date is None:
                continue
            line: dict[str, Any] = {"race_date": run_date}

            sct = cell(cells, "Sct. T.")
            m = re.match(r"(\d+\.?\d*)", sct)
            if m:
                line["sectional_time"] = float(m.group(1))

            pos = cell(cells, "Sct. P.")
            if re.match(r"^\d{2,6}$", pos):
                line["running_positions"] = pos

            wt = cell(cells, "Wt.")
            m = re.match(r"(\d+\.?\d*)", wt)
            if m:
                line["weight_kg"] = float(m.group(1))

            trap = cell(cells, "Trap")
            m = re.search(r"(\d)", trap)
            if m:
                line["trap"] = int(m.group(1))

            place = cell(cells, "Place")
            m = re.match(r"(\d+)", place)
            if m:
                line["finish_position"] = int(m.group(1))

            line["going_allowance"] = _parse_going_allowance(cell(cells, "Going"))

            sp = cell(cells, "SP")
            if sp:
                line["starting_price"] = sp

            lines.append(line)
        break
    return lines


async def scrape_dog_profile(
    name: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch and parse one dog's profile. Raises ScrapeError on failure."""
    url = f"{DOG_DETAILS_URL}?gid={quote(name)}"
    close = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30)
        close = True
    try:
        last_err = None
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    return {
                        "header": parse_profile_header(soup),
                        "form_lines": parse_form_lines(soup),
                    }
                last_err = f"HTTP {resp.status_code}"
            except Exception as e:
                last_err = str(e)
            if delay is None:
                break
            await asyncio.sleep(delay)
        raise ScrapeError(f"{url}: {last_err}")
    finally:
        if close:
            await client.aclose()


def apply_profile(db: Session, dog: Dog, profile: dict[str, Any]) -> dict[str, int]:
    """Write a scraped profile onto the dog and its race entries.

    Form lines are matched to entries by (dog_id, race_date) — a dog runs
    at most one race per day. Returns counts of what was written.
    """
    from scraping.db_pipeline import normalize_name

    stats = {"entries_updated": 0, "races_updated": 0}
    header = profile.get("header", {})

    if header.get("birth_date") and not dog.birth_date:
        dog.birth_date = header["birth_date"]
    if header.get("sex") and not dog.sex:
        dog.sex = header["sex"]
    if header.get("colour") and not dog.colour:
        dog.colour = header["colour"]
    if header.get("owner_name") and not dog.owner_name:
        dog.owner_name = header["owner_name"]
    if header.get("trainer_name"):
        # Profile trainer beats nothing; never overwrite a real name with
        # the placeholder "Owner" if we somehow have better data.
        t = normalize_name(header["trainer_name"])
        if not dog.trainer_name:
            dog.trainer_name = t
    if header.get("sire") and not dog.sire:
        dog.sire = header["sire"]
    if header.get("dam") and not dog.dam:
        dog.dam = header["dam"]

    if not profile.get("form_lines"):
        return stats

    entries = (
        db.query(RaceEntry, Race)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.dog_id == dog.id)
        .all()
    )
    by_date: dict[date, tuple[RaceEntry, Race]] = {}
    for entry, race in entries:
        rd = race.race_date
        if isinstance(rd, str):
            rd = date.fromisoformat(rd)
        by_date[rd] = (entry, race)

    for line in profile["form_lines"]:
        hit = by_date.get(line["race_date"])
        if not hit:
            continue
        entry, race = hit
        changed = False
        if line.get("sectional_time") is not None and entry.sectional_time is None:
            entry.sectional_time = line["sectional_time"]
            changed = True
        if line.get("running_positions") and getattr(entry, "running_positions", None) is None:
            entry.running_positions = line["running_positions"]
            changed = True
        if line.get("weight_kg") and not entry.weight_kg:
            entry.weight_kg = line["weight_kg"]
            changed = True
        if changed:
            stats["entries_updated"] += 1

        allowance = line.get("going_allowance")
        if allowance is not None and race.going_allowance is None:
            race.going_allowance = allowance
            stats["races_updated"] += 1
        if race.going_allowance is not None and entry.finish_time is not None \
                and entry.adjusted_time is None:
            entry.adjusted_time = round(entry.finish_time + race.going_allowance, 3)

    return stats

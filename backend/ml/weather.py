"""Open-Meteo weather integration for Irish greyhound tracks.

Free, keyless API. Historical days come from the archive endpoint (one
call covers years); today/tomorrow come from the forecast endpoint so
predictions see the same features the training set carried. Track
coordinates are town-level — weather varies on a scale far coarser than a
stadium.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Iterable

import httpx
from sqlalchemy.orm import Session

from app.models.track import Track
from app.models.weather import TrackWeather

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "precipitation_sum,temperature_2m_mean,wind_speed_10m_max"

# Town-level coordinates for every location string in the tracks table.
TOWN_COORDS: dict[str, tuple[float, float]] = {
    "dublin": (53.339, -6.228),       # Shelbourne Park / Harolds Cross
    "cork": (51.888, -8.545),         # Curraheen Park
    "belfast": (54.532, -5.977),      # Drumbo Park
    "derry": (54.997, -7.309),
    "dundalk": (54.003, -6.416),
    "enniscorthy": (52.501, -6.566),
    "galway": (53.271, -9.062),
    "kilkenny": (52.654, -7.244),
    "limerick": (52.658, -8.630),
    "longford": (53.727, -7.793),
    "mullingar": (53.526, -7.338),
    "newbridge": (53.181, -6.797),
    "thurles": (52.679, -7.814),
    "tralee": (52.271, -9.700),
    "waterford": (52.246, -7.139),
    "youghal": (51.951, -7.850),
    "clonmel": (52.355, -7.704),
}


def coords_for_track(track: Track) -> tuple[float, float] | None:
    loc = (track.location or track.name or "").strip().lower()
    for town, coords in TOWN_COORDS.items():
        if town in loc:
            return coords
    return None


def _rows_from_payload(payload: dict) -> dict[date, dict]:
    daily = payload.get("daily") or {}
    out: dict[date, dict] = {}
    times = daily.get("time") or []
    for i, day_str in enumerate(times):
        def pick(key):
            arr = daily.get(key) or []
            return arr[i] if i < len(arr) else None
        out[date.fromisoformat(day_str)] = {
            "precip_mm": pick("precipitation_sum"),
            "temp_mean_c": pick("temperature_2m_mean"),
            "wind_max_kmh": pick("wind_speed_10m_max"),
        }
    return out


def _add_prev48(rows: dict[date, dict]) -> None:
    for d, vals in rows.items():
        p1 = rows.get(d - timedelta(days=1), {}).get("precip_mm")
        p2 = rows.get(d - timedelta(days=2), {}).get("precip_mm")
        if p1 is None and p2 is None:
            vals["precip_prev48h_mm"] = None
        else:
            vals["precip_prev48h_mm"] = (p1 or 0.0) + (p2 or 0.0)


# Successive waits after a 429. Multi-year archive calls are weighted
# heavily against Open-Meteo's free per-minute/hourly budgets, so a burst
# of town fetches can trip the limit mid-backfill; waiting out the window
# beats failing the whole run.
_RETRY_DELAYS = (30, 90, 300, 900)


def fetch_archive(lat: float, lon: float, start: date, end: date) -> dict[date, dict]:
    """One archive call for a full date range (multi-year is fine).

    Retries on 429, honouring Retry-After when the API sends one.
    """
    for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
        resp = httpx.get(ARCHIVE_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": DAILY_VARS, "timezone": "Europe/Dublin",
        }, timeout=120)
        if resp.status_code == 429 and delay is not None:
            retry_after = resp.headers.get("Retry-After")
            wait = max(delay, int(retry_after)) if (
                retry_after and retry_after.isdigit()) else delay
            logger.warning("Open-Meteo 429 (attempt %d), waiting %ds",
                           attempt + 1, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        rows = _rows_from_payload(resp.json())
        _add_prev48(rows)
        return rows


def fetch_forecast(lat: float, lon: float) -> dict[date, dict]:
    """Today + a few days ahead, with past days for the trailing-rain sum."""
    resp = httpx.get(FORECAST_URL, params={
        "latitude": lat, "longitude": lon,
        "daily": DAILY_VARS, "timezone": "Europe/Dublin",
        "past_days": 3, "forecast_days": 3,
    }, timeout=60)
    resp.raise_for_status()
    rows = _rows_from_payload(resp.json())
    _add_prev48(rows)
    return rows


def upsert_weather(
    db: Session, track_id: int, rows: dict[date, dict],
    only_dates: set[date] | None = None,
) -> int:
    n = 0
    existing = {
        w.date: w for w in
        db.query(TrackWeather).filter(TrackWeather.track_id == track_id).all()
    }
    for d, vals in rows.items():
        if only_dates is not None and d not in only_dates:
            continue
        if "precip_prev48h_mm" not in vals:
            continue
        row = existing.get(d)
        if row is None:
            db.add(TrackWeather(track_id=track_id, date=d, **vals))
            n += 1
        else:
            for k, v in vals.items():
                if v is not None:
                    setattr(row, k, v)
    return n


def backfill_archive(db: Session, log=None) -> dict:
    """Backfill daily weather for every track over the full race-date range.

    One archive call per distinct town covers the whole range; rows are
    stored per (track_id, date) only for dates that track actually raced.
    Idempotent — existing rows are updated, not duplicated. Returns a
    summary dict; `log` (if given) receives progress strings.
    """
    from sqlalchemy import func, text

    from app.models.race import Race

    def say(msg: str) -> None:
        if log:
            log(msg)

    lo, hi = db.query(func.min(Race.race_date), func.max(Race.race_date)).one()
    if lo is None:
        return {"inserted": 0, "table_rows": 0, "detail": "no races in DB"}
    if isinstance(lo, str):
        lo, hi = date.fromisoformat(lo), date.fromisoformat(hi)
    start = lo - timedelta(days=2)
    end = min(hi, date.today() - timedelta(days=3))  # archive lags a few days
    say(f"Backfilling weather {start} .. {end}")

    tracks = db.query(Track).all()
    by_coords: dict[tuple, list[Track]] = {}
    skipped: list[str] = []
    for t in tracks:
        c = coords_for_track(t)
        if c is None:
            skipped.append(t.code)
            say(f"  ! no coordinates for track {t.code} {t.name!r} — skipped")
            continue
        by_coords.setdefault(c, []).append(t)

    race_dates: dict[int, set] = {}
    for tid, rd in db.query(Race.track_id, Race.race_date).distinct():
        if isinstance(rd, str):
            rd = date.fromisoformat(rd)
        race_dates.setdefault(tid, set()).add(rd)

    # Dates already covered, so an interrupted run resumes where it left
    # off instead of re-spending API budget on towns it finished. The
    # trailing two weeks never count as covered: those rows may hold
    # forecast values (written at serve time) that the archive's actuals
    # should overwrite once available.
    refresh_after = end - timedelta(days=14)
    covered: dict[int, set] = {}
    for tid, d in db.query(TrackWeather.track_id, TrackWeather.date).filter(
            TrackWeather.precip_mm.isnot(None)):
        if isinstance(d, str):
            d = date.fromisoformat(d)
        if d > refresh_after:
            continue
        covered.setdefault(tid, set()).add(d)

    inserted = 0
    fetched_any = False
    for i, (coords, town_tracks) in enumerate(by_coords.items(), 1):
        town_needed = {
            t.id: {d for d in race_dates.get(t.id, set()) if d <= end}
            - covered.get(t.id, set())
            for t in town_tracks
        }
        if not any(town_needed.values()):
            say(f"[{i}/{len(by_coords)}] {[t.code for t in town_tracks]} "
                f"already covered — skipped")
            continue
        if fetched_any:
            time.sleep(8)  # pace the burst — see _RETRY_DELAYS note
        say(f"[{i}/{len(by_coords)}] fetching {coords} "
            f"for {[t.code for t in town_tracks]}")
        rows = fetch_archive(coords[0], coords[1], start, end)
        fetched_any = True
        for t in town_tracks:
            dates = race_dates.get(t.id)
            if not dates:
                continue
            inserted += upsert_weather(db, t.id, rows, only_dates=dates)
        db.commit()

    table_rows = db.execute(text("SELECT COUNT(*) FROM track_weather")).scalar()
    say(f"DONE: inserted {inserted} new rows; table now {table_rows} rows")
    return {
        "inserted": inserted,
        "table_rows": table_rows,
        "range": [str(start), str(end)],
        "towns": len(by_coords),
        "tracks_skipped_no_coords": skipped,
    }


def ensure_weather_for_date(db: Session, target: date) -> int:
    """Make sure every active track has a weather row for `target` (and the
    trailing days feeding precip_prev48h). Called before daily predictions
    so serve-time features exist. Uses the forecast endpoint."""
    total = 0
    tracks = db.query(Track).filter(Track.active.is_(True)).all()
    fetched: dict[tuple[float, float], dict[date, dict]] = {}
    for track in tracks:
        coords = coords_for_track(track)
        if coords is None:
            continue
        have = (
            db.query(TrackWeather)
            .filter(TrackWeather.track_id == track.id, TrackWeather.date == target)
            .first()
        )
        if have and have.precip_mm is not None:
            continue
        if coords not in fetched:
            try:
                fetched[coords] = fetch_forecast(*coords)
            except Exception as e:
                logger.warning("forecast fetch failed for %s: %s", track.name, e)
                fetched[coords] = {}
        rows = fetched[coords]
        if rows:
            total += upsert_weather(db, track.id, rows)
    db.commit()
    return total

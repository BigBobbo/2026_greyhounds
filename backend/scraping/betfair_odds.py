"""Betfair Exchange odds capture for Irish greyhound win markets.

Writes pre-race price snapshots into odds_snapshots — the table the
market-drift features and the live Benter blend read. Requires a (free,
delayed) Betfair app key plus a session token; see docs in the repo's
account-setup guide. Until credentials exist, everything here that
touches the network stays dormant — the matching logic is pure and unit
tested so the integration is a config change, not a build.

Betfair specifics encoded here:
  * greyhound racing eventTypeId is 4339; Irish markets carry
    marketCountries=["IE"]; the win market type is "WIN".
  * runner names on greyhound win markets are prefixed with the trap,
    e.g. "1. Some Dog" — the digit is the trap number, which is how we
    join to race_entries without fuzzy name matching.
  * venue comes in marketCatalogue.event.venue (e.g. "Shelbourne Park");
    marketStartTime is UTC ISO-8601.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.config import settings
from app.models.odds import OddsSnapshot
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track

logger = logging.getLogger(__name__)

API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"
LOGIN_URL = "https://identitysso.betfair.com/api/login"
GREYHOUND_EVENT_TYPE = "4339"
DUBLIN = ZoneInfo("Europe/Dublin")

# Betfair venue string -> our track name, where they differ.
VENUE_ALIASES = {
    "curraheen": "Curraheen Park",
    "cork": "Curraheen Park",
    "shelbourne": "Shelbourne Park",
    "thurles": "Thurles Park",
}


def normalise_venue(venue: str) -> str:
    v = (venue or "").strip().lower()
    for key, name in VENUE_ALIASES.items():
        if key in v:
            return name
    return venue.strip()


def parse_runner_trap(runner_name: str) -> int | None:
    """'3. Ballymac Star' -> 3."""
    m = re.match(r"\s*(\d)\s*\.", runner_name or "")
    return int(m.group(1)) if m else None


def market_local_date_time(market_start_iso: str) -> tuple[date, str]:
    """UTC marketStartTime -> (Irish race_date, 'HH:MM' Irish time)."""
    dt = datetime.fromisoformat(market_start_iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(DUBLIN)
    return local.date(), local.strftime("%H:%M")


def match_market_to_race(
    market: dict[str, Any],
    races: list[Any],
) -> Any | None:
    """Match one marketCatalogue entry to a Race row.

    `races` are candidate rows (same Irish date) carrying .race_time,
    .race_number and .track_name (joined). Match by venue + exact local
    start time; fall back to venue + closest time within 5 minutes.
    """
    venue = normalise_venue((market.get("event") or {}).get("venue", ""))
    _, hhmm = market_local_date_time(market["marketStartTime"])

    same_venue = [
        r for r in races
        if normalise_venue(getattr(r, "track_name", "")) == venue
    ]
    if not same_venue:
        return None

    def _time_str(r):
        t = getattr(r, "race_time", None)
        return str(t)[:5] if t is not None else None

    exact = [r for r in same_venue if _time_str(r) == hhmm]
    if len(exact) == 1:
        return exact[0]

    target_min = int(hhmm[:2]) * 60 + int(hhmm[3:])
    best, best_gap = None, 6  # minutes
    for r in same_venue:
        ts = _time_str(r)
        if not ts:
            continue
        gap = abs(int(ts[:2]) * 60 + int(ts[3:]) - target_min)
        if gap < best_gap:
            best, best_gap = r, gap
    return best


def snapshot_rows(
    market_book: dict[str, Any],
    catalogue: dict[str, Any],
    race_id: int,
    entries_by_trap: dict[int, Any],
    scraped_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Turn one listMarketBook result into odds_snapshots row dicts.

    Uses the best available back price per runner; skips runners without
    a price or without a trap match. Pure function — unit tested."""
    stamp = scraped_at or datetime.utcnow()
    names = {
        r["selectionId"]: r.get("runnerName", "")
        for r in catalogue.get("runners", [])
    }
    rows: list[dict[str, Any]] = []
    for runner in market_book.get("runners", []):
        if runner.get("status") not in (None, "ACTIVE"):
            continue
        trap = parse_runner_trap(names.get(runner.get("selectionId"), ""))
        entry = entries_by_trap.get(trap)
        if entry is None:
            continue
        backs = ((runner.get("ex") or {}).get("availableToBack")) or []
        if not backs:
            continue
        price = float(backs[0]["price"])
        if price <= 1.0:
            continue
        rows.append({
            "race_id": race_id,
            "dog_id": entry.dog_id,
            "bookmaker": "betfair_exchange",
            "odds_decimal": price,
            "implied_prob": 1.0 / price,
            "scraped_at": stamp,
            "is_sp": False,
        })
    return rows


class BetfairClient:
    """Minimal REST client. Interactive login (username/password) yields a
    session token good for the capture loop; certificate-based bot login
    can replace it later without touching the capture logic."""

    def __init__(self, app_key: str, session_token: str):
        self.app_key = app_key
        self.session_token = session_token

    @classmethod
    def login_interactive(cls, app_key: str, username: str, password: str) -> "BetfairClient":
        import httpx

        resp = httpx.post(
            LOGIN_URL,
            data={"username": username, "password": password},
            headers={"X-Application": app_key, "Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "SUCCESS":
            raise RuntimeError(f"Betfair login failed: {payload.get('error')}")
        return cls(app_key, payload["token"])

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        import httpx

        resp = httpx.post(
            f"{API_URL}/{path}/",
            json=body,
            headers={
                "X-Application": self.app_key,
                "X-Authentication": self.session_token,
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def irish_win_markets(self, hours_ahead: int = 12) -> list[dict[str, Any]]:
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        return self._post("listMarketCatalogue", {
            "filter": {
                "eventTypeIds": [GREYHOUND_EVENT_TYPE],
                "marketCountries": ["IE"],
                "marketTypeCodes": ["WIN"],
                "marketStartTime": {
                    "from": now.isoformat(),
                    "to": (now + timedelta(hours=hours_ahead)).isoformat(),
                },
            },
            "marketProjection": ["EVENT", "RUNNER_DESCRIPTION", "MARKET_START_TIME"],
            "maxResults": 200,
        })

    def market_books(self, market_ids: list[str]) -> list[dict[str, Any]]:
        return self._post("listMarketBook", {
            "marketIds": market_ids,
            "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
        })


def capture_once(db: Session, client: BetfairClient) -> int:
    """One capture pass: snapshot best back prices for every upcoming
    Irish win market that matches a scheduled race. Returns rows written."""
    markets = client.irish_win_markets()
    if not markets:
        logger.info("Odds capture: no upcoming Irish win markets")
        return 0

    dates = {market_local_date_time(m["marketStartTime"])[0] for m in markets}
    races = (
        db.query(Race, Track.name.label("track_name"))
        .join(Track, Race.track_id == Track.id)
        .filter(Race.race_date.in_(list(dates)))
        .all()
    )
    race_rows = [
        type("RaceRow", (), {
            "id": r.Race.id, "race_time": r.Race.race_time,
            "race_number": r.Race.race_number, "track_name": r.track_name,
        })()
        for r in races
    ]

    books = {b["marketId"]: b for b in client.market_books(
        [m["marketId"] for m in markets],
    )}

    written = 0
    for market in markets:
        race = match_market_to_race(market, race_rows)
        if race is None:
            continue
        book = books.get(market["marketId"])
        if not book:
            continue
        entries = db.query(RaceEntry).filter(RaceEntry.race_id == race.id).all()
        by_trap = {e.trap: e for e in entries}
        for row in snapshot_rows(book, market, race.id, by_trap):
            db.add(OddsSnapshot(**row))
            written += 1
    db.commit()
    logger.info("Odds capture: wrote %d snapshots", written)
    return written


def capture_from_settings(db: Session) -> int:
    """Entry point for the scheduler: builds a client from env settings.
    No-op (returns -1) until credentials are configured."""
    if not settings.betfair_api_key or not settings.betfair_username:
        logger.info("Odds capture dormant: Betfair credentials not configured")
        return -1
    client = BetfairClient.login_interactive(
        settings.betfair_api_key,
        settings.betfair_username,
        settings.betfair_password,
    )
    return capture_once(db, client)

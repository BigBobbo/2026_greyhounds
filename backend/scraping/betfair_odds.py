"""Betfair Exchange odds capture for Irish greyhound win markets.

Writes pre-race price snapshots into odds_snapshots — the table the
market-drift features and the live Benter blend read — and, after the
off, the Betfair Starting Price for the same markets, which is the
honest price a bet would actually have been struck at.

Betfair specifics encoded here:
  * greyhound racing eventTypeId is 4339; Irish markets carry
    marketCountries=["IE"]; the win market type is "WIN".
  * runner names on greyhound win markets are prefixed with the trap,
    e.g. "1. Some Dog" — the digit is the trap number, which is how we
    join to race_entries without fuzzy name matching.
  * venue comes in marketCatalogue.event.venue (e.g. "Shelbourne Park");
    marketStartTime is UTC ISO-8601.
  * listMarketBook is weighted per runner returned and caps the number of
    markets per request, so books are fetched in small chunks and only
    for markets close to the off.

Authentication supports both of Betfair's paths: interactive login
(username/password, simplest, but refused on accounts with two-factor
authentication) and certificate login (username/password + a client
certificate pair, which is the non-interactive path Betfair intends for
bots). Whichever is configured, the session token is cached and kept
alive rather than re-issued on every capture pass — Betfair rate-limits
logins far more tightly than data requests.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import date, datetime, timedelta, timezone
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
CERT_LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"
KEEPALIVE_URL = "https://identitysso.betfair.com/api/keepAlive"
GREYHOUND_EVENT_TYPE = "4339"
DUBLIN = ZoneInfo("Europe/Dublin")

# listMarketBook is weighted by runners returned and rejects oversized
# requests; 20 six-dog markets per call stays well inside the limit.
BOOK_CHUNK = 20

# Betfair venue string -> our track name, where they differ.
VENUE_ALIASES = {
    "curraheen": "Curraheen Park",
    "cork": "Curraheen Park",
    "shelbourne": "Shelbourne Park",
    "thurles": "Thurles Park",
}


class BetfairError(RuntimeError):
    """A Betfair API call failed. Carries the APING error code when the
    response body had one — 'INVALID_APP_KEY' and 'INVALID_SESSION_
    INFORMATION' are the two that actually matter operationally."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


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
    market_id = market_book.get("marketId") or catalogue.get("marketId")
    names = {
        r["selectionId"]: r.get("runnerName", "")
        for r in catalogue.get("runners", [])
    }
    rows: list[dict[str, Any]] = []
    for runner in market_book.get("runners", []):
        if runner.get("status") not in (None, "ACTIVE"):
            continue
        selection_id = runner.get("selectionId")
        trap = parse_runner_trap(names.get(selection_id, ""))
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
            "market_id": market_id,
            "selection_id": selection_id,
        })
    return rows


def bsp_rows(
    market_book: dict[str, Any],
    selection_to_dog: dict[int, int],
    race_id: int,
    scraped_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Turn a settled listMarketBook result into Betfair-SP row dicts.

    ``actualSP`` only exists once the market has been reconciled, so a
    book pulled too early yields nothing and the caller simply retries
    later. Runners removed before the off (status REMOVED) are skipped:
    they have no SP and were never bettable at the off. Pure function.
    """
    stamp = scraped_at or datetime.utcnow()
    market_id = market_book.get("marketId")
    rows: list[dict[str, Any]] = []
    for runner in market_book.get("runners", []):
        if runner.get("status") == "REMOVED":
            continue
        selection_id = runner.get("selectionId")
        dog_id = selection_to_dog.get(selection_id)
        if dog_id is None:
            continue
        sp = (runner.get("sp") or {}).get("actualSP")
        try:
            price = float(sp)
        except (TypeError, ValueError):
            continue
        if price <= 1.0:
            continue
        rows.append({
            "race_id": race_id,
            "dog_id": dog_id,
            "bookmaker": "betfair_sp",
            "odds_decimal": price,
            "implied_prob": 1.0 / price,
            "scraped_at": stamp,
            "is_sp": True,
            "market_id": market_id,
            "selection_id": selection_id,
        })
    return rows


class BetfairClient:
    """Minimal REST client over the Betting API."""

    def __init__(self, app_key: str, session_token: str):
        self.app_key = app_key
        self.session_token = session_token

    # --- authentication -------------------------------------------------

    @classmethod
    def login_interactive(cls, app_key: str, username: str,
                          password: str) -> "BetfairClient":
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
            raise BetfairError(
                f"Betfair login failed: {payload.get('error')}",
                code=payload.get("error"),
            )
        return cls(app_key, payload["token"])

    @classmethod
    def login_certificate(cls, app_key: str, username: str, password: str,
                          cert_file: str, key_file: str) -> "BetfairClient":
        """Non-interactive login with a client certificate pair. This is
        the path that works on accounts with two-factor authentication,
        where interactive login is refused outright."""
        import httpx

        resp = httpx.post(
            CERT_LOGIN_URL,
            data={"username": username, "password": password},
            headers={"X-Application": app_key, "Accept": "application/json"},
            cert=(cert_file, key_file),
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("loginStatus") != "SUCCESS":
            raise BetfairError(
                f"Betfair certificate login failed: {payload.get('loginStatus')}",
                code=payload.get("loginStatus"),
            )
        return cls(app_key, payload["sessionToken"])

    def keep_alive(self) -> None:
        """Extend the session. Betfair expires an idle session token after
        20 minutes; the capture cron runs on a similar cadence, so without
        this every pass would need a fresh login."""
        import httpx

        resp = httpx.post(
            KEEPALIVE_URL,
            headers={
                "X-Application": self.app_key,
                "X-Authentication": self.session_token,
                "Accept": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "SUCCESS":
            raise BetfairError(
                f"Betfair keepAlive failed: {payload.get('error')}",
                code=payload.get("error"),
            )

    # --- betting API ----------------------------------------------------

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
        if resp.status_code >= 400:
            raise BetfairError(*_api_error(path, resp))
        return resp.json()

    def _list_markets(self, hours_ahead: int, countries: list[str] | None,
                      max_results: int) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        market_filter: dict[str, Any] = {
            "eventTypeIds": [GREYHOUND_EVENT_TYPE],
            "marketTypeCodes": ["WIN"],
            "marketStartTime": {
                "from": now.isoformat(),
                "to": (now + timedelta(hours=hours_ahead)).isoformat(),
            },
        }
        if countries:
            market_filter["marketCountries"] = countries
        return self._post("listMarketCatalogue", {
            "filter": market_filter,
            "marketProjection": ["EVENT", "RUNNER_DESCRIPTION", "MARKET_START_TIME"],
            "maxResults": max_results,
        })

    def irish_win_markets(self, hours_ahead: int = 12,
                          fallback_venues: set[str] | None = None,
                          ) -> list[dict[str, Any]]:
        """Upcoming Irish greyhound win markets.

        The ``marketCountries=["IE"]`` filter is the cheap, precise query,
        but it keys off the event's registered country and has been known
        to omit Irish cards routed through a GB-registered event. When it
        comes back empty and we know which venues to look for, fall back
        to listing every greyhound win market and keeping the ones whose
        venue is one of ours — an empty list here is otherwise
        indistinguishable from a quiet night, and the capture would just
        silently write nothing.
        """
        markets = self._list_markets(hours_ahead, ["IE"], 200)
        if markets or not fallback_venues:
            return markets

        wanted = {normalise_venue(v).lower() for v in fallback_venues}
        every = self._list_markets(hours_ahead, None, 1000)
        matched = [
            m for m in every
            if normalise_venue((m.get("event") or {}).get("venue", "")).lower()
            in wanted
        ]
        if matched:
            logger.warning(
                "Betfair marketCountries=IE returned nothing; matched %d of "
                "%d unfiltered greyhound markets by venue instead",
                len(matched), len(every),
            )
        return matched

    def market_books(self, market_ids: list[str],
                     sp: bool = False) -> list[dict[str, Any]]:
        """Price books for the given markets, chunked to stay inside
        Betfair's per-request market cap and data-weight budget.

        ``sp=True`` asks for starting-price data instead of the exchange
        ladder — used after the off to read each runner's actual BSP.
        """
        projection: dict[str, Any] = (
            {"priceData": ["SP_TRADED"], "virtualise": False}
            if sp else
            {
                "priceData": ["EX_BEST_OFFERS"],
                # One price per side is all the snapshot stores; asking
                # for the default three deepens the data weight for
                # nothing.
                "exBestOffersOverrides": {"bestPricesDepth": 1},
                "virtualise": False,
            }
        )
        out: list[dict[str, Any]] = []
        for i in range(0, len(market_ids), BOOK_CHUNK):
            out.extend(self._post("listMarketBook", {
                "marketIds": market_ids[i:i + BOOK_CHUNK],
                "priceProjection": projection,
            }))
        return out


def _api_error(path: str, resp: Any) -> tuple[str, str | None]:
    """Extract an APING error code from a failed Betting API response.

    Betfair returns HTTP 400 with a JSON envelope for application errors
    (bad app key, expired session), so the status code alone tells the
    operator nothing useful."""
    code = None
    detail = resp.text[:300]
    try:
        payload = resp.json()
        code = (((payload.get("detail") or {}).get("APINGException") or {})
                .get("errorCode")) or payload.get("errorCode")
        if code:
            detail = code
    except Exception:
        pass
    return f"Betfair {path} failed (HTTP {resp.status_code}): {detail}", code


# --- cached session ----------------------------------------------------

_session_lock = threading.Lock()
_session: dict[str, Any] = {"client": None, "created_at": None}
# Betfair sessions live at most ~24h; re-login well before that.
SESSION_MAX_AGE = timedelta(hours=8)


def _login() -> BetfairClient:
    """Build a client from settings, preferring certificate login when a
    certificate pair is configured."""
    if settings.betfair_cert_file and settings.betfair_cert_key_file:
        return BetfairClient.login_certificate(
            settings.betfair_api_key, settings.betfair_username,
            settings.betfair_password, settings.betfair_cert_file,
            settings.betfair_cert_key_file,
        )
    return BetfairClient.login_interactive(
        settings.betfair_api_key, settings.betfair_username,
        settings.betfair_password,
    )


def get_client(force_new: bool = False) -> BetfairClient:
    """A logged-in client, reusing the cached session where possible.

    Logging in on every 20-minute capture pass burns Betfair's login rate
    limit for no benefit; a keepAlive is a fraction of the cost and keeps
    the same token valid. A failed keepAlive means the session is gone,
    so fall through to a fresh login."""
    with _session_lock:
        client = _session["client"]
        created = _session["created_at"]
        fresh_enough = (
            client is not None and created is not None
            and datetime.utcnow() - created < SESSION_MAX_AGE
        )
        if client is not None and fresh_enough and not force_new:
            try:
                client.keep_alive()
                return client
            except Exception as e:
                logger.info("Betfair keepAlive failed (%s); re-logging in", e)
        client = _login()
        _session.update(client=client, created_at=datetime.utcnow())
        return client


def reset_session() -> None:
    """Drop the cached session (used by tests and by credential changes)."""
    with _session_lock:
        _session.update(client=None, created_at=None)


def imminent(markets: list[dict[str, Any]], within_minutes: int) -> list[dict[str, Any]]:
    """Markets starting within the next `within_minutes`.

    Price books are only fetched for these: prices far from the off are
    thin and unrepresentative, and Betfair's data-request charging counts
    every runner returned, so pulling a full 12-hour card every pass
    wastes weight on markets nobody will bet."""
    now = datetime.now(timezone.utc)
    out = []
    for m in markets:
        start = datetime.fromisoformat(m["marketStartTime"].replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if 0 <= (start - now).total_seconds() <= within_minutes * 60:
            out.append(m)
    return out


def _race_rows(db: Session, dates: list[date]) -> list[Any]:
    """Lightweight race rows (id, time, number, track name) for matching."""
    rows = (
        db.query(Race, Track.name.label("track_name"))
        .join(Track, Race.track_id == Track.id)
        .filter(Race.race_date.in_(dates))
        .all()
    )
    return [
        type("RaceRow", (), {
            "id": r.Race.id, "race_time": r.Race.race_time,
            "race_number": r.Race.race_number, "track_name": r.track_name,
        })()
        for r in rows
    ]


def _known_venues(db: Session) -> set[str]:
    return {name for (name,) in db.query(Track.name).all() if name}


def capture_once(db: Session, client: BetfairClient,
                 within_minutes: int = 120) -> int:
    """One capture pass: snapshot best back prices for every upcoming
    Irish win market that matches a scheduled race. Returns rows written."""
    markets = client.irish_win_markets(fallback_venues=_known_venues(db))
    if not markets:
        logger.info("Odds capture: no upcoming Irish win markets")
        return 0
    markets = imminent(markets, within_minutes)
    if not markets:
        logger.info("Odds capture: no markets within %d minutes", within_minutes)
        return 0

    dates = {market_local_date_time(m["marketStartTime"])[0] for m in markets}
    race_rows = _race_rows(db, list(dates))

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


def capture_bsp(db: Session, client: BetfairClient,
                target_date: date | None = None) -> dict[str, int]:
    """Settle the day's captured markets at their Betfair Starting Price.

    The pre-race snapshots record what was showing at capture time; the
    BSP is what a bet actually struck at the off. Storing it turns each
    night's captures into training data for the model/market blend, which
    to date has only ever been fitted on bookmaker SPs scraped from GRI.

    Idempotent: a market that already has BSP rows is skipped, so this can
    run repeatedly (markets reconcile at different times after the off).
    """
    target_date = target_date or datetime.now(DUBLIN).date()

    # Markets we priced today, minus those already settled.
    priced = (
        db.query(OddsSnapshot.market_id, OddsSnapshot.race_id)
        .join(Race, OddsSnapshot.race_id == Race.id)
        .filter(Race.race_date == target_date)
        .filter(OddsSnapshot.market_id.isnot(None))
        .filter(OddsSnapshot.is_sp.is_(False))
        .distinct()
        .all()
    )
    settled = {
        m for (m,) in db.query(OddsSnapshot.market_id)
        .filter(OddsSnapshot.bookmaker == "betfair_sp")
        .filter(OddsSnapshot.market_id.isnot(None))
        .distinct().all()
    }
    pending = [(m, r) for m, r in priced if m not in settled]
    if not pending:
        logger.info("BSP capture: nothing pending for %s", target_date)
        return {"markets": 0, "rows": 0}

    race_by_market = dict(pending)
    books = client.market_books([m for m, _ in pending], sp=True)

    rows_written = 0
    markets_settled = 0
    for book in books:
        market_id = book.get("marketId")
        race_id = race_by_market.get(market_id)
        if race_id is None:
            continue
        # selectionId -> dog_id, learned from this market's own snapshots.
        mapping = {
            sel: dog for sel, dog in
            db.query(OddsSnapshot.selection_id, OddsSnapshot.dog_id)
            .filter(OddsSnapshot.market_id == market_id)
            .filter(OddsSnapshot.selection_id.isnot(None))
            .distinct().all()
        }
        rows = bsp_rows(book, mapping, race_id)
        if not rows:
            continue
        markets_settled += 1
        for row in rows:
            db.add(OddsSnapshot(**row))
            rows_written += 1
    db.commit()
    logger.info("BSP capture: settled %d market(s), wrote %d rows",
                markets_settled, rows_written)
    return {"markets": markets_settled, "rows": rows_written}


def credentials_configured() -> bool:
    return bool(settings.betfair_api_key and settings.betfair_username
                and settings.betfair_password)


def capture_from_settings(db: Session) -> int:
    """Entry point for the scheduler: builds a client from env settings.
    No-op (returns -1) until credentials are configured."""
    if not credentials_configured():
        logger.info("Odds capture dormant: Betfair credentials not configured")
        return -1
    return capture_once(db, get_client())


def capture_bsp_from_settings(db: Session,
                              target_date: date | None = None) -> dict[str, int]:
    """Scheduler entry point for post-race BSP settlement."""
    if not credentials_configured():
        logger.info("BSP capture dormant: Betfair credentials not configured")
        return {"markets": -1, "rows": -1}
    return capture_bsp(db, get_client(), target_date)


def _redact(text: str) -> str:
    """Strip anything credential-shaped from a message before it leaves
    the process — diagnostics are read by people who shouldn't see keys."""
    out = str(text)
    for secret in (settings.betfair_password, settings.betfair_api_key,
                   settings.betfair_username):
        if secret and len(secret) > 3:
            out = out.replace(secret, "***")
    return out[:500]


def diagnose(db: Session) -> dict[str, Any]:
    """End-to-end connectivity check that never returns credentials.

    Reports whether config is present, whether login succeeds, how many
    Irish win markets Betfair offers, and how many of them match races we
    have scheduled — the last number being the one that actually matters,
    since a market we can't match is a market we can't price.
    """
    result: dict[str, Any] = {
        "configured": credentials_configured(),
        "app_key_present": bool(settings.betfair_api_key),
        "username_present": bool(settings.betfair_username),
        "password_present": bool(settings.betfair_password),
        "login_mode": ("certificate" if settings.betfair_cert_file
                       and settings.betfair_cert_key_file else "interactive"),
    }
    if not result["configured"]:
        result["status"] = "not_configured"
        result["hint"] = ("Set BETFAIR_API_KEY, BETFAIR_USERNAME and "
                          "BETFAIR_PASSWORD in the deployment environment")
        return result

    try:
        client = get_client(force_new=True)
    except Exception as e:
        result["status"] = "login_failed"
        result["error"] = f"{type(e).__name__}: {_redact(e)}"
        result["hint"] = (
            "Common causes: wrong username/password; the app key is not yet "
            "activated; two-factor authentication on the account (set "
            "BETFAIR_CERT_FILE/BETFAIR_CERT_KEY_FILE to use certificate "
            "login instead); or Betfair geo-blocking the server's country."
        )
        return result
    result["login"] = "ok"

    try:
        markets = client.irish_win_markets(fallback_venues=_known_venues(db))
    except Exception as e:
        result["status"] = "market_list_failed"
        result["error"] = f"{type(e).__name__}: {_redact(e)}"
        code = getattr(e, "code", None)
        if code == "INVALID_APP_KEY":
            result["hint"] = (
                "Betfair rejected the application key. A newly created key "
                "is inactive until Betfair activates it, and the delayed and "
                "live keys are different strings — check you copied the one "
                "marked active on developer.betfair.com."
            )
        return result

    result["markets_next_12h"] = len(markets)
    soon = imminent(markets, 120)
    result["markets_next_2h"] = len(soon)

    dates = {market_local_date_time(m["marketStartTime"])[0] for m in markets}
    race_rows = _race_rows(db, list(dates))
    matched = 0
    unmatched_venues: list[str] = []
    samples: list[dict[str, Any]] = []
    for market in markets:
        race = match_market_to_race(market, race_rows)
        venue = (market.get("event") or {}).get("venue", "")
        _, hhmm = market_local_date_time(market["marketStartTime"])
        if race is None:
            if venue not in unmatched_venues:
                unmatched_venues.append(venue)
            continue
        matched += 1
        if len(samples) < 5:
            samples.append({"venue": venue, "time": hhmm,
                            "race_id": race.id,
                            "race_number": race.race_number})

    result["markets_matched_to_races"] = matched
    result["unmatched_venues"] = unmatched_venues[:10]
    result["sample_matches"] = samples
    result["snapshots_stored"] = db.query(OddsSnapshot.id).count()
    result["status"] = "ok" if matched else "no_matches"
    if not matched and markets:
        result["hint"] = (
            "Betfair markets were listed but none matched a scheduled race. "
            "Either today's cards haven't been scraped yet, or a venue name "
            "needs an alias in VENUE_ALIASES (see unmatched_venues)."
        )
    if not markets:
        result["hint"] = (
            "Betfair listed no Irish greyhound win markets in the next 12 "
            "hours. Outside racing hours that is expected; during an Irish "
            "card it means the exchange is not offering these markets to "
            "this account."
        )
    return result

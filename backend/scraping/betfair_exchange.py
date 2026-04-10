"""
Betfair Exchange API client for live/pre-race greyhound odds.

Uses the Betfair Exchange API (APING) to fetch current market odds
for upcoming greyhound races.

API docs: https://docs.developer.betfair.com/display/1smk3cen4v3lu3yomq5qye0ni/API+Overview

Authentication flow:
  1. Login via https://identitysso-cert.betfair.com/api/certlogin (cert-based)
     or https://identitysso.betfair.com/api/login (interactive)
  2. Use session token + API key for subsequent requests

Endpoints used:
  - listEventTypes: Confirm greyhound racing event type ID (4339)
  - listMarketCatalogue: Find upcoming greyhound markets
  - listMarketBook: Get current odds for a market
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

BETFAIR_LOGIN_URL = "https://identitysso.betfair.com/api/login"
BETFAIR_API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"

GREYHOUND_EVENT_TYPE_ID = "4339"


class BetfairClient:
    """Async client for the Betfair Exchange API."""

    def __init__(
        self,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self.api_key = api_key or settings.betfair_api_key
        self.username = username or settings.betfair_username
        self.password = password or settings.betfair_password
        self.session_token: str | None = None
        self._client: httpx.AsyncClient | None = None

    def is_configured(self) -> bool:
        """Check if Betfair credentials are set."""
        return bool(self.api_key and self.username and self.password)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def login(self) -> bool:
        """Authenticate with Betfair and obtain a session token."""
        if not self.is_configured():
            logger.error("Betfair credentials not configured")
            return False

        client = await self._get_client()
        try:
            resp = await client.post(
                BETFAIR_LOGIN_URL,
                headers={
                    "X-Application": self.api_key,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "username": self.username,
                    "password": self.password,
                },
            )
            data = resp.json()
            if data.get("status") == "SUCCESS":
                self.session_token = data["token"]
                logger.info("Betfair login successful")
                return True
            else:
                logger.error("Betfair login failed: %s", data.get("error", "unknown"))
                return False
        except Exception as e:
            logger.error("Betfair login error: %s", e)
            return False

    def _api_headers(self) -> dict[str, str]:
        return {
            "X-Application": self.api_key,
            "X-Authentication": self.session_token or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _api_request(self, operation: str, params: dict) -> Any:
        """Make a Betfair API request."""
        if not self.session_token:
            raise RuntimeError("Not logged in — call login() first")

        client = await self._get_client()
        url = f"{BETFAIR_API_URL}/{operation}/"
        resp = await client.post(url, headers=self._api_headers(), json=params)

        if resp.status_code != 200:
            logger.error("Betfair API error %d: %s", resp.status_code, resp.text[:500])
            raise RuntimeError(f"Betfair API error: {resp.status_code}")

        return resp.json()

    async def list_greyhound_markets(
        self,
        hours_ahead: int = 24,
        irish_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List upcoming greyhound win markets.

        Returns market catalogue entries with runner info.
        """
        now = datetime.now(timezone.utc)
        time_filter = {
            "from": now.isoformat(),
            "to": (now + timedelta(hours=hours_ahead)).isoformat(),
        }

        market_filter: dict[str, Any] = {
            "eventTypeIds": [GREYHOUND_EVENT_TYPE_ID],
            "marketTypeCodes": ["WIN"],
            "marketStartTime": time_filter,
        }

        if irish_only:
            market_filter["marketCountries"] = ["IE"]

        params = {
            "filter": market_filter,
            "maxResults": "200",
            "marketProjection": [
                "MARKET_START_TIME",
                "RUNNER_DESCRIPTION",
                "EVENT",
                "COMPETITION",
                "MARKET_DESCRIPTION",
            ],
        }

        try:
            markets = await self._api_request("listMarketCatalogue", params)
            logger.info("Found %d upcoming greyhound markets", len(markets))
            return markets
        except Exception as e:
            logger.error("Failed to list markets: %s", e)
            return []

    async def get_market_odds(self, market_ids: list[str]) -> list[dict[str, Any]]:
        """
        Get current odds for specified market IDs.

        Returns market books with runner prices.
        """
        if not market_ids:
            return []

        params = {
            "marketIds": market_ids,
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "SP_AVAILABLE", "SP_TRADED"],
            },
        }

        try:
            books = await self._api_request("listMarketBook", params)
            return books
        except Exception as e:
            logger.error("Failed to get market odds: %s", e)
            return []

    async def fetch_live_odds(
        self,
        hours_ahead: int = 24,
        irish_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Fetch live odds for upcoming greyhound races.

        Returns parsed odds records ready for the OddsSnapshot model.
        """
        markets = await self.list_greyhound_markets(hours_ahead, irish_only)
        if not markets:
            return []

        market_ids = [m["marketId"] for m in markets]

        # Build lookup: marketId -> market info
        market_info = {}
        for m in markets:
            runners = {}
            for r in m.get("runners", []):
                runners[r["selectionId"]] = {
                    "name": r.get("runnerName", "").upper(),
                    "sort_priority": r.get("sortPriority"),
                }
            market_info[m["marketId"]] = {
                "event_name": m.get("event", {}).get("name", ""),
                "market_start": m.get("marketStartTime", ""),
                "competition": m.get("competition", {}).get("name", ""),
                "runners": runners,
            }

        # Fetch odds in batches of 20 markets
        all_records = []
        for i in range(0, len(market_ids), 20):
            batch = market_ids[i : i + 20]
            books = await self.get_market_odds(batch)

            for book in books:
                mid = book.get("marketId")
                info = market_info.get(mid, {})

                for runner in book.get("runners", []):
                    sel_id = runner.get("selectionId")
                    runner_info = info.get("runners", {}).get(sel_id, {})

                    # Best back price
                    back_prices = runner.get("ex", {}).get("availableToBack", [])
                    best_back = back_prices[0]["price"] if back_prices else None

                    # Best lay price
                    lay_prices = runner.get("ex", {}).get("availableToLay", [])
                    best_lay = lay_prices[0]["price"] if lay_prices else None

                    # SP price
                    sp_price = runner.get("sp", {}).get("nearPrice")

                    # Use best back price as primary odds
                    odds = best_back or sp_price
                    if not odds:
                        continue

                    record = {
                        "market_id": mid,
                        "selection_id": sel_id,
                        "event_name": info.get("event_name", ""),
                        "market_start": info.get("market_start", ""),
                        "competition": info.get("competition", ""),
                        "dog_name": runner_info.get("name", ""),
                        "trap": runner_info.get("sort_priority"),
                        "odds_decimal": odds,
                        "implied_prob": round(1.0 / odds, 4) if odds > 0 else None,
                        "best_back": best_back,
                        "best_lay": best_lay,
                        "sp_near": sp_price,
                        "status": runner.get("status"),
                        "bookmaker": "betfair_exchange",
                    }
                    all_records.append(record)

        logger.info("Fetched live odds: %d records from %d markets", len(all_records), len(market_ids))
        return all_records

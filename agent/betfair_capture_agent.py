#!/usr/bin/env python3
"""Betfair price-capture agent — runs in Ireland, posts prices to the app.

Betfair blocks connections from the region the app is hosted in, so the
prices are captured here instead: this script logs in from the account
holder's own machine, in the account holder's own country, and forwards
best back prices for upcoming Irish greyhound win markets to the app.

The Betfair username and password never leave this machine. The only
credential shared with the server is an ingest token that can post
prices and nothing else.

Standard library only — no pip install needed.

Setup: copy agent.env.example to agent.env, fill it in, then run

    python3 betfair_capture_agent.py            # loop during race hours
    python3 betfair_capture_agent.py --once     # single pass (for cron)
    python3 betfair_capture_agent.py --check    # verify config, no posting
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LOGIN_URL = "https://identitysso.betfair.com/api/login"
API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"
GREYHOUND_EVENT_TYPE = "4339"

# Capture only markets starting within this window: prices far from the
# off are thin, and Betfair charges for data requests by volume.
WITHIN_MINUTES = 120
# Racing runs afternoon to late evening Irish time; outside these hours
# the loop sleeps instead of polling an empty card.
ACTIVE_HOURS = range(11, 23)
POLL_SECONDS = 20 * 60


def load_config() -> dict:
    """Read agent.env (KEY=value lines); real env vars win."""
    cfg = {}
    path = os.path.join(HERE, "agent.env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("BETFAIR_API_KEY", "BETFAIR_USERNAME", "BETFAIR_PASSWORD",
                "INGEST_URL", "INGEST_TOKEN"):
        if os.environ.get(key):
            cfg[key] = os.environ[key]
    missing = [k for k in ("BETFAIR_API_KEY", "BETFAIR_USERNAME",
                           "BETFAIR_PASSWORD", "INGEST_URL", "INGEST_TOKEN")
               if not cfg.get(k)]
    if missing:
        sys.exit(f"Missing settings in agent.env: {', '.join(missing)}")
    return cfg


def _post(url: str, data: bytes, headers: dict, timeout: int = 30):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode() or "{}")


def login(cfg: dict) -> str:
    """Interactive login -> session token."""
    body = urllib.parse.urlencode({
        "username": cfg["BETFAIR_USERNAME"],
        "password": cfg["BETFAIR_PASSWORD"],
    }).encode()
    try:
        payload = _post(LOGIN_URL, body, {
            "X-Application": cfg["BETFAIR_API_KEY"],
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        })
    except urllib.error.HTTPError as e:
        if e.code == 403:
            sys.exit(
                "Betfair returned 403 Forbidden. This machine's connection "
                "is being blocked — check you are in Ireland/UK and not on "
                "a VPN routing through another country."
            )
        sys.exit(f"Betfair login failed: HTTP {e.code}")
    if payload.get("status") != "SUCCESS":
        code = payload.get("error") or payload.get("status") or "UNKNOWN"
        sys.exit(f"Betfair login rejected: {code}\n\n{explain_login_error(code)}")
    return payload["token"]


# Betfair answers a rejected login with a specific code. Anything other
# than INVALID_USERNAME_OR_PASSWORD means the credentials were accepted
# and something about the ACCOUNT needs attention — so don't send people
# back to re-check a password that is provably fine.
LOGIN_ERRORS = {
    "INVALID_USERNAME_OR_PASSWORD":
        "The username or password is wrong. Note Betfair wants the "
        "username, not the email address.",
    "EMAIL_LOGIN_NOT_ALLOWED":
        "An email address was used. Use the Betfair username instead.",
    "SUSPENDED":
        "The Betfair account is suspended — the login itself worked. For a "
        "new account this is normally identity verification: log in at "
        "betfair.com in a browser and complete whatever it asks for "
        "(usually photo ID and proof of address).",
    "KYC_SUSPEND":
        "Betfair needs identity documents before the account can be used. "
        "Log in at betfair.com and upload what it asks for.",
    "ACCOUNT_NOT_FULLY_REGISTERED":
        "Registration was never finished. Log in at betfair.com and "
        "complete the remaining steps.",
    "ACCOUNT_NOW_LOCKED": "The account has just been locked. Contact Betfair.",
    "ACCOUNT_ALREADY_LOCKED": "The account is locked. Contact Betfair.",
    "PENDING_AUTH": "The account is awaiting authorisation from Betfair.",
    "CHANGE_PASSWORD_REQUIRED":
        "Betfair wants the password changed. Do that at betfair.com, then "
        "update agent.env with the new one.",
    "PERSONAL_MESSAGE_REQUIRED":
        "Betfair needs a personal message set on the account. Log in at "
        "betfair.com to set one.",
    "INTERNATIONAL_TERMS_ACCEPTANCE_REQUIRED":
        "New terms need accepting. Log in at betfair.com once and accept.",
    "CERT_AUTH_REQUIRED":
        "This account requires certificate-based login rather than a "
        "password. Tell Rob — the agent needs a different login method.",
    "SECURITY_RESTRICTED_LOCATION":
        "Betfair is blocking this location. Check the machine is in "
        "Ireland or the UK and not on a VPN.",
    "BETTING_RESTRICTED_LOCATION":
        "Betfair does not allow betting from this location.",
    "TEMPORARY_BAN_TOO_MANY_REQUESTS":
        "Too many login attempts. Wait a while before trying again.",
    "SELF_EXCLUDED": "The account is self-excluded and cannot be used.",
    "CLOSED": "The account is closed.",
}


def explain_login_error(code: str) -> str:
    return LOGIN_ERRORS.get(code, (
        "The credentials were accepted but Betfair would not open a "
        "session. Logging in at betfair.com in a browser usually reveals "
        "what the account needs."
    ))


def api(cfg: dict, token: str, path: str, body: dict):
    return _post(f"{API_URL}/{path}/", json.dumps(body).encode(), {
        "X-Application": cfg["BETFAIR_API_KEY"],
        "X-Authentication": token,
        "Content-Type": "application/json",
    })


def upcoming_markets(cfg: dict, token: str) -> list[dict]:
    now = datetime.now(timezone.utc)
    return api(cfg, token, "listMarketCatalogue", {
        "filter": {
            "eventTypeIds": [GREYHOUND_EVENT_TYPE],
            "marketCountries": ["IE"],
            "marketTypeCodes": ["WIN"],
            "marketStartTime": {
                "from": now.isoformat(),
                "to": (now + timedelta(minutes=WITHIN_MINUTES)).isoformat(),
            },
        },
        "marketProjection": ["EVENT", "RUNNER_DESCRIPTION", "MARKET_START_TIME"],
        "maxResults": 100,
    })


def collect(cfg: dict, token: str) -> list[dict]:
    """One capture pass -> payload markets ready to post."""
    catalogue = upcoming_markets(cfg, token)
    if not catalogue:
        return []
    ids = [m["marketId"] for m in catalogue]
    books = {b["marketId"]: b for b in api(cfg, token, "listMarketBook", {
        "marketIds": ids,
        "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
    })}

    out = []
    for market in catalogue:
        book = books.get(market["marketId"])
        if not book:
            continue
        names = {r["selectionId"]: r.get("runnerName", "")
                 for r in market.get("runners", [])}
        runners = []
        for runner in book.get("runners", []):
            if runner.get("status") not in (None, "ACTIVE"):
                continue
            backs = ((runner.get("ex") or {}).get("availableToBack")) or []
            if not backs:
                continue
            runners.append({
                "runner_name": names.get(runner.get("selectionId"), ""),
                "price": float(backs[0]["price"]),
            })
        if runners:
            out.append({
                "market_id": market["marketId"],
                "venue": (market.get("event") or {}).get("venue", ""),
                "market_start_time": market["marketStartTime"],
                "runners": runners,
            })
    return out


def post_to_app(cfg: dict, markets: list[dict]) -> dict:
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "markets": markets,
    }
    url = cfg["INGEST_URL"].rstrip("/")
    return _post(url, json.dumps(payload).encode(), {
        "Authorization": f"Bearer {cfg['INGEST_TOKEN']}",
        "Content-Type": "application/json",
    }, timeout=60)


def one_pass(cfg: dict, post: bool = True) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    try:
        token = login(cfg)
        markets = collect(cfg, token)
    except urllib.error.URLError as e:
        print(f"[{stamp}] Betfair unreachable: {e}", flush=True)
        return
    if not markets:
        print(f"[{stamp}] no Irish markets in the next {WITHIN_MINUTES} min",
              flush=True)
        return
    runners = sum(len(m["runners"]) for m in markets)
    if not post:
        print(f"[{stamp}] would send {len(markets)} markets, {runners} prices")
        for m in markets[:3]:
            print(f"    {m['venue']} {m['market_start_time']} "
                  f"({len(m['runners'])} runners)")
        return
    try:
        result = post_to_app(cfg, markets)
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:200]
        print(f"[{stamp}] app rejected the prices: HTTP {e.code} {detail}",
              flush=True)
        return
    except urllib.error.URLError as e:
        print(f"[{stamp}] could not reach the app: {e}", flush=True)
        return
    print(f"[{stamp}] sent {len(markets)} markets / {runners} prices -> "
          f"{result.get('snapshots_written', 0)} stored, "
          f"{result.get('markets_unmatched', 0)} unmatched", flush=True)
    if result.get("unmatched"):
        print(f"    unmatched: {', '.join(result['unmatched'])}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="single pass then exit (for cron/Task Scheduler)")
    ap.add_argument("--check", action="store_true",
                    help="verify login and show what would be sent")
    args = ap.parse_args()
    cfg = load_config()

    if args.check:
        one_pass(cfg, post=False)
        print("Config looks good.")
        return
    if args.once:
        one_pass(cfg)
        return

    print(f"Capture agent running. Polling every {POLL_SECONDS // 60} minutes "
          f"between {ACTIVE_HOURS[0]}:00 and {ACTIVE_HOURS[-1]}:59. "
          "Press Ctrl-C to stop.", flush=True)
    while True:
        try:
            if datetime.now().hour in ACTIVE_HOURS:
                one_pass(cfg)
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception as e:  # never let one bad pass kill the agent
            print(f"[{datetime.now():%H:%M:%S}] unexpected error: {e}",
                  flush=True)
            time.sleep(60)


if __name__ == "__main__":
    main()

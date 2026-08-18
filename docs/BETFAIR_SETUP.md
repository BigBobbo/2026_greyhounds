# Betfair activation checklist

Everything on the exchange side is built and unit-tested; turning it on is
a configuration change plus two verification calls. This is the order to
do it in, and what each step proves.

## 0. What you need

| Value | Where it comes from |
|---|---|
| `BETFAIR_API_KEY` | developer.betfair.com → your application → **the key marked ACTIVE**. A new key is inactive until Betfair activates it, and the *delayed* and *live* keys are different strings. |
| `BETFAIR_USERNAME` | your Betfair account username (not the email address, unless that is your username) |
| `BETFAIR_PASSWORD` | your Betfair account password |

The free **delayed** key is enough for this system: it serves prices with
a short delay pre-race, which is fine for drift features, for blending,
and for a sheet a human executes by hand. It is *not* enough to place
bets programmatically at a guaranteed price — nothing here does that.

If the account has **two-factor authentication** enabled, Betfair refuses
interactive login outright. Use certificate login instead: register a
client certificate with Betfair, put the pair on the host, and set
`BETFAIR_CERT_FILE` and `BETFAIR_CERT_KEY_FILE`. When both are set the
client switches to `identitysso-cert.betfair.com` automatically.

## 1. Set the variables

In Railway → the backend service → Variables:

```
BETFAIR_API_KEY=...
BETFAIR_USERNAME=...
BETFAIR_PASSWORD=...
```

Redeploy (Railway does this on variable change). Nothing else needs
touching: the capture job is already scheduled and has been running as a
no-op, logging `Odds capture dormant: Betfair credentials not configured`.

Never commit these to the repo. `backend/.env.example` lists the names
only.

## 2. Prove the credentials work

```bash
curl -H "Authorization: Bearer $ADMIN_BACKUP_TOKEN" \
     https://<host>/api/admin/betfair-check
```

Returns counts and statuses only — never a credential, and errors are
scrubbed of anything credential-shaped. What the `status` field means:

| status | meaning | what to do |
|---|---|---|
| `not_configured` | the variables didn't reach the process | check the Railway variable names and that the service redeployed |
| `login_failed` | Betfair rejected the login | see the `hint` field: wrong password, unactivated key, 2FA (use certificate login), or geo-blocking |
| `market_list_failed` | login worked, the Betting API refused | usually `INVALID_APP_KEY` — the key isn't activated, or it's the wrong one of the two |
| `no_matches` | markets listed but none matched a scheduled race | check `unmatched_venues`; either today's cards aren't scraped yet or a venue needs an alias |
| `ok` | login, markets and race matching all work | go to step 3 |

Run this **during Irish racing hours** (roughly 18:00–22:00 Dublin, plus
afternoon cards). Outside them Betfair legitimately offers no markets and
you'll get `markets_next_12h: 0`, which tells you nothing about whether
the integration works.

If `unmatched_venues` shows a venue we do run, add it to `VENUE_ALIASES`
in `backend/scraping/betfair_odds.py` and redeploy.

## 3. Capture a book by hand

```bash
curl -X POST -H "Authorization: Bearer $ADMIN_BACKUP_TOKEN" \
     https://<host>/api/admin/capture-odds
```

Returns `{"snapshots_written": N}`. N is one row per priced runner in
every market starting in the next two hours. Zero outside race hours is
expected; zero during a card with `betfair-check` reporting `ok` means
the markets exist but no book came back — worth investigating.

From here the scheduler takes over: every 20 minutes between 12:00 and
22:00 Dublin.

## 4. Settle the day at Betfair SP

After the last race:

```bash
curl -X POST -H "Authorization: Bearer $ADMIN_BACKUP_TOKEN" \
     "https://<host>/api/admin/settle-bsp?target_date=2026-08-18"
```

This re-lists the exact markets we priced and records each runner's
**actual** Betfair Starting Price. It runs automatically at 23:15 and
00:15 Dublin, is idempotent, and skips markets already settled — markets
reconcile at different times after the off, so running it twice is normal
and free.

Why it matters: the pre-race snapshots say what was showing when we
looked; the BSP is the price a bet actually struck at. Accumulating both
is what will eventually let the model/market blend be refitted on
exchange prices instead of the bookmaker SPs scraped from GRI, which is
the single biggest known weakness in the current blend.

## 5. The live bet sheet

This is the point of the whole exercise. The existing sheet
(`generate_bet_sheet.py`) is odds-conditional and model-only, because
there was no price feed. The new one blends:

```bash
curl -X POST -H "Authorization: Bearer $ADMIN_BACKUP_TOKEN" \
     "https://<host>/api/admin/live-bet-sheet?experiment_id=<id>"
```

or, locally:

```bash
cd backend
DATABASE_URL=... python3 scripts/generate_live_bet_sheet.py \
    --experiment-id <id> [--date 2026-08-18]
```

Per race: model probabilities → blended with the de-vigged exchange book
using the trained bundle's alpha/beta → joint Kelly against the actual
back price → daily exposure cap. Races whose book is incomplete or whose
prices are stale (default: older than 45 minutes) fall back to model-only
and are excluded from staking, labelled as such in the output. That is
deliberate: a half-book cannot be de-vigged honestly, and a stale price
is not a price you can bet.

Run it close to the first race — the fresher the book, the more races
qualify.

## Ongoing schedule

| Job | When (Dublin) |
|---|---|
| Exchange odds capture | every 20 min, 12:00–22:00 |
| Betfair SP settlement | 23:15 and 00:15 (settles today and yesterday) |

Both are no-ops without credentials, so nothing changes if the variables
are ever removed.

## Notes on cost and etiquette

Betfair charges data requests by weight, counted per runner returned.
The capture deliberately keeps that small: only markets within two hours
of the off get a price book, only one price per side is requested, and
books are fetched in chunks of 20 markets. Session tokens are cached and
kept alive rather than re-issued each pass — Betfair rate-limits logins
much more tightly than data requests.

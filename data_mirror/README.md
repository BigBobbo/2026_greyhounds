# Production data mirror

Faithful extract of the production database on Railway (`2026greyhounds-production.up.railway.app`),
taken 2026-07-30 by crawling the deployed app's own REST API (per-race detail calls for entries;
paged lists for the rest). Zero failed fetches; counts match the live `/api/scraping/data-summary`
census exactly.

## Contents

| File | Rows | Notes |
|---|---|---|
| `tracks.jsonl.gz` | 31 | All Irish tracks |
| `dogs.jsonl.gz` | 36,441 | Name, sire/dam, trainer, IDs |
| `races.jsonl.gz` | 88,082 | 2021-01 → 2026-07 (plus Curraheen 2015–16); all status `resulted` |
| `race_entries.jsonl.gz` | 518,695 | One row per runner per race |
| `misc.json` | — | bankroll_config, feature_definitions (46), experiments (58), bets (empty) |

## Field coverage (entries)

- finish_position 100%, weight 100%, finish_time 98.2%, in-running comment 97.4%
- starting price 87.6% overall — 90–93% for 2022–2026, 67% for 2021
- beaten_distance 70.9%
- sectional_time **0%** — never scraped; sourced from GRI dog-profile pages in Phase 0 enrichment

Each JSONL line is the API response object for one row; `id` fields are the production
primary keys, so foreign keys (`race_id`, `dog_id`, `track_id`) join across files.

## Betfair BSP history (`bsp/`)

Betfair's free daily BSP CSVs (greyhound win/place markets, including Irish
tracks) are the price backbone for honest backtesting — but
`promo.betfair.com` geo-blocks non-IE/UK IPs, so they cannot be fetched from
this repo's cloud environment. Run `download_bsp.py` from any Irish/UK
connection (plain Python 3, no dependencies, resumable):

```bash
python3 data_mirror/download_bsp.py        # 2021-01-01 → today, IRE rows only
```

then commit the resulting `bsp/win_greyhound.csv.gz` and
`bsp/place_greyhound.csv.gz`.

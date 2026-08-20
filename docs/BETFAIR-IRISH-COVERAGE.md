# Betfair does not price Irish greyhound racing

**Status: confirmed, 2026-08-20. This is a fact about Betfair's market
coverage, not a bug, and no code change affects it.**

## Evidence

Two independent observations, months apart, agree:

1. **Historical.** The free Betfair BSP (starting price) archive contains
   no Irish greyhound data at all — verified via the Wayback Machine
   after the live promo site geo-blocked us. Recorded in
   `data_mirror/README.md`.
2. **Live.** The capture agent, running in Ireland on a verified account
   with a working delayed app key, queried Betfair for every greyhound
   WIN market in a 24-hour window spanning a full Irish racing evening:

   ```
   Greyhound WIN markets in the next 24h (all countries): 80
     GB: 45 markets across 4 venues (Hove, Monmore, Newcastle, Valley)
     AU: 35 markets across 3 venues (Healesville, Murray Bridge, Townsville)
   No Irish (IE) greyhound markets in this window.
   ```

Betfair's greyhound business is British and Australian. GRI tracks are
absent.

## What this does NOT invalidate

The evaluation in `retrain_report_2026-08-01.json` never used Betfair
prices. Its market data is **SP scraped from GRI results** — Irish
on-course starting prices — and execution was modelled as `SP - 5%`
slippage with 5% commission. Losing Betfair costs us nothing that the
backtest depended on.

## What it does cost

A **pre-race price feed**. SP is a closing price recorded after the race,
so it can price a backtest but cannot be read before a bet is placed.
Betfair was the intended live source; it does not cover these races.

This matters most for the market blend, and the distinction between the
headline numbers is important:

| Strategy (test set, 5,192 races, SP−5%, 5% commission) | ROI |
|---|---|
| Blended (model + market) Kelly | +50.8% |
| Blended value bets | +35.3% |
| **Model-only Kelly** | **+28.9%** |
| Model-only value bets | −4.1% |
| Model-only top pick | −7.2% |
| Back the favourite (baseline) | −12.8% |

The blended figures require a market probability as an *input* to the
estimate. Without a live feed that input is SP, which is not knowable
before the off — so **the blended numbers are not attainable**, and
quoting +50.8% as the expected return would be wrong.

The realizable candidate is **model-only Kelly, +28.9% over 1,607 bets**.
It still needs a price at bet time, but only to size and filter, which a
human reading prices off a betting app can supply.

## Caution on the model-only number

One night of live scoring (2026-08-01, 97 strict pre-race races) found
the market's de-vigged SP probabilities *sharper* than the model's:
winner log loss 1.579 for the market vs 1.723 for the model, and the
favourite beat the model's top pick on strike rate (35.7% vs 24.3%).
The fitted blend agrees — it weights the market above the model
(β = 1.12 vs α = 0.71).

A model that is less accurate than the market can still profit, by
disagreeing selectively where the disagreement is large. But it is a
weaker position than the headline suggested, and it is why the paper
week measures rather than assumes.

## Options for a live Irish price feed

1. **Manual (available today, zero build).** The bet sheet already
   prints a *minimum acceptable price* per selection. The person
   executing checks the live price in any Irish bookmaker app and skips
   anything below it. This is what the sheet was designed for, and it
   needs no feed at all.
2. **Scrape Irish bookmakers** (Paddy Power, BoyleSports, …) for pre-race
   greyhound prices. Buildable; enables automation and the blend, at the
   cost of ongoing scraper maintenance.
3. **Pivot to British racing**, where Betfair provides both live exchange
   prices and historical BSP. The modelling transfers; the data source
   does not — GRI would have to be replaced with GBGB, and the model
   retrained from scratch.

## Status of the capture agent

Kept, and working. It authenticates and queries correctly; it simply
finds nothing for Ireland. If British racing is ever added it works
unchanged, and `--explore` re-checks coverage at any time in case
Betfair's offering changes.

The server-side odds-capture cron stays dormant, as it has been.

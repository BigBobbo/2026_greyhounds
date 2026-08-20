# Orientation — what this app is, and what is actually true about it

Written 2026-08-20 for whoever picks this up next. `CLAUDE.md` covers
commands and layout; this covers **state, reasoning and traps** — the
things that cost days to rediscover.

Read the "Hard-won facts" section before changing anything in the
modelling or betting path. Several of them are counter-intuitive and all
of them were learned the expensive way.

---

## 1. What it does, honestly

Scrapes Irish greyhound racing from GRI, builds features, trains models,
and produces daily win probabilities and a bet sheet.

**Is it profitable? Unknown, and the evidence is mixed.** Be precise
about which number you are quoting:

| Claim | Status |
|---|---|
| Model beats an uninformed baseline | **Established.** Beats uniform-field log loss and Brier on held-out data, and calibration is good through the 5–30% band where most runners sit. |
| Model-only Kelly returns +28.9% | **Backtest only**, 1,607 bets. Not reproduced live. |
| Blended model+market returns +50.8% | **Not attainable.** See §4. Do not quote this. |
| The strategy makes money in practice | **Unproven.** Live replay over 368 races gives −7% with a CI of [−41%, +32%] — too few bets to conclude either way. |
| Betting top picks at SP loses money | **Established.** −16.5%, CI [−28.6%, −4.0%]. The favourite loses too, −19.7%. |

The market is a genuinely strong opponent. On every measurement so far
the market's implied probabilities have been **sharper than the model's**
(1 Aug: winner log loss 1.579 market vs 1.723 model; the favourite
out-strikes the model's top pick 35.8% to 31.1%). The fitted blend agrees,
weighting the market above the model (β 1.12 vs α 0.71). A model less
accurate than the market can still profit by disagreeing selectively —
but assume nothing.

---

## 2. Where things live

```
backend/
  app/          FastAPI: api/ routers, models/ ORM, services/ logic
  ml/           feature building, training, evaluation, staking, blend
  scraping/     GRI scrapers + DB pipeline
  scripts/      operational entry points (see §6)
  alembic/      migrations — the schema authority
frontend/       React/TS, deploys to Vercel
agent/          Betfair price agent that runs on someone's home machine
data_mirror/    committed DB dumps so a fresh container can bootstrap
docs/           audit, findings, and the daily prediction record
```

`backend/models_store/retrain_model.joblib` is the live model artifact,
**committed deliberately** despite a global `*.joblib` ignore rule. It
must stay in git: the Docker image and the nightly bootstrap both need
it. `data/` is not a home for it — that path is shadowed by the Railway
volume mount.

---

## 3. How it runs in production

Railway deploys automatically from the **default branch**, which is
`claude/greyhound-prediction-app-4vNeO`. **There is no `main` branch.**
That is unusual but intentional; check the default before assuming.

APScheduler crons (all Europe/Dublin, in `app/tasks/scheduler.py`):

| When | What |
|---|---|
| 23:00 | today's results |
| 08:00 | yesterday's results (late posts) |
| 04:30 | re-scrape trailing 14 days — GRI **amends** results after publication |
| 05:30 | profile-scrape newly seen dogs |
| Sun 06:00 | weather archive top-up (replaces forecast rows with actuals) |
| 12–22, :00/:20/:40 | Betfair odds capture — permanently dormant, see §4 |
| 11:30 | daily predictions for today+tomorrow (a `ModelSchedule` row) |

Separately, a **scheduled Claude session** runs
`backend/scripts/nightly_from_api.py` each morning: it reads production
over HTTP, writes the day's prediction sheet and bet sheet into
`docs/predictions/`, scores yesterday's, commits, and refreshes three
artifacts.

Admin endpoints, all gated by `ADMIN_BACKUP_TOKEN` and 404 without it:
`/api/admin/backup`, `/backfill-weather`, `/backfill-dogs`,
`/register-model`, `/betfair-check`, `/capture-odds`.
`/api/admin/odds-ingest` uses the narrower `ODDS_INGEST_TOKEN` instead,
because that credential lives on someone's home machine.

---

## 4. Betfair: closed, do not reopen

**Betfair does not price Irish greyhound racing.** Confirmed twice: no
Irish data in their historical BSP archive, and a live query in a
24-hour window spanning a full Irish racing evening returned 80
greyhound markets — 45 GB, 35 AU, **zero IE**. Full evidence in
`BETFAIR-IRISH-COVERAGE.md`.

Consequences:

- The odds-capture cron, `scraping/betfair_odds.py`, the ingest endpoint
  and `agent/` all work correctly and will simply never see Irish
  markets. They are kept for a possible British pivot.
- **There is no live pre-race price feed, and none is expected.** The bet
  sheet is designed around that: it prints a *minimum acceptable price*
  per selection, and a human checks the real price in a betting app.
- Anything requiring a market probability as an *input* is therefore
  unavailable in practice. That is why the +50.8% blended figure is
  unreachable: without a feed, its market input is SP, which is a
  **closing** price and unknowable before the off.

Also note: the app's own host is geo-blocked by Betfair (US region →
HTTP 403 before authentication). That is why the agent exists at all.

---

## 5. Hard-won facts

**Race IDs are database-local.** An id means nothing outside the database
that issued it. Sheets generated from a local DB scored against
production compared predictions to unrelated races and produced a
plausible 15.6% hit rate — wrong, and only detectable because the
calibration table was scrambled. `nightly_from_api.py` now verifies
date/track/race-number per race and refuses to score on mismatch. **Never
loosen that check.**

**Leakage is the default failure mode.** Every aggregate must be
point-in-time (`merge_asof`, `allow_exact_matches=False`). Post-race
fields — SP, weight vs career average, going-conditional trap bias — are
listed in `ml/feature_availability.py` and excluded from training. If you
add a feature, ask what is knowable before the off, then add it to that
list if the answer is "nothing".

**SP is a closing price.** It prices a backtest honestly but cannot be
read before betting. Any strategy whose *selection* depends on SP is
unimplementable, however good its backtest looks.

**GRI amends published results.** Corrected SPs, weights, even runner
identities days later. Hence the 14-day re-scrape and
corrections-allowed upserts. Do not assume a scraped result is final.

**The deployed rule ≠ the validated rule.** The backtest that produced
+28.9% used `min_edge=0.02` measured against **gross** odds
(hardcoded in `ml/evaluation.py`); the bet sheet uses `min_edge` from the
live bankroll config (**0.05**) measured **after commission**. Live
replay shows all variants perform alike (−6.6% to −7.5%), but they differ
in volume: 98 bets vs 53 over the same window. **This inconsistency is
open** — see §7.

**macOS ships Python 3.9.** `agent/` runs on machines we do not control,
so it is standard-library-only with deferred annotations, enforced by
`backend/tests/test_agent_compat.py`. A PEP 604 union in a signature once
killed it at import.

**Betfair replies in XML as well as JSON**, and sometimes with a 200
status on a rejection. Parse defensively; the error code carries the
whole diagnosis.

---

## 6. Scripts worth knowing

| Script | Purpose |
|---|---|
| `nightly_from_api.py` | the daily cycle. Reads production over HTTP, stdlib only, ~13s. **Start here.** |
| `evaluate_bet_sheet_strategy.py` | replays betting rules against settled results with bootstrap CIs |
| `local_retrain_eval.py` | full retrain + honest evaluation on the local mirror (slow) |
| `register_retrain_model.py` | idempotently registers the committed artifact as an experiment |
| `backfill_dog_profiles.py` | per-dog career enrichment from GRI profile pages |
| `backfill_weather.py` | Open-Meteo archive backfill (rate-limit aware, resumes) |

`generate_prediction_sheet.py` / `generate_bet_sheet.py` are the older
local-DB equivalents of the nightly script. They still work but produce
**local race ids**, so their output cannot be scored against production.

---

## 7. Open issues

1. **`min_edge` mismatch** (§5). Decide 0.02-gross or 0.05-net and make
   `evaluation.py`, `staking.py` and the bankroll config agree. Favour
   0.02-gross during measurement: same returns, ~2× the bets, so a
   verdict arrives sooner.
2. **`PredictionDataError` on a few races daily.** Production's schedule
   reports "partial" most days, skipping 4–10 races. Undiagnosed.
3. **Sample size.** ~5 qualifying bets/day means a meaningful verdict is
   months away, not weeks. The nightly record accrues in
   `docs/predictions/record.csv`.
4. **Model vs market.** The market has out-predicted the model in every
   measurement. Worth attacking directly — better features, or accepting
   the model's role is to disagree selectively rather than to out-rank.
5. **No `main` branch**, and the default branch has an unusual name.
   Renaming would mean updating Railway's deploy branch.

---

## 8. Before you trust a number

- Was it measured on data the model could have seen? (leakage)
- Does its *selection* use anything unknowable before the off? (SP)
- What is the confidence interval? Most samples here are small enough
  that point estimates mislead.
- Do the race ids come from the database being scored against?
- Is it the rule that is actually deployed, or a variant?

The project's value depends on measuring honestly. A number that is
quietly wrong is worse than no number, because it gets acted on.

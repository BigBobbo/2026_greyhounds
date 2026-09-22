# Pre-registered decision protocol for the paper ledger

Written 2026-08-27, before checkpoint data existed. The rule below is
fixed; changing it after seeing ledger data invalidates the experiment.

## The quantity

Per-bet profit per 1 unit staked, at the recorded tissue price
(`pnl_tissue / stake` per row of `docs/predictions/paper_ledger.csv`),
pooled over all settled bets of the certified rule (VERDICT-2026-08-21).

## Checkpoints and rule

The ledger is *read for a decision* only when cumulative settled bets
first reach **100, 250, 500, and 1000**. At each checkpoint compute a
bootstrap CI on ROI (race-level resampling, 4000 draws) at the
Bonferroni-adjusted level **98.75%** (= 95% split over 4 looks).

- **GO (real money, small)**: CI lower bound > 0 at any checkpoint.
- **STOP (strategy dead)**: CI upper bound < 0 at any checkpoint.
- **Otherwise**: keep accruing. If checkpoint 1000 is reached with the
  CI still straddling zero and the point estimate below +3%, treat as
  STOP — an edge too small to survive execution friction isn't worth
  the effort.

Between checkpoints the daily totals are reported for transparency but
carry no decision weight. Secondary diagnostics (mean CLV, P&L at SP,
per-track splits) inform *model work*, never the go/stop call.

## Status log

| Date | Bets settled | Note |
|---|---|---|
| 2026-08-27 | 23 | Protocol registered. Next decision read at 100 bets. |
| 2026-09-05 | 105 | **CHECKPOINT 100: CONTINUE.** ROI +19.1%, 98.75% CI [−22.2%, +63.0%] — straddles zero. Next read at 250. |
| 2026-09-08 | 126 | **Full ledger audit vs Racing Post results** after a phantom win was found (the DB attaches results by trap, so a withdrawn dog inherits its reserve's result). All 39 wins positively confirmed; 5 bets on non-runners voided (1 recorded win, 4 recorded losses — rows removed, stakes returned). Checkpoint-100 recomputed on the corrected as-of-Sep-4 set: n=101, ROI +22.5%, 98.75% CI [−20.9%, +66.1%] — **verdict unchanged: CONTINUE**. Daily settlement now verifies every row against RP before it enters the ledger. |
| 2026-09-22 | 254 | **CHECKPOINT 250: GO.** ROI +41.6%, 98.75% CI [+10.5%, +74.6%] — lower bound clears zero (race-level bootstrap cross-check agrees, [+10.1%, +73.2%]). First GO signal of the protocol. Verdict computed *after* voiding a phantom win found in verification (Stripe Blueboy, Enniscorthy 20:30, a non-runner whose reserve won in its trap); the settle script's printed CI [+11.8%, +75.6%] included that row, GO both ways. Secondary: P&L at SP +7.7%, mean CLV 1.328. **GO = the pre-registered green light for real money at small stakes** — a signal to the operator, not an instruction the loop can execute (no bookmaker/exchange automation exists; placement remains a human action). Accrual continues; next read at 500. |

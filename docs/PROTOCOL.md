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

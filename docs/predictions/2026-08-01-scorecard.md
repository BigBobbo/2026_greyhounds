# Scorecard — prediction sheet 2026-08-01

Scored 2026-08-02 with the pre-registered methodology in
`backend/scripts/score_prediction_sheet.py` (committed alongside the
predictions, before results existed). Sheet generated 2026-08-01 18:07 UTC;
strict cohort = races starting after that. 108/109 races returned a result
(one Waterford race did not run).

## Pre-registered metrics

| Metric | All races (108) | Strict pre-race (97) |
|---|---|---|
| Top-pick hit rate | 31/108 = **28.7%** | 24/97 = **24.7%** |
| Model's own expected hit rate | 32.8% | 32.9% |
| Winner log loss (model / uniform) | **1.696** / 1.788 | **1.738** / 1.788 |
| Brier per runner (model / uniform) | **0.1349** / 0.1392 | **0.1380** / 0.1393 |

Reliability, strict cohort (predicted → observed):

| Bucket | Predicted | Observed | n |
|---|---|---|---|
| 0–5% | 3.0% | 7.8% | 64 |
| 5–10% | 7.9% | 8.5% | 94 |
| 10–15% | 12.5% | 12.4% | 121 |
| 15–20% | 17.5% | 15.8% | 120 |
| 20–30% | 24.4% | 27.5% | 138 |
| 30–40% | 33.8% | 29.4% | 34 |
| 40%+ | 70.4% | 22.2% | 9 |

Top-pick hit rate by tier (strict): strong 4/15, moderate 3/8, weak 9/40,
avoid 8/34.

## Supplementary (post-hoc, labelled as such): vs the SP market

Strict-cohort races with a complete SP book (70; 27 lacked full SPs):

| | Model | Market (de-vigged SP) |
|---|---|---|
| Winner log loss | 1.723 | **1.579** |
| Top pick / favourite hit rate | 24.3% | **35.7%** |

Model top pick == market favourite in only 19/70 (27.1%) races. Note the
market's structural advantage: SP forms at the off, hours after the 18:07
sheet, with scratches, going and money flows priced in.

## Reading

- Real skill vs ignorance: the model beats the uniform baseline on log
  loss and Brier in both cohorts, and calibration is good through the
  5–30% range where most runners live (within ~2pts per bucket).
- Two warning signs, both small-sample: the strict-cohort hit rate came
  in ~8pts under the model's own expectation (z ≈ −1.7, borderline), and
  the few 40%+ "banker" picks went 2/9 — the top of the probability
  range looks overconfident on this night.
- The market was clearly sharper on the night, as expected: the honest
  backtest already showed the fitted blend leaning on the market
  (β = 1.12 vs α = 0.71). Model-only predictions are the raw ingredient;
  the validated edge comes from blending with exchange prices and betting
  only where model and market disagree at value — which is what the
  Betfair feed enables.
- One evening is ~100 races; every number above carries wide error bars.
  Keep accumulating nightly sheets before drawing conclusions.

# Prediction-Time Feature Availability Audit

## Why this document exists

Cross-validation accuracy during training is materially higher than real-world
accuracy on upcoming races. The root cause is that **a large portion of the
features used by the model depend on data that GRI Ireland only publishes
*after* the race has run** (finish positions, finish/sectional times, weights,
starting prices, going). At training time these are all populated. At
prediction time on a future card they are `NULL`, and the prediction pipeline
silently fills the missing values with a training-set median or `0` instead of
loudly failing. The result is a distribution shift between train and serve.

This document enumerates every input the model uses, classifies each as
*pre-race available* or *post-race only*, and shows exactly where and how the
silent fallbacks happen today, with a fix plan that prefers loud failure over
imputation.

---

## 1. Scraper output: what each endpoint actually returns

### 1.1 Historical / resulted races — `parse_results_page`
File: `backend/scraping/gri_scraper.py:75-126`

Per entry the GRI results page yields:
`trap, dog_name, finish_position, finish_time, sectional_time, beaten_distance,
weight_kg, starting_price, sp_decimal, comment, grade_at_entry`
Per race: `going, prize_money, distance_m, grade, race_type, race_time`.

### 1.2 Upcoming race cards — `parse_card_page`
File: `backend/scraping/gri_scraper.py:432-495`

Per entry the card page only yields `{trap, dog_name}`
(`gri_scraper.py:475`). At the race level it explicitly hard-codes:

```python
"going": None,
"prize_money": None,
"status": "scheduled",
```
(`gri_scraper.py:488-490`).

### 1.3 Upcoming form pages — `parse_card_form_page`
File: `backend/scraping/gri_scraper.py:528-615`

Adds `trainer_name, owner_name, sire_name, dam_name, best_time` for each trap.
The docstring is explicit (`gri_scraper.py:533`):

> the page does NOT carry an intended weight for the upcoming race (dogs are
> weighed-in on the night).

### 1.4 Delta between historical and upcoming scrapes

| Field | Historical | Upcoming card | Reason it's missing pre-race |
|---|---|---|---|
| `finish_position` | yes | **no** | Race hasn't run |
| `finish_time` | yes | **no** | Race hasn't run |
| `sectional_time` | yes | **no** | Race hasn't run |
| `beaten_distance` | yes | **no** | Race hasn't run |
| `weight_kg` | yes | **no** | Weigh-in is on the night |
| `starting_price` / `sp_decimal` | yes | **no** | GRI only publishes SP on results |
| `comment` (in-running) | yes | **no** | Generated post-race |
| `going` | yes | **no** | Reported with results |
| `prize_money` | yes | **no** | Published with results |
| `going_allowance` | yes | **no** | Derived from `going` |
| `adjusted_time` | yes | **no** | Requires `finish_time` + going |
| `grade_at_entry` | yes | partial | Sometimes on entry slip |
| `trap`, `dog_name` | yes | yes | – |
| `trainer_name`, `sire_name`, `dam_name`, `best_time` | yes | yes (Tier 2 form page) | – |
| `distance_m`, `grade`, `race_type`, `race_time` | yes | yes | – |

---

## 2. How missing fields propagate into the database

`backend/scraping/db_pipeline.py:174-189` writes new entries with
`entry_data.get(...)` for every result column. For a scheduled race the
scraper passes nothing for those keys, so each one becomes `NULL` in the row.
For updates, lines 158-168 only write if a value is `not None`, so an entry
re-scraped before the race still has all result columns `NULL`.

`backend/app/models/race.py` — `Race.going`, `going_allowance`, `prize_money`
are nullable.
`backend/app/models/race_entry.py` — `finish_position`, `finish_time`,
`sectional_time`, `adjusted_time`, `beaten_distance`, `weight_kg`,
`starting_price`, `sp_decimal`, `comment`, `grade_at_entry`,
`days_since_last` are all nullable.

**Net effect:** every `RaceEntry` row for a scheduled race is a hollow shell
of `(race_id, dog_id, trap, dog_name)` plus form metadata on the dog itself.

---

## 3. How features are computed for the *current* (upcoming) entry

`backend/app/services/prediction_service.py:93-171` runs three feature passes
per upcoming entry:

1. **Visual / code features** from dog history.
   `get_dog_history` (`feature_engine.py:73-77`) correctly filters
   `Race.status == "resulted"` and `race_date < before_date`, so it only sees
   completed prior races. This part is fine *if* the dog has prior history.
2. **Built-in race-context features** —
   `ml/race_features.py:compute_race_context_features` (line 48-160).
3. **ELO and H2H features** — `compute_elo_features_batch`,
   `compute_h2h_features_batch`.

Most features are computed from the **dog's prior resulted races**, which is
correct. The leakage is concentrated in features that read from the **current
race entry** (the unresolved row) or that depend on **race-level** state that
only exists post-race.

### 3.1 Built-in features that read fields from the *current* (unresolved) entry

| Feature | Reads from current entry | Behaviour for upcoming race |
|---|---|---|
| `days_since_last` | `entry.days_since_last` (`race_features.py:74`) | Falls back to `(race_date - history.race_date.max()).days` if entry field NULL. OK *if* dog has history; else returns `None`. |
| `weight_change` | `entry.weight_kg` (`race_features.py:88`) | `weight_kg` is **always** NULL pre-race ⇒ `_weight_change(history, None)` returns `None`. |
| `trap_win_rate_at_track` | `race_context["trap"]` (pre-race ✓) | Filters DB on `RaceEntry.finish_position.isnot(None)` (`race_features.py:_trap_win_rate`). Returns `None` if no prior history at this `(track, distance, trap)` bucket. |
| `early_speed_ratio` / `is_front_runner` | dog history only (✓) | If past sectional/comment data is sparse for this dog, returns `None`. |
| `track_speed_rating` | dog `best_time` and history | OK if history present. |
| `dog_age_years` / `dog_age_squared` | `Dog.birth_date` | `birth_date` is rarely scraped for Irish dogs ⇒ commonly `None`. |

### 3.2 SP / odds features (the biggest source of skew)

- **`_add_sp_features`** (`dataset_builder.py:620`) — gated by
  `include_sp_features=False` *by default* (`dataset_builder.py:37`). The
  docstring (lines 624-630) explicitly warns it leaks post-race data, so this
  one is safe by default — but if you enable the toggle in the Training Lab
  the model learns to lean on the closing-line SP and there is no way to
  reproduce that signal at predict time.
- **`_add_odds_snapshot_features`** (`dataset_builder.py:700`) — gated on by
  default at training and at predict time (`prediction_service.py:364, 413`).
  This is *intended* to use a separate `odds_snapshots` table populated by a
  live odds scraper. **The scraper is not currently writing to that table**
  (`_add_odds_snapshot_features` returns early when `snap_count == 0`,
  `dataset_builder.py:751-753`). So at training time the three columns are
  computed but mostly NaN, then `dataset_builder.py:224` fills them with the
  column median (which is itself NaN, then `0.0`). At predict time the same
  thing happens. This branch is benign today *only because* the snapshot
  table is empty; the moment any historical snapshots exist for resulted
  races but not for upcoming ones, training will see real values and predict
  will see zeros — this is exactly the train/serve skew you are observing.

### 3.3 Pace-shape features

`_add_pace_shape_features` (`dataset_builder.py:842`) builds
`num_front_runners_in_race`, `is_sole_front_runner`, `pace_pressure`,
`early_speed_rank`, `is_predicted_leader` from `is_front_runner` and
`early_speed_ratio` per entry in the field. These are pre-race derivable
*in principle*, because they only depend on each dog's past comments and
sectionals. But if any dog in the field has no prior comments or sectional
data the per-dog inputs collapse to `None`, the race-level aggregates collapse
toward a constant, and within-race variation falls to near zero. The comment
in `prediction_service.py:404-410` describes the prior incarnation of the same
bug.

### 3.4 Race-relative features

`add_race_relative_features` (`dataset_builder.py`) emits `<feature>_vs_field`
and `<feature>_rank_in_field` columns per base feature. These are derived
from in-race comparisons of the base features above, so any silent imputation
in the base features cascades into the relatives — every dog has the same
imputed value, the rank is therefore arbitrary, and `_vs_field` is `0.0` for
the entire race.

---

## 4. Every silent-imputation site (today's behaviour)

| Site | What it does | Effect at predict time |
|---|---|---|
| `feature_engine.py:151` `series = df[metric].dropna()` | Drops NaN rows from history before aggregating | Returns `None` if every prior race lacks the metric |
| `feature_engine.py:154-155` `if series.empty: return None` | Returns `None` if no values left | Cascades to NaN downstream |
| `dataset_builder.py:224` `X = X.fillna(X.median()).fillna(0.0)` | **Training-time** imputation | Fixes the training median values that get baked into the experiment |
| `prediction_service.py:388` `X.apply(pd.to_numeric, errors="coerce")` | Coerces None → NaN | Required because `object` dtype breaks XGBoost/LGBM |
| `prediction_service.py:391-392` `X = X.fillna(feature_medians)` | **Silent fill with the training-set medians snapshotted in the experiment** | This is the headline bug — every missing pre-race input becomes the median across all *resulted* races |
| `prediction_service.py:394` `X = X.fillna(0)` | Catch-all `0` for any feature without a training median | New / renamed features all become `0` for the race |
| `prediction_service.py:400-402` `X[col] = 0` for missing FeatureDefinition columns | Adds zero-valued columns | Same as above |
| `prediction_service.py:432-436` Trained-feature alignment, missing columns filled with `feature_medians.get(col, 0.0)` | Aligns to the model's exact column set | Any column not produced by the predict path is silently faked |
| `prediction_service.py:440` `X = X.fillna(0)` | Final sweep | Last line of defence, also silent |
| `_add_sp_features` if `include_sp_features=True` | Adds `current_sp_*` from `entry.sp_decimal` which is `NULL` pre-race | Whole column becomes NaN ⇒ filled with median ⇒ every dog gets the average SP-implied probability |
| `_add_odds_snapshot_features` (`dataset_builder.py:700`) | Reads from `odds_snapshots`, which is empty | Three columns silently NaN → median → `0` |

`feature_medians` is built once in `dataset_builder.py:241`
(`X.median().to_dict()`) and persisted on the experiment. So at predict time
a missing `weight_change` is replaced with the median weight delta across all
resulted races (typically near `0.0`); a missing `current_sp_implied_prob` is
replaced with `~1/8` (the average implied probability across all dogs); a
missing `trap_win_rate_at_track` is replaced with the average win rate per
trap (`~0.166`). None of these are flagged anywhere.

---

## 5. Master delta table

Legend: ✅ usable for upcoming races, ⚠️ usable only when the dog has
prior resulted races, ❌ not available pre-race today.

| Feature / column | Source | Pre-race available? | Today's fallback | Severity |
|---|---|---|---|---|
| `finish_position`, `finish_time`, `sectional_time`, `beaten_distance`, `adjusted_time` (current race) | `RaceEntry.*` | ❌ | n/a — never read for current race | ok |
| `entry.weight_kg` (current race) | `RaceEntry.weight_kg` | ❌ | `weight_change` ⇒ `None` ⇒ training median `≈0` | high |
| `entry.starting_price`, `sp_decimal` (current race) | `RaceEntry.sp_decimal` | ❌ | every `current_sp_*` feature ⇒ training median; **only if** `include_sp_features=True` | critical when on |
| `entry.days_since_last` (current race) | `RaceEntry.days_since_last` | ⚠️ recomputed from history | training median (`0` if dog never raced) | medium |
| `entry.grade_at_entry` (current race) | `RaceEntry.grade_at_entry` | ⚠️ sometimes on card | feature derived; falls back to median | low |
| `race.going`, `going_allowance`, `prize_money` | `Race.*` | ❌ | features that filter on `going` lose a filter dimension; `adjusted_time`-based features collapse | medium |
| `mean_finish_time_lastN`, `min_finish_time_lastN`, etc. | dog history | ⚠️ | empty history → `None` → training median | medium |
| `trap_win_rate_at_track` | DB aggregate over resulted races | ⚠️ | `None` if no prior bucket; filled with `0` for new dogs at new bucket | medium |
| `early_speed_ratio`, `is_front_runner` | dog history (sectional/comment) | ⚠️ | `None` for dogs with no sectional/comment data; filled with median | medium |
| `early_speed_x_trap`, `early_speed_x_inside`, `early_speed_x_outside`, `front_runner_x_*` | derived | ⚠️ | inherits from `early_speed_ratio` and `is_front_runner` | medium |
| `weight_change` | needs **today's** weight | ❌ | always `None` ⇒ median `≈0` | high |
| `dog_age_years`, `dog_age_squared` | `Dog.birth_date` | ⚠️ rarely scraped | `None` ⇒ median | low |
| `position_consistency`, `career_races` | dog history | ⚠️ | `0` for first-time runners | low |
| `track_speed_rating` | dog history vs track baseline | ⚠️ | `None` ⇒ median | low |
| Trainer / sire stats | DB aggregates | ⚠️ | `None` for new trainers/sires ⇒ median | low |
| ELO ratings (overall, per-distance, per-track) | chronological pass over resulted races | ⚠️ | uninitialised dog ⇒ default ELO (`1500`) which *is* a real value, OK | ok |
| H2H features vs today's opponents | DB | ⚠️ | `None` for unseen matchups ⇒ median | low |
| `current_sp_*` (10 features in `_add_sp_features`) | current SP | ❌ | only emitted when toggle is on; if on, silently median-imputed at predict | critical when on |
| `opening_to_sp_drift`, `odds_steam_rate`, `cross_book_disagreement` | `odds_snapshots` table | ❌ today (table empty) | NaN → training median → 0 | latent — will become high once snapshots are collected |
| Pace-shape (`num_front_runners_in_race`, `is_sole_front_runner`, `pace_pressure`, `early_speed_rank`, `is_predicted_leader`) | per-race aggregation of `is_front_runner`/`early_speed_ratio` | ⚠️ collapses toward constant when inputs are sparse | per-race aggregates near constant ⇒ no within-race variation | high |
| Race-relative (`<feature>_vs_field`, `<feature>_rank_in_field`) | derived | inherits from base | constant within race when base imputed | high |

---

## 6. Fix plan (loud-failure first) — **applied**

The fixes below were implemented in commit
`Fail loudly on missing pre-race features for upcoming race predictions`.
Each subsection documents the change and the new file/line that owns it.

### 6.1 Classify every feature as `pre_race` vs `post_race` ✅

Implemented as a single source of truth in
`backend/ml/feature_availability.py`. The module exports
`POST_RACE_FEATURE_NAMES` (a dict of feature_name → human-readable reason)
and a helper `post_race_features_in_use(trained_feature_names)` that the
prediction service and the new preflight endpoint both consume. The
covered feature set is `weight_change`, the eleven `current_sp_*` columns
emitted by `_add_sp_features`, and the three odds-snapshot drift columns
emitted by `_add_odds_snapshot_features`.

Stored on the FeatureDefinition row was rejected as overkill: the user
defines visual/code features that are computed only from dog history, and
those are pre-race by construction. The post-race set is small, fixed,
and lives entirely in the built-in feature builders, so a static list is
the right home for it.

### 6.2 Refuse to use post-race features for scheduled races ✅

`predict_race` (`backend/app/services/prediction_service.py`) now reads
`Race.status` for the target race. When the race is `scheduled` and the
trained feature list contains any post-race-only column, the function
raises `PredictionDataError` (a `ValueError` subclass exported from
`feature_availability.py`) listing the offending features and their
human-readable reasons. The check runs **before** any feature
computation so the failure is fast and the error message tells the user
exactly which experiment is unsafe to serve on upcoming races.

### 6.3 Stop silent imputation at predict time ✅

The cascade in `prediction_service.predict_race` now branches on
`is_scheduled`:

* **Scheduled (upcoming) races** call `_raise_for_missing_features` after
  the `pd.to_numeric` coerce step. The helper builds a per-feature
  breakdown listing up to six `(trap, dog_name)` offenders and raises
  `PredictionDataError`. A second copy of the same check runs after the
  trained-feature alignment step so a NaN introduced by race-relative or
  pace-shape derivation cannot slip through.
* **Resulted races** (back-tests, results comparison) keep the existing
  `fillna(feature_medians)` path, but now log the count of imputed cells
  at INFO so an unexpected drift in resulted-race coverage is visible.

### 6.4 Drop the `feature_medians` plumbing for serve-time fills ⚠️ partial

For scheduled races the `feature_medians` cascade is now bypassed
entirely: any NaN raises before alignment. For resulted races the
medians are still used so historical back-tests do not crash on a single
debutant with no prior history — that branch is the same code path
training uses, so the imputation is consistent between fit and score.
Tightening the training-time fill in `dataset_builder.py:224` was left
out of scope: shrinking it changes which features survive the
all-NaN-column drop and would alter every existing experiment's feature
count. Revisit once the dataset assembly is otherwise stable.

### 6.5 Disable the empty-snapshot odds branch ✅

`include_odds_snapshot_features` is now `False` by default in:

* `backend/ml/dataset_builder.py:build_dataset` (signature default).
* Both call sites in `backend/app/services/training_service.py`
  (the `split_cfg.get(...)` defaults).
* `backend/app/services/prediction_service.predict_race` (the
  `split_cfg.get(...)` default).

Existing experiments that had it explicitly enabled in their
`split_config` are unaffected — they continue to set the toggle to
`True`. A retrain is what flips a model to the new default.

### 6.6 Don't read `entry.weight_kg` from the current race — handled via §6.1

The function in `race_features.py` is intentionally left alone. At
**training** time `RaceEntry.weight_kg` *is* populated (it's the actual
weigh-in result for a resulted race), so `weight_change` is a genuine
signal there. The leakage only appears at **predict** time on a
scheduled race, where the column is NULL and the silent median fill
fired. That is now caught by §6.1 + §6.2: any model trained with
`weight_change` is refused on a scheduled race with a `PredictionDataError`
that names the feature and explains why. Renaming the function to
`weight_trend_in_history` would require recomputing every existing
experiment, which is more disruption than the loud-failure guard.

### 6.7 Pre-flight check API endpoint ✅

`GET /api/predictions/preflight/{race_id}?experiment_id=…` is live in
`backend/app/api/predictions.py`. It computes the same feature matrix
`predict_race` would use, then returns:

```jsonc
{
  "race_id": 123,
  "race_status": "scheduled",
  "experiment_id": 7,
  "n_entries": 6,
  "post_race_features_in_use": [
    { "feature": "weight_change",
      "reason": "RaceEntry.weight_kg (weigh-in is on the night)" }
  ],
  "missing_features": [
    { "feature": "early_speed_ratio",
      "missing_for": [
        { "entry_id": 901, "trap": 4, "dog_name": "BALLINA BLAZE" }
      ]
    }
  ],
  "would_fail": true
}
```

The endpoint never trains, never imputes, and never raises — it lets the
UI surface a "this prediction will fail because …" banner before the
user clicks predict.

### 6.8 Compute the recoverable (category C) features at scrape time ✅

The audit splits missing fields into three buckets:

* **A. Future data** — outcomes that simply don't exist yet (finish time,
  SP, sectional time, etc.). These remain unavailable; the post-race
  classifier in §6.1 plus the predict-time guard in §6.2 keeps them out
  of any model used on scheduled races.
* **B. Exists pre-race but not on the GRI card page** — `weight_kg`
  (trackside weigh-in only), `going` (post-race report). These would
  require a different scrape source; out of scope for this pass.
* **C. Computable from data we already store** — `days_since_last`,
  `grade_at_entry`, plus the history-derived features (`grade_movement`,
  `early_speed_ratio`, `is_front_runner`, ELO, trainer/sire stats, H2H,
  pace-shape).

The history-derived features were already computed correctly when the
dog has a non-empty resulted-race history. The two row-level fields
(`days_since_last`, `grade_at_entry`) were *not* being populated for
upcoming races because the GRI card page omits them. They are now
backfilled at scrape time in
`backend/scraping/db_pipeline.upsert_race_entry`:

* `grade_at_entry` defaults to `race.grade` when the card scrape doesn't
  carry an entry-level grade slip.
* `days_since_last` is computed by `_last_resulted_race_date` from the
  dog's most recent prior resulted race.

Both fields are filled on the new-entry path *and* the existing-entry
update path, so re-scraping an already-loaded card heals rows that
predate the fix without a separate migration.

For the C features that depend on a non-empty history (everything that
reads `dog_history`), genuine debutants still produce NaN at predict
time. The preflight endpoint surfaces this distinctly in the
`entries_missing_history` field, and each cell in `missing_features`
now carries a `reason` tag (`post_race_data` / `dog_has_no_history` /
`history_field_missing`) so the UI can tell the user whether the gap is
fixable by re-scraping prior races (`history_field_missing`) or
fundamental (`dog_has_no_history`, `post_race_data`).

### 6.9 Exclude post-race features at training time ✅

`build_dataset` accepts `exclude_post_race_features` (defaults to
`True`). With the flag on, every column listed in
`POST_RACE_FEATURE_NAMES` is dropped from the training matrix before
the trainer ever sees it, with the dropped names logged at INFO. The
toggle threads through both `training_service.run_experiment` call
sites via `split_cfg.get("exclude_post_race_features", True)`, so any
new experiment is built without them by default. Setting it to `False`
is reserved for back-test-only experiments where you accept that
`predict_race` will refuse to serve the resulting model on a scheduled
race.

`backend/scripts/audit_experiments_for_post_race_features.py` walks
every completed experiment, loads its model artifact, intersects the
trained `feature_names` with `POST_RACE_FEATURE_NAMES`, and prints a
`BLOCKED` / `ok` line per experiment plus the offending features. Use
it to size the retrain blast radius before flipping the strict guard
on in production.

## 7. Retrain workflow

1. `python backend/scripts/audit_experiments_for_post_race_features.py`
   to list the experiments the strict guard will refuse to serve.
2. Re-scrape today's upcoming cards so `grade_at_entry` and
   `days_since_last` get backfilled on the existing rows (the
   db_pipeline backfill only fires on new scrapes).
3. Retrain the affected experiments. The default
   `exclude_post_race_features=True` is enough; no explicit override is
   needed unless you want a back-test-only model.
4. Compare the new CV scores against the old ones. A drop is expected
   and is the corrected baseline — the old number was inflated by
   features the live system can never see.
5. Hit `/predictions/preflight/{race_id}?experiment_id=NEW_ID` for a
   couple of upcoming races and confirm `would_fail: false`.
6. Repoint live prediction to the new experiment.

---

## 8. Two kinds of missingness, two different fixes

The original strict mode treated every NaN cell at predict time as a
hard error. That was correct for one class of feature and wrong for
another, and refusing the whole race when one dog had thin history
turned out to be a worse user experience than silently imputing.

| Class | Examples | Train-time behaviour | Predict-time behaviour | Right answer |
|---|---|---|---|---|
| Post-race-only | `weight_change`, `current_sp_*`, odds-snapshot drift | Always populated (resulted races have all post-race fields) | Always NaN on a scheduled race | **Hard refuse**: the model learned to use a signal that cannot exist live. |
| History-dependent | `mean_finish_time_last5`, `early_speed_ratio`, ELO, trainer/sire stats | NaN for debutants and lightly-raced dogs; median-filled by `dataset_builder.py`'s `X.fillna(X.median()).fillna(0.0)` | NaN for debutants and lightly-raced dogs | **Match training**: median-fill at predict time too, then surface a per-entry `data_completeness` score so Kelly staking can downweight bets on thin-history dogs. |

The split is now implemented in `prediction_service.predict_race`:

* The up-front guard keeps refusing scheduled races when the trained
  feature list contains any post-race-only column.
* The per-(entry, feature) NaN error has been removed for the
  history-dependent class. NaN cells are median-filled exactly the way
  `dataset_builder.build_dataset` did at fit time.
* Each prediction now carries a `data_completeness` field (0.0–1.0)
  measured on the raw feature matrix before imputation. A debutant
  whose features were entirely median-filled gets ~0.0; a veteran
  with full coverage gets ~1.0.
* `/predictions/preflight/{race_id}` reports the same number per
  entry under `data_completeness[]`, alongside the existing
  `entries_missing_history` and `missing_features` lists.

### Phase 2: switch GBM trainers to native NaN passthrough (deferred)

Research review (XGBoost paper, LightGBM advanced topics, sklearn
HistGradientBoosting docs, Benter and Bolton & Chapman racing
literature) is unambiguous that median-imputing before fitting a
gradient-boosted tree model is strictly worse than passing NaN through
and letting the model learn an optimal default split direction. The
reason: the missingness itself is an informative feature (a debutant's
missing `mean_finish_time_last5` is a real signal, not noise), and
median-fill collapses it into "this dog ran the average time."

The current Phase-1 code keeps median-filling because that's what the
already-trained model expects — XGBoost has a known footgun where a
model trained without NaN will silently route NaN to the right branch
at predict time, which was never optimised. Train and serve must use
the same NaN policy.

Phase 2 is the clean fix:

1. Add `impute_missing: bool` to `dataset_builder.build_dataset`.
   Default to `False` (NaN passthrough) for new experiments.
2. For sklearn-style trainers that don't natively handle NaN, do the
   imputation *inside* the trainer's `fit` and persist the imputer in
   the model artifact so predict can reuse it.
3. Use `SimpleImputer(add_indicator=True)` for the sklearn path so the
   missingness signal isn't lost.
4. Mirror the policy at predict time: if the artifact says "trained
   with NaN passthrough", skip the median fill; otherwise apply it.

Phase 2 requires a one-time retrain across every active experiment, so
it's gated on a follow-up PR rather than rolled into the loud-failure
fix.

### Key citations

* Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, sparsity-aware split finding — <https://arxiv.org/abs/1603.02754>
* LightGBM advanced topics, missing-value handling — <https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html>
* sklearn `SimpleImputer(add_indicator=True)` — <https://scikit-learn.org/stable/modules/impute.html>
* Benter (1994), annotated Hong Kong syndicate paper — <https://actamachina.com/posts/annotated-benter-paper>
* Baker & McHale (2017) on Kelly under probability uncertainty — <https://arxiv.org/pdf/1701.02814>

---

## 9. Quick-reference: where to look in the code

- **Scraper card vs results split:** `backend/scraping/gri_scraper.py:75-126`
  (results), `:432-495` (cards), `:528-615` (form pages).
- **DB writes:** `backend/scraping/db_pipeline.py:130-190`.
- **Visual feature dropna:** `backend/app/services/feature_engine.py:99-160`.
- **Built-in race-context features:** `backend/ml/race_features.py:48-160`.
- **Training-time NaN fill:** `backend/ml/dataset_builder.py:217-224`.
- **Training-set medians snapshot:** `backend/ml/dataset_builder.py:241`.
- **Predict-time NaN cascade:** `backend/app/services/prediction_service.py:385-440`.
- **SP toggle (defaults off):** `backend/ml/dataset_builder.py:37, 620-697`.
- **Odds-snapshot toggle (defaults on, table empty):**
  `backend/ml/dataset_builder.py:41, 700-839`;
  `backend/app/services/prediction_service.py:364, 413`.
- **Pace-shape derivation:** `backend/ml/dataset_builder.py:842+`.

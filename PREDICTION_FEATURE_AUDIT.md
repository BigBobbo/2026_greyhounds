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

## 6. Recommended fix plan (loud-failure first)

The user preference is to **fail loudly when prediction-time data is missing**
rather than to keep silently filling values. The smallest set of changes that
implements that:

### 6.1 Classify every feature as `pre_race` vs `post_race`

Add a column to `feature_definitions` (and the seed in
`backend/scripts/seed_features.py`):

```python
availability: Literal["pre_race", "post_race", "history_dependent"]
```

For built-in features, hard-code the same classification at the top of
`backend/ml/race_features.py`. The classification matches the table in §5.

### 6.2 Refuse to use post-race features for scheduled races

In `prediction_service.py` (around line 360, where the toggles are read):

```python
if any(e.Race.status == "scheduled" for e in entries):
    forbidden = {"current_sp_decimal", "current_sp_implied_prob", ...}
    used = set(trained_feature_names) & forbidden
    if used:
        raise PredictionDataError(
            f"Experiment {experiment.id} was trained with post-race features "
            f"{sorted(used)} that are not available for scheduled races. "
            f"Retrain with include_sp_features=False, or wait for results."
        )
```

Keep `include_sp_features=False` as the only supported value for any model
that will be served on upcoming races. Surface this in the Training Lab UI
as a warning, not just a docstring.

### 6.3 Stop silent imputation at predict time

Replace the cascade in `prediction_service.py:388-440` with:

```python
X = X.apply(pd.to_numeric, errors="coerce")

missing = X.columns[X.isna().any()]
if len(missing) > 0:
    # Per-feature breakdown so the error names the dogs and features
    sample = {col: X.index[X[col].isna()].tolist()[:3] for col in missing}
    raise PredictionDataError(
        f"Missing feature values for race {race_id}: {sample}. "
        f"Refusing to silently impute. "
        f"Run scraping for completed prior races, or exclude these features "
        f"from the experiment."
    )
```

Optionally keep a `strict=False` escape hatch for backtests on resulted races
where `dataset_builder.py:224` is the source of truth.

### 6.4 Drop the `feature_medians` plumbing for serve-time fills

Remove the `fillna(feature_medians)` call in `prediction_service.py:391-392`
and the alignment-time fill in `prediction_service.py:432-436`. Keep
`feature_medians` only for diagnostics. The training-time
`X.fillna(X.median()).fillna(0.0)` in `dataset_builder.py:224` should also be
narrowed to the columns we explicitly classify as "fill missing with median",
and every other column should be excluded from the dataset entirely if it has
NaN at training time.

### 6.5 Disable the empty-snapshot odds branch

Until the live odds-snapshot scraper is actually populating
`odds_snapshots` for both training races *and* upcoming races, set
`include_odds_snapshot_features=False` by default in
`build_dataset` (`dataset_builder.py:41`) and in the predict-time toggle
(`prediction_service.py:364`). This eliminates three latent NaN columns that
will become a real skew the moment partial data appears.

### 6.6 Don't read `entry.weight_kg` from the current race

`weight_change` (`race_features.py:87-89`) should compute against
`history["weight_kg"].iloc[-1]`, not `entry.weight_kg`. Today's weight is
never available, so feeding `entry.weight_kg` is structurally dead and just
causes the function to return `None`. Better behaviour: omit the feature
entirely, or rename it to `weight_trend_in_history` to make the intent clear.

### 6.7 Pre-flight check API endpoint

Add a small endpoint, e.g. `GET /api/predictions/preflight/{race_id}`, that
returns the per-feature availability for the upcoming race so the UI can show
the user which features are about to be missing **before** they hit predict.
This lets the user catch missing scrapes (e.g. a dog with no prior history,
or a feature that requires sectional times we haven't backfilled) without
having to read logs.

---

## 7. Quick-reference: where to look in the code

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

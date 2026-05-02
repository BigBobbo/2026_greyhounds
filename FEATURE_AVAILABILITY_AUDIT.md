# Feature Availability Audit — Training vs Upcoming-Race Prediction

## TL;DR

Predictions on upcoming race cards underperform CV holdouts because **training rows are silently filtered to `Race.status == "resulted"`** (`dataset_builder.py:98`), so every training row carries fields that are physically not on a card scrape: `finish_position`, `finish_time`, `sectional_time`, `adjusted_time`, `beaten_distance`, `weight_kg`, `starting_price` / `sp_decimal`, `comment`, and (often) `going`.

At predict time those columns are absent on the *current* `RaceEntry` and on the dog's *most recent* historical entry if it has not yet been resulted. Features that depend on them return `None`, which is then **silently replaced with the training-set median** (`prediction_service.py:391-394`). That median is computed over the post-race population, so an upcoming dog gets handed a plausible-looking historical average for a feature that should be undefined. The model can't tell the difference, the prediction degrades toward the field median, and trap-/ELO-derived signals dominate by default.

The user has asked for **loud failure** instead of silent imputation. The minimum fix is to: (a) record per-feature post-race dependency metadata, (b) refuse to predict (or predict in a clearly-degraded mode) when any required field is `NULL` on the input row, and (c) stop calling `fillna(median)` / `fillna(0)` on prediction inputs without explicit opt-in.

---

## 1. What the scraper writes for upcoming vs resulted races

### Card scrape (`status="scheduled"`)

`backend/scraping/gri_scraper.py:432-495` (`parse_card_page`) and `:528-615` (`parse_card_form_page`) only extract:

| Field | Source | Notes |
|---|---|---|
| `race_number`, `race_date`, `race_time` | card header | always present |
| `track_code`, `distance_m`, `grade`, `race_type` | card header | always present |
| `going` | — | **forced to `None`** (`gri_scraper.py:488`) |
| `prize_money` | — | **forced to `None`** (`gri_scraper.py:489`) |
| `status` | — | hardcoded `"scheduled"` (`gri_scraper.py:490`) |
| `trap`, `dog_name` | card row | always present |
| `trainer_name`, `owner_name`, `sire_name`, `dam_name`, `best_time` | per-race form page | **only if the form page is fetched** via `scrape_card_form` and merged via `merge_card_form_into_race` (`gri_scraper.py:618-660`); otherwise these are missing on first-time dogs |

The card-form page docstring is explicit (`gri_scraper.py:533`): *"the page does NOT carry an intended weight for the upcoming race (dogs are weighed-in on the night)"*.

### Result scrape (`status="resulted"`)

`parse_results_page` (`gri_scraper.py:75-296`) additionally writes per entry: `finish_position`, `finish_time`, `sectional_time` (where present), `beaten_distance`, `weight_kg`, `starting_price` / `sp_decimal`, `comment`, and the race-level `going` / `prize_money`.

### What `db_pipeline` does with them

`backend/scraping/db_pipeline.py:118` sets `Race.status="resulted"` only if at least one entry has a finish position. `upsert_race_entry` (`db_pipeline.py:174-189`) inserts `RaceEntry` with `finish_position`, `finish_time`, `sectional_time`, `beaten_distance`, `weight_kg`, `starting_price`, `sp_decimal`, `comment`, `grade_at_entry` all sourced from the scraped dict. For a card scrape every one of those is `None`. There is no second-pass on a resulted scrape that overwrites NULLs with results — *that* path is correctly handled (`db_pipeline.py:156-172`). The asymmetry is at the *time of prediction*: the scheduled row is what feature code reads.

---

## 2. The feature-availability delta

### Always available on a scheduled card (safe to use)

- Race-level: `track_id`, `race_date`, `distance_m`, `grade`, `race_type`, `race_number`, `num_runners`, `trap`
- Dog-level (from `Dog` table built up over history): `birth_date` (→ `dog_age_years`), `trainer_name`, `sire`, `dam`
- Anything derived from the dog's **prior** resulted history: ELO ratings, trainer-stats query (`race_features.py:300-406`), sire stats, trap-win-rates at track, mean/min finish time over last N resulted races, days-since-last (when computed from `dog_history.race_date.max()` rather than the `RaceEntry.days_since_last` cache column).

### Only available post-race (NOT on a card scrape)

These are NULL on the `RaceEntry` row representing the upcoming dog. Any feature that reads them off the *current* entry will silently degrade.

| Field on current `RaceEntry` | Used by |
|---|---|
| `weight_kg` | `_weight_change` (`race_features.py:87-89`, `:232-246`) |
| `sp_decimal`, `starting_price` | `_add_sp_features` (`dataset_builder.py:620-697`), `_add_odds_snapshot_features` (`dataset_builder.py:700-799`), Kelly staking, betting eval metadata |
| `finish_position`, `finish_time`, `sectional_time`, `adjusted_time`, `beaten_distance`, `comment` | not used directly on the *current* row by any feature, but their absence prevents the row from being included in *training* (see §3), which is the core leakage |
| Race-level `going` | `going_win_rate` (visual feature in `seed_features.py`) and any going-conditional model |

### Available pre-race in principle, but missing in practice on this scraper

- **`weight_kg`** — public weigh-ins happen ~30 min before off; not on the form page. Could be obtained from a separate live source if you add one.
- **`going`** — sometimes published on the morning card. Currently always NULL on cards (`gri_scraper.py:488`).
- **Live odds / `sp_decimal`** — would require a Betfair/SP feed *before* race time. The `OddsSnapshot` table exists but the scraper does not populate it for upcoming races (see `dataset_builder.py:705-710`: *"When the table is empty (i.e. the scraper is not yet collecting live odds snapshots) every snapshot-derived feature is left as NaN and the function is effectively a no-op — the median imputation downstream fills them with 0 without harming the model."* — this is the silent-fill behaviour).
- **`days_since_last`** — the `RaceEntry.days_since_last` column (`race_features.py:75`) is not populated on card scrapes. The fallback in `race_features.py:76-84` recomputes from history, which works.

---

## 3. Why training silently filters to "resulted" rows

`backend/ml/dataset_builder.py:85-101`:

```python
query = (
    db.query(...)
    .join(Race)
    .filter(Race.status == "resulted")
    .filter(RaceEntry.finish_position.isnot(None))
    .order_by(Race.race_date.desc())
)
```

`backend/ml/feature_store.py:217-224` — the bulk feature materializer also restricts to resulted races:

```python
entries_query = (
    db.query(RaceEntry.id)
    .join(Race)
    .filter(Race.status == "resulted")
)
```

`backend/app/services/feature_engine.py` — `get_dog_history()` filters to `Race.status == "resulted"`.

This is the right thing to do for *training labels* (you can't have a target without a result), but it has two side-effects that drive the train/serve gap:

1. The training matrix's `_add_sp_features` (`dataset_builder.py:191-193`, `:620-697`) reads `entries_df["sp_decimal"]` — which is non-null for every training row but null for every prediction row. **Default is `include_sp_features=False`**, but the toggle exists and the docstring at `:622-630` warns that any saved model relying on it cannot be served pre-race. Confirm the toggle is off in your active experiment.
2. Built-in features that read the *current* entry's post-race fields (e.g., `_weight_change` reading `entry.weight_kg`) get a real number in training and `None` at predict time.

---

## 4. The silent fill-in: where imputation hides the gap

There are four imputation sites; all of them happen without surfacing a warning per row.

### 4.1 Training-set median saved with the model

`backend/ml/dataset_builder.py:222-224`:

```python
# Fill remaining NaN with column median; columns where ALL values are NaN
# get median=NaN, so fill those with 0.0 as a safe default
X = X.fillna(X.median()).fillna(0.0)
```

`backend/ml/dataset_builder.py:241`:

```python
feature_medians = X.median().to_dict()
```

These medians are persisted with the model artifact and reused at predict time.

### 4.2 Prediction service applies the same medians, then zero-fills the rest

`backend/app/services/prediction_service.py:388-394`:

```python
X = X.apply(pd.to_numeric, errors="coerce")

# Fill NaN using training set medians (consistent with training)
if feature_medians:
    X = X.fillna(feature_medians)
# Any remaining NaN (new features, etc.) fill with 0
X = X.fillna(0)
```

### 4.3 Missing-column backfill at column alignment

`backend/app/services/prediction_service.py:432-436`:

```python
if trained_feature_names:
    for col in trained_feature_names:
        if col not in X.columns:
            X[col] = feature_medians.get(col, 0.0) if feature_medians else 0.0
    X = X[list(trained_feature_names)]
```

### 4.4 Final blanket `fillna(0)` on the aligned matrix

`backend/app/services/prediction_service.py:440`:

```python
X = X.fillna(0)
```

#### Net effect by feature family

| Feature family | What it reads on the current row | What predict sees on a card | What gets injected |
|---|---|---|---|
| `weight_change` | `RaceEntry.weight_kg` | `None` | training median (a typical historical weight delta) |
| `early_speed_ratio` | dog's prior `sectional_time/finish_time` | usually OK if dog has resulted history; `None` for debutants | training median |
| `_is_front_runner` | dog's prior `comment` text + sectionals | OK if history exists, else `0.0` (`race_features.py:268-274`) | hardcoded 0 → looks like "not a front runner" |
| `_add_sp_features` (10 cols) | `entries_df["sp_decimal"]` of the current race | all `None` → entire branch returns early at `dataset_builder.py:649-651` | columns absent → backfilled with median per `prediction_service.py:435` |
| `_add_odds_snapshot_features` (3 cols) | `OddsSnapshot` rows for current race | empty when scraper does not collect snapshots | zero-fill (`prediction_service.py:440`) |
| `_add_pace_shape_features` (~13 cols) | `is_front_runner` aggregations within race | all entries have `is_front_runner = 0.0` (debutant default) | constants per race; no in-race variance |
| Race-relative `__vs_field` / `__rank` | base feature varies | base feature is constant after median fill → vs_field collapses to 0, rank collapses to ties | uniform values across the field |
| Built-in trainer/sire/track-rating queries | DB queries over resulted history | usually OK; min-sample thresholds (e.g., `min_races=30` at `race_features.py:169`) return `None` | training median |

Once the SP / pace-shape / weight-change / odds-snapshot features collapse to a per-race constant, the only intra-race variance left for the model to act on is trap number, ELO, and prior-finish-time stats. That matches the symptom of upcoming-race predictions tracking trap order more than they should.

---

## 5. Suspected leakage in addition to the train/serve gap

These do not affect serve-time correctness directly but they *inflate* the CV holdout numbers, widening the apparent gap:

- **`include_sp_features`** — even if off by default, any experiment with it on bakes the realised SP into both train *and* CV folds. SP is realised on results day, so CV folds within the same era leak market knowledge. Confirm `experiment.split_config.include_sp_features` is `False` for the deployed model.
- **Comment-derived rolling windows** — `quick_away_rate_last10`, `led_at_bend1_rate_last10`, `trouble_rate_last10`, `is_front_runner` (`race_comments.py`, `race_features.py:268`). These are only legitimate if computed strictly from races *before* the target race. The current code uses `dog_history` as supplied; spot-check `feature_engine.get_dog_history()` to confirm it filters by `race_date < race_date_of_target`.
- **`sire_progeny_*`, `trainer_*` aggregations** (`race_features.py:300-406`) — confirmed to filter `Race.status == "resulted"` and (typically) by date. Verify each call site passes `before_date`. Spot-checked at `race_features.py:331`, `:376`; recommend an integration test asserting the date guard.

---

## 6. Recommended fix — fail loudly

The user has asked for explicit failure instead of imputation. The smallest path that delivers that without rewriting the feature store:

### 6.1 Tag every feature with its post-race dependencies

Add to `FeatureDefinition` (and to the built-in feature registry in `ml/race_features.py`) a metadata field:

```python
post_race_dependencies: list[Literal[
    "current_finish_position", "current_finish_time", "current_sectional_time",
    "current_adjusted_time", "current_beaten_distance", "current_weight_kg",
    "current_sp_decimal", "current_comment", "current_going",
]] = []
```

Examples:
- `weight_change` → `["current_weight_kg"]`
- `current_sp_decimal`, `sp_rank_in_field`, `market_overround`, … → `["current_sp_decimal"]`
- `early_speed_ratio` → `[]` (reads only history)
- `_is_front_runner` → `[]` (reads only history; debutants legitimately get 0.0)

### 6.2 Make the predict path refuse to fill silently

Replace `prediction_service.py:388-394` with strict-mode logic, gated by an experiment flag (default *strict*):

```python
strict_features = experiment.split_config.get("strict_feature_availability", True)

X = X.apply(pd.to_numeric, errors="coerce")

if strict_features:
    nan_cells = X.isna()
    if nan_cells.any().any():
        missing = (
            nan_cells.stack()[lambda s: s]
            .index.tolist()
        )
        raise FeatureUnavailableError(
            f"Refusing to predict on {len(missing)} (entry, feature) cells with NaN. "
            f"First 10: {missing[:10]}. "
            f"Either populate the missing source columns "
            f"(weight_kg, sp_decimal, comment, going, …) or set "
            f"experiment.split_config['strict_feature_availability'] = False "
            f"to opt back into median imputation."
        )
else:
    if feature_medians:
        X = X.fillna(feature_medians)
    X = X.fillna(0)
```

Apply the same treatment at the column-alignment site (`prediction_service.py:432-436`): in strict mode, raise if a trained column is missing instead of injecting `feature_medians.get(col, 0.0)`.

### 6.3 Validate at the entry level before computing any features

Before calling `compute_features_for_entries`, walk the active feature set, union their `post_race_dependencies`, and check the upcoming `RaceEntry` and `Race` rows. Raise a structured error early so the predictions endpoint can return HTTP 422 with the list of missing fields per dog rather than producing scores that look real.

### 6.4 Stop training on features the live data cannot provide

Add a CI check (or a `_assert_servable_feature_set` step in `build_dataset`) that fails the training run when:
- `include_sp_features = True` and there is no live odds feed configured
- `include_odds_snapshot_features = True` and `OddsSnapshot` is empty for the most recent N days
- any feature in the chosen set has `post_race_dependencies` that are not in the scheduled-card column list

This prevents the asymmetric matrix from being saved to a model artifact in the first place.

### 6.5 Surface data-completeness to the API consumer

`ComputedFeature.data_complete` already exists (`alembic/versions/c3f8a1d42b7e_add_data_complete_to_computed_features.py`). It is used during materialization but not surfaced to the prediction endpoint. Return it on `/predictions` so a downstream consumer can suppress or flag dogs whose features were computed against incomplete data.

### 6.6 Backfill the missing pre-race signals (longer term)

If you want SP-based features at predict time without leakage, ingest a live Betfair / Oddschecker pre-off price into `OddsSnapshot` and freeze a snapshot at T-5min. Until that exists, drop SP and odds-snapshot features from the deployed model entirely.

---

## 7. Quick-reference cheat sheet

**Files to change for the loud-failure mode:**
- `backend/app/services/prediction_service.py:388-394` — replace silent fillna with strict check
- `backend/app/services/prediction_service.py:432-440` — same, for the column-alignment step
- `backend/app/models/features.py` — add `post_race_dependencies` column to `FeatureDefinition` (+ Alembic migration)
- `backend/ml/race_features.py` — annotate each built-in feature
- `backend/scripts/seed_features.py` — annotate visual/code features
- `backend/ml/dataset_builder.py:85-101` — emit a warning when `include_sp_features=True` and the deployed runtime cannot provide SP

**Smallest test that would have caught this:**
A `test_predict_on_scheduled_race_uses_no_postrace_data.py` that:
1. inserts a `Race` with `status="scheduled"` and three `RaceEntry` rows with all post-race fields `NULL`,
2. computes features and predicts,
3. asserts that no feature in the returned matrix was filled from `feature_medians` (i.e., assert `X.notna().all().all()` *before* the imputation step).

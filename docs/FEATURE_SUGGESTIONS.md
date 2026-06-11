# Feature Engineering Suggestions for Greyhound Predictor

This document proposes new features for the greyhound race prediction model, organized by category, priority, and implementation approach. Each feature includes rationale, implementation details, and validation steps.

## How to Use This Document

Features are grouped into categories and tiered by expected predictive impact. Each feature specifies:
- **Implementation type**: `visual` (JSON config added to `seed_features.py`), `code` (sandboxed Python added to `seed_features.py`), `built-in` (added to `ml/race_features.py`), or `engine` (requires changes to `feature_engine.py` or `dataset_builder.py`)
- **Files to modify**: Exact file paths
- **Implementation details**: Code logic or config JSON
- **Validation steps**: How to confirm correctness

---

## Existing Feature Inventory (for reference)

Before implementing, know what already exists:

**Visual/code features** (in `backend/scripts/seed_features.py` — 40 features):
- Time: `mean_finish_time_last5`, `min_finish_time_last10`, `mean_finish_time_last5_same_dist`, `stdev_finish_time_last5`, `finish_time_trend_last5`
- Going-adjusted time: `mean_adjusted_time_last5`, `mean_adjusted_time_last5_same_dist`, `best_adjusted_time_last10`, `best_adjusted_time_last10_same_dist`
- EWM (recency-weighted): `ewm_finish_time_last10`, `ewm_position_last10`, `ewm_adjusted_time_last10`
- Position: `mean_position_last5`, `win_rate_last10`, `place_rate_last10`, `win_rate_same_track`, `win_rate_same_trap`
- Sectional: `mean_sectional_last5`
- Weight: `mean_weight_last5`
- Beaten distance: `mean_beaten_dist_last5`
- SP: `mean_sp_last5`
- Experience: `career_runs`, `runs_at_track`, `runs_at_distance`
- Code features: `days_since_last_race`, `days_since_last_win`, `improving_form`, `track_distance_affinity`, `trap_win_rate_at_track`, `trap_place_rate_at_track`, `races_in_last_30_days`, `grade_change`, `going_win_rate`, `finish_position_stdev_last5`
- Trouble-in-running: `trouble_rate_last10`, `first_bend_trouble_rate`
- Rest/fitness: `optimal_rest_window`, `rest_category`
- Bayesian-smoothed: `bayesian_win_rate`, `bayesian_place_rate`

**Built-in features** (in `backend/ml/race_features.py` — 21 features):
- `trap_win_rate_at_track`, `grade_movement`, `days_since_last`, `weight_change`, `early_speed_ratio`, `is_front_runner`, `career_races`, `position_consistency`
- `dog_age_years`, `dog_age_squared`
- `early_speed_x_trap`, `early_speed_x_inside`, `early_speed_x_outside`, `front_runner_x_inside`, `front_runner_x_outside`
- `trainer_win_rate`, `trainer_place_rate`, `trainer_win_rate_at_track`
- `sire_progeny_win_rate`, `sire_progeny_mean_time_at_dist`
- `track_speed_rating`

**SP-derived features** (in `backend/ml/dataset_builder.py`):
- `current_sp_decimal`, `current_sp_implied_prob`, `sp_rank_in_field`, `market_overround`

**Pace shape features** (in `backend/ml/dataset_builder.py`):
- `num_front_runners_in_race`, `is_sole_front_runner`, `pace_pressure`, `early_speed_rank`, `is_predicted_leader`

**Race-relative features** (in `backend/ml/dataset_builder.py`):
- `__vs_field` (value minus field mean) and `__rank` (rank within race) variants of key features

---

## TIER 1: High-Impact, Low-to-Medium Effort [COMPLETED]

These features have been implemented. See PR #11.

---

### 1.1 Going-Adjusted Time Features

**Rationale**: Raw finish times are misleading without going adjustments. A 29.50s on "slow" going is a better performance than 29.50s on "fast" going. The `adjusted_time` field already exists on `RaceEntry` but is never used in features. This is low-hanging fruit.

**Implementation type**: `visual` (requires adding `adjusted_time` as a supported metric)

**Files to modify**:
- `backend/app/services/feature_engine.py` — Add `"adjusted_time"` to the supported metrics. In `get_dog_history()` (line 47), `RaceEntry.adjusted_time` must be included in the query. It is currently NOT queried. Add it after `RaceEntry.sectional_time` (line 52), and add `"adjusted_time"` to the `columns` list (line 83).
- `backend/scripts/seed_features.py` — Add new feature definitions

**New features to seed**:

```python
{
    "name": "mean_adjusted_time_last5",
    "display_name": "Mean Adjusted Time (last 5)",
    "description": "Average going-adjusted finish time over last 5 races. Normalizes for track conditions.",
    "feature_type": "visual",
    "config_json": {
        "metric": "adjusted_time",
        "aggregation": "mean",
        "window": {"type": "last_n", "n": 5},
        "filters": {}
    },
    "input_columns": ["adjusted_time"],
},
{
    "name": "mean_adjusted_time_last5_same_dist",
    "display_name": "Mean Adjusted Time (last 5, same distance)",
    "description": "Average going-adjusted finish time over last 5 races at same distance",
    "feature_type": "visual",
    "config_json": {
        "metric": "adjusted_time",
        "aggregation": "mean",
        "window": {"type": "last_n", "n": 5},
        "filters": {"same_distance": true}
    },
    "input_columns": ["adjusted_time"],
},
{
    "name": "best_adjusted_time_last10",
    "display_name": "Best Adjusted Time (last 10)",
    "description": "Fastest going-adjusted time in last 10 races — the dog's ceiling performance",
    "feature_type": "visual",
    "config_json": {
        "metric": "adjusted_time",
        "aggregation": "min",
        "window": {"type": "last_n", "n": 10},
        "filters": {}
    },
    "input_columns": ["adjusted_time"],
},
{
    "name": "best_adjusted_time_last10_same_dist",
    "display_name": "Best Adjusted Time (last 10, same distance)",
    "description": "Fastest going-adjusted time in last 10 races at the same distance",
    "feature_type": "visual",
    "config_json": {
        "metric": "adjusted_time",
        "aggregation": "min",
        "window": {"type": "last_n", "n": 10},
        "filters": {"same_distance": true}
    },
    "input_columns": ["adjusted_time"],
},
```

**Validation**:
1. After modifying `feature_engine.py`, write a test that calls `get_dog_history()` for a dog with known races and confirms `adjusted_time` is present in the returned DataFrame.
2. Pick a dog with races on different going conditions. Manually compute `finish_time - going_allowance` for their last 5 races. Verify `mean_adjusted_time_last5` matches your manual calculation.
3. Verify the feature is `None` for dogs with no `adjusted_time` data (rather than erroring).
4. Run `pytest backend/` — no existing tests should break.

---

### 1.2 Current SP as Model Feature

**Rationale**: Academic consensus (Benter 1994, Bolton & Chapman) is that starting price is the single strongest predictor of race outcomes. The public odds encode enormous information. Currently `sp_decimal` is only used in dataset_builder for betting evaluation (`meta` columns), not as an input feature.

**Implementation type**: `built-in` (add to `dataset_builder.py`)

**Files to modify**:
- `backend/ml/dataset_builder.py` — In the section where built-in features are added to the dataframe, also add `sp_decimal` and `sp_implied_prob` as feature columns sourced from the race entry data.

**Implementation details**:

In `dataset_builder.py`, in the function that assembles the feature matrix, after the built-in race-context features are added, add these columns:

```python
# SP-derived features
if "sp_decimal" in df.columns:
    features_df["current_sp_decimal"] = df["sp_decimal"]
    features_df["current_sp_implied_prob"] = 1.0 / df["sp_decimal"]
    # SP rank within the race (1 = favourite)
    features_df["sp_rank_in_field"] = df.groupby("race_id")["sp_decimal"].rank(method="min")
```

Also add:

```python
# Market overround per race (sum of implied probs — higher = more uncertain market)
race_overround = df.groupby("race_id")["sp_decimal"].transform(lambda x: (1.0 / x).sum())
features_df["market_overround"] = race_overround

# SP relative to dog's career mean SP
features_df["sp_vs_career_mean"] = df["sp_decimal"] / df.groupby("dog_id")["sp_decimal"].transform("mean")
```

**Important caveat**: These features use information from the current race, not historical data. This is valid because SP is known before the race starts. However, if you want to predict races *before* the market forms, you'll need to exclude these features. Consider making them togglable via a flag like `include_sp_features=True`.

**Validation**:
1. Build a dataset with and without SP features. Confirm the SP-included dataset has the extra columns.
2. For a specific race, manually look up the SP of all runners. Verify `sp_rank_in_field` correctly assigns rank 1 to the shortest-priced dog.
3. Verify `market_overround` is the same for all entries in the same race and is > 1.0.
4. Train two identical experiments: one with SP features, one without. The SP-included model should have notably better accuracy (expect 3-8% improvement in top-1 accuracy).
5. Check that `sp_vs_career_mean` handles dogs with only one prior race (career mean = that one race's SP).

---

### 1.3 Dog Age

**Rationale**: Greyhounds have a well-documented performance curve — they peak between ages 2-3.5 years, then decline. The `Dog.birth_date` field exists but is never used in features.

**Implementation type**: `built-in` (add to `ml/race_features.py`)

**Files to modify**:
- `backend/ml/race_features.py` — Add `_dog_age` helper and include in `compute_race_context_features`

**Implementation details**:

```python
def _dog_age(birth_date, race_date) -> float | None:
    """Dog's age in years at the time of the race."""
    if birth_date is None or race_date is None:
        return None
    delta = race_date - birth_date
    return delta.days / 365.25
```

In `compute_race_context_features`, query the dog's birth_date and add:

```python
# Dog age
dog = db.query(Dog).filter(Dog.id == dog_id).first()
if dog and dog.birth_date:
    age = _dog_age(dog.birth_date, race_context.get("race_date"))
    features["dog_age_years"] = age
    features["dog_age_squared"] = age ** 2 if age else None  # captures non-linear decline
else:
    features["dog_age_years"] = None
    features["dog_age_squared"] = None
```

You'll need to import `Dog` from `app.models.dog` at the top of `race_features.py` and pass `dog_id` into `compute_race_context_features` (check if it's already available via the entry/race_context).

**Validation**:
1. Pick a dog with a known birth_date. Manually compute their age at a specific race date. Verify the feature matches.
2. Check that `dog_age_squared` = `dog_age_years ** 2`.
3. Verify the feature is `None` for dogs with no birth_date (not an error).
4. After training, check SHAP values — age should show a non-linear pattern (positive contribution around 2-3 years, negative for older dogs).

---

### 1.4 Pace/Trap Interaction Features

**Rationale**: The interaction between a dog's early speed and their trap draw is the most under-modelled dynamic in greyhound racing. A fast-breaking dog from trap 1 has a fundamentally different race profile than from trap 6. Academic studies show this interaction explains 3-5% of additional variance beyond the individual features.

**Implementation type**: `built-in` (add to `ml/race_features.py`)

**Files to modify**:
- `backend/ml/race_features.py` — Add to `compute_race_context_features`

**Implementation details**:

```python
# Pace-trap interactions
trap = race_context.get("trap")
early_speed = features.get("early_speed_ratio")
front_runner = features.get("is_front_runner")

if trap is not None and early_speed is not None:
    features["early_speed_x_trap"] = early_speed * trap
    features["early_speed_x_inside"] = early_speed * (1.0 if trap <= 2 else 0.0)
    features["early_speed_x_outside"] = early_speed * (1.0 if trap >= 5 else 0.0)

if trap is not None and front_runner is not None:
    features["front_runner_x_inside"] = front_runner * (1.0 if trap <= 2 else 0.0)
    features["front_runner_x_outside"] = front_runner * (1.0 if trap >= 5 else 0.0)
```

Add these AFTER `early_speed_ratio` and `is_front_runner` are computed (since they depend on those values).

**Validation**:
1. For a known front-runner (high `is_front_runner`) drawn in trap 1: verify `front_runner_x_inside` is high and `front_runner_x_outside` is 0.
2. For the same dog in trap 6: verify `front_runner_x_inside` is 0 and `front_runner_x_outside` is high.
3. Train a model including these features. Check feature importance — `early_speed_x_inside` should rank in the top 15 features.
4. Verify no NaN propagation when `early_speed_ratio` is None (the feature should be None, not NaN).

---

### 1.5 Early Speed Rank in Field

**Rationale**: The absolute early speed matters less than whether a dog is the fastest breaker *in this specific race*. Research shows the predicted first-bend leader wins ~30-35% of the time (vs 17% base rate in 6-runner races). This is a race-context feature that requires seeing all runners.

**Implementation type**: `engine` (add to `dataset_builder.py` in `add_race_relative_features`)

**Files to modify**:
- `backend/ml/dataset_builder.py` — Extend the `add_race_relative_features` function

**Implementation details**:

In the `add_race_relative_features` function (or equivalent section that has access to all entries in a race):

```python
# Early speed rank within race (1 = predicted fastest to first bend)
if "early_speed_ratio" in df.columns:
    # Lower early_speed_ratio = faster to first bend
    df["early_speed_rank"] = df.groupby("race_id")["early_speed_ratio"].rank(method="min")
    # Binary: is this dog the predicted leader?
    df["is_predicted_leader"] = (df["early_speed_rank"] == 1).astype(float)
```

**Validation**:
1. For a specific race, manually check each dog's `early_speed_ratio`. The dog with the lowest value should have `early_speed_rank = 1` and `is_predicted_leader = 1.0`.
2. Verify ties are handled correctly (both get rank 1 with `method="min"`).
3. Verify NaN handling — dogs with no sectional history should get NaN rank, not a default.
4. Compute the actual win rate of `is_predicted_leader == 1` dogs across historical data. It should be significantly above 17% (the 1/6 base rate).

---

### 1.6 Pace Shape / Front-Runner Count

**Rationale**: When multiple front-runners are in the same race, they interfere with each other at the first bend, creating opportunities for closers. A sole front-runner wins at a much higher rate. This race-context feature encodes the most important tactical dynamic.

**Implementation type**: `engine` (add to `dataset_builder.py`)

**Files to modify**:
- `backend/ml/dataset_builder.py`

**Implementation details**:

```python
# Pace shape features (require all runners in race)
if "is_front_runner" in df.columns:
    # Count front-runners per race (is_front_runner > 0.5)
    front_runner_count = df.groupby("race_id")["is_front_runner"].transform(
        lambda x: (x > 0.5).sum()
    )
    df["num_front_runners_in_race"] = front_runner_count
    
    # Is this dog a sole front-runner? (huge advantage)
    df["is_sole_front_runner"] = (
        (df["is_front_runner"] > 0.5) & (front_runner_count == 1)
    ).astype(float)
    
    # Pace pressure: interaction of being a front-runner with number of rivals
    df["pace_pressure"] = df["is_front_runner"] * front_runner_count
```

**Validation**:
1. Find a race where exactly one dog has `is_front_runner > 0.5`. Verify `is_sole_front_runner = 1.0` for that dog and `0.0` for all others.
2. Find a race with 3 front-runners. Verify `num_front_runners_in_race = 3` for all entries.
3. Compute historical win rate for `is_sole_front_runner == 1` dogs. It should be substantially higher than the win rate for front-runners in multi-front-runner races.
4. Verify `pace_pressure` is 0 for non-front-runners regardless of how many front-runners exist.

---

### 1.7 Exponentially Weighted Mean (Time-Decay) Aggregation

**Rationale**: Equal-weighting the last N races ignores that the most recent race is far more informative than a race from 3 months ago. Exponential decay weighting consistently improves prediction by 1-3% in racing Kaggle competitions.

**Implementation type**: `engine` (add `"ewm"` aggregation to `feature_engine.py`)

**Files to modify**:
- `backend/app/services/feature_engine.py` — Add `"ewm"` to the aggregation options in `_aggregate()`

**Implementation details**:

In `feature_engine.py`, find the `_aggregate` function and add a new case:

```python
elif aggregation == "ewm":
    # Exponential weighted mean: most recent race has highest weight
    # alpha=0.5 means each older race gets half the weight of the next newer one
    # Series must be in chronological order (oldest first, which it is after sort)
    if len(series) < 2:
        return float(series.iloc[0])
    return float(series.ewm(alpha=0.5, adjust=True).mean().iloc[-1])
```

Also update the docstring/comments at the top of the file to list `"ewm"` as a valid aggregation.

Then seed new features:

```python
{
    "name": "ewm_finish_time_last10",
    "display_name": "Exp. Weighted Mean Finish Time (last 10)",
    "description": "Exponentially weighted mean finish time — recent races weighted more heavily. Alpha=0.5.",
    "feature_type": "visual",
    "config_json": {
        "metric": "finish_time",
        "aggregation": "ewm",
        "window": {"type": "last_n", "n": 10},
        "filters": {}
    },
    "input_columns": ["finish_time"],
},
{
    "name": "ewm_position_last10",
    "display_name": "Exp. Weighted Mean Position (last 10)",
    "description": "Exponentially weighted mean finish position — recent races weighted more heavily",
    "feature_type": "visual",
    "config_json": {
        "metric": "finish_position",
        "aggregation": "ewm",
        "window": {"type": "last_n", "n": 10},
        "filters": {}
    },
    "input_columns": ["finish_position"],
},
{
    "name": "ewm_adjusted_time_last10",
    "display_name": "Exp. Weighted Mean Adjusted Time (last 10)",
    "description": "Exponentially weighted mean going-adjusted time — combines recency with going normalization",
    "feature_type": "visual",
    "config_json": {
        "metric": "adjusted_time",
        "aggregation": "ewm",
        "window": {"type": "last_n", "n": 10},
        "filters": {}
    },
    "input_columns": ["adjusted_time"],
},
```

**Validation**:
1. Create a dog history with 5 known finish times. Manually compute the ewm with alpha=0.5 using the pandas formula. Verify the feature matches.
2. Verify that `ewm` with a single data point returns that value (not NaN).
3. Compare `ewm_finish_time_last10` vs `mean_finish_time_last5` for the same dogs. The ewm should differ most for dogs whose recent form diverges from their older form.
4. Train two models: one using `mean_*` features, one replacing them with `ewm_*` features. Compare accuracy — ewm should perform at least as well, likely better.

---

## TIER 2: Strong Evidence, Moderate Effort [COMPLETED]

Implemented: 2.1 (trainer stats), 2.2 (sire/dam), 2.3 (trouble-in-running), 2.4 (track speed rating), 2.6 (rest windows), 2.7 (Bayesian rates).
Deferred: 2.5 (SP drift) — requires `odds_snapshots` table to be populated by odds scraping.

---

### 2.1 Trainer Performance Features

**Rationale**: Trainer strike rate is a significant predictor in both greyhound and horse racing. Some trainers consistently achieve 20%+ win rates while others hover at 10%. The `Dog.trainer_name` field is available but unused.

**Implementation type**: `built-in` (add to `ml/race_features.py`)

**Files to modify**:
- `backend/ml/race_features.py`

**Implementation details**:

```python
def _trainer_stats(
    db: Session,
    trainer_name: str | None,
    track_id: int | None,
    before_date=None,
    days_window: int = 90,
    min_runners: int = 20,
) -> dict[str, float | None]:
    """Trainer win/place rate overall and at this track."""
    result = {"trainer_win_rate": None, "trainer_place_rate": None, 
              "trainer_win_rate_at_track": None}
    
    if not trainer_name:
        return result
    
    from datetime import timedelta
    cutoff = before_date - timedelta(days=days_window) if before_date else None
    
    # Overall trainer stats
    query = (
        db.query(
            func.count(RaceEntry.id).label("total"),
            func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
            func.sum(case((RaceEntry.finish_position <= 3, 1), else_=0)).label("places"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .filter(
            Dog.trainer_name == trainer_name,
            Race.status == "resulted",
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date:
        query = query.filter(Race.race_date < before_date)
    if cutoff:
        query = query.filter(Race.race_date >= cutoff)
    
    row = query.first()
    if row and row.total and row.total >= min_runners:
        result["trainer_win_rate"] = float(row.wins) / float(row.total)
        result["trainer_place_rate"] = float(row.places) / float(row.total)
    
    # Trainer at this track
    if track_id:
        track_query = query.filter(Race.track_id == track_id)
        track_row = track_query.first()
        if track_row and track_row.total and track_row.total >= 10:
            result["trainer_win_rate_at_track"] = float(track_row.wins) / float(track_row.total)
    
    return result
```

In `compute_race_context_features`, add:

```python
# Trainer stats
dog = db.query(Dog).filter(Dog.id == dog_id).first()
trainer_stats = _trainer_stats(
    db, dog.trainer_name if dog else None,
    race_context.get("track_id"), race_context.get("race_date")
)
features.update(trainer_stats)
```

**Validation**:
1. Pick a known trainer. Manually count their wins and total runners in the last 90 days. Verify `trainer_win_rate` matches `wins / total`.
2. Verify `trainer_win_rate_at_track` is `None` when the trainer has fewer than 10 runners at that track (not zero, but None).
3. Verify the `before_date` filter prevents data leakage — only races before the target race are counted.
4. After training, check if `trainer_win_rate` appears in the top 20 features by importance.

---

### 2.2 Sire/Dam Progeny Features

**Rationale**: Breeding is a major factor in greyhound racing. Certain sires produce progeny that excel at specific distances or have faster early speed. The `Dog.sire` and `Dog.dam` fields exist but are unused.

**Implementation type**: `built-in` (add to `ml/race_features.py`)

**Files to modify**:
- `backend/ml/race_features.py`

**Implementation details**:

```python
def _sire_stats(
    db: Session,
    sire_name: str | None,
    distance_m: int | None,
    before_date=None,
    min_progeny_runs: int = 50,
) -> dict[str, float | None]:
    """Win rate and mean time of sire's progeny."""
    result = {"sire_progeny_win_rate": None, "sire_progeny_mean_time_at_dist": None}
    
    if not sire_name:
        return result
    
    # Win rate of all progeny of this sire
    query = (
        db.query(
            func.count(RaceEntry.id).label("total"),
            func.sum(case((RaceEntry.finish_position == 1, 1), else_=0)).label("wins"),
        )
        .join(Dog, RaceEntry.dog_id == Dog.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Dog.sire == sire_name,
            Race.status == "resulted",
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date:
        query = query.filter(Race.race_date < before_date)
    
    row = query.first()
    if row and row.total and row.total >= min_progeny_runs:
        result["sire_progeny_win_rate"] = float(row.wins) / float(row.total)
    
    # Mean time of progeny at this distance
    if distance_m:
        time_query = (
            db.query(func.avg(RaceEntry.finish_time))
            .join(Dog, RaceEntry.dog_id == Dog.id)
            .join(Race, RaceEntry.race_id == Race.id)
            .filter(
                Dog.sire == sire_name,
                Race.status == "resulted",
                Race.distance_m == distance_m,
                RaceEntry.finish_time.isnot(None),
            )
        )
        if before_date:
            time_query = time_query.filter(Race.race_date < before_date)
        
        avg_time = time_query.scalar()
        if avg_time:
            result["sire_progeny_mean_time_at_dist"] = float(avg_time)
    
    return result
```

**Validation**:
1. Pick a sire with many progeny (e.g., a popular stud dog). Manually count wins / total runs. Verify.
2. Verify the `min_progeny_runs` threshold prevents noisy results for rare sires.
3. Verify `before_date` prevents leakage.
4. Check that `sire_progeny_mean_time_at_dist` varies by distance — a sprint sire's progeny should have faster 480m times but possibly slower 750m times.

---

### 2.3 Trouble-in-Running Features (Comment Parsing)

**Rationale**: The `comment` field contains rich information about race incidents. Dogs that encountered trouble (checked, bumped, crowded) may have been unlucky — their raw form underrates them. Professional form analysts treat "hard luck stories" as key indicators.

**Implementation type**: `code` (sandboxed Python in `seed_features.py`)

**Files to modify**:
- `backend/scripts/seed_features.py`

**New features to seed**:

```python
{
    "name": "trouble_rate_last10",
    "display_name": "Trouble in Running Rate (last 10)",
    "description": "Fraction of last 10 races where the dog encountered trouble (checked, bumped, crowded, fell, hampered). Higher = more unlucky.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    recent = dog_history.tail(10)
    comments = recent["comment"].dropna().str.lower()
    if comments.empty:
        return 0.0
    trouble_keywords = ["ck", "bmp", "crd", "fell", "hampered", "baulked", "stumbled", "crowded", "checked", "bumped"]
    trouble_count = comments.apply(lambda c: 1 if any(kw in c for kw in trouble_keywords) else 0).sum()
    return float(trouble_count) / float(len(recent))
""",
    "input_columns": ["comment"],
},
{
    "name": "first_bend_trouble_rate",
    "display_name": "First Bend Trouble Rate (last 10)",
    "description": "Fraction of last 10 races with trouble specifically at the first bend",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    recent = dog_history.tail(10)
    comments = recent["comment"].dropna().str.lower()
    if comments.empty:
        return 0.0
    first_bend_patterns = ["ck 1", "bmp 1", "crd 1", "crowded 1", "checked 1", "bumped 1"]
    trouble_count = comments.apply(lambda c: 1 if any(p in c for p in first_bend_patterns) else 0).sum()
    return float(trouble_count) / float(len(recent))
""",
    "input_columns": ["comment"],
},
```

**Validation**:
1. Find a dog whose race comments include "ck" or "bmp" keywords. Manually count occurrences in last 10 races. Verify the rate matches.
2. Find a dog with clean comments (no trouble keywords). Verify `trouble_rate_last10 = 0.0`.
3. Test the sandbox execution by calling the feature sandbox's `validate_and_run` function with test data.
4. Cross-reference: dogs with high trouble rates from inside traps (1-2) should have higher future performance improvements when drawn wider.

---

### 2.4 Track-Normalized Speed Rating

**Rationale**: Different tracks produce different standard times for the same distance due to track geometry, surface, and altitude. A 29.30s at Shelbourne is a different performance level than 29.30s at Mullingar. Normalizing times to a track standard makes cross-track comparisons meaningful.

**Implementation type**: `built-in` (add to `ml/race_features.py`)

**Files to modify**:
- `backend/ml/race_features.py`

**Implementation details**:

```python
def _track_speed_rating(
    db: Session,
    dog_adjusted_time: float | None,
    track_id: int | None,
    distance_m: int | None,
    before_date=None,
    days_window: int = 180,
    min_races: int = 50,
) -> float | None:
    """Dog's time relative to the track/distance standard (median).
    
    Negative = faster than standard (good). Positive = slower.
    """
    if dog_adjusted_time is None or not track_id or not distance_m:
        return None
    
    from datetime import timedelta
    cutoff = before_date - timedelta(days=days_window) if before_date else None
    
    query = (
        db.query(func.count(RaceEntry.id), func.avg(RaceEntry.adjusted_time))
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            Race.track_id == track_id,
            Race.distance_m == distance_m,
            Race.status == "resulted",
            RaceEntry.adjusted_time.isnot(None),
        )
    )
    if before_date:
        query = query.filter(Race.race_date < before_date)
    if cutoff:
        query = query.filter(Race.race_date >= cutoff)
    
    count, avg_time = query.first()
    if not count or count < min_races or avg_time is None:
        return None
    
    return float(dog_adjusted_time - avg_time)
```

This feature requires the dog's best or mean adjusted time at this distance, computed from their history. In `compute_race_context_features`, use the dog's `best_adjusted_time_last10` or compute it inline from `dog_history`.

**Validation**:
1. Pick two dogs with similar raw times at different tracks. The speed rating should differentiate them.
2. Verify the track standard changes over time (the `days_window=180` ensures it reflects recent conditions).
3. Verify negative values = faster than standard, positive = slower.
4. Check that the feature is `None` when the track/distance combo has fewer than 50 races in the window.

---

### 2.5 SP Drift (Odds Movement)

**Rationale**: The direction of odds movement is highly predictive. When a dog's price shortens (money coming in), it signals informed bettors backing that dog. This was highlighted in multiple Kaggle horse racing competition solutions. The `odds_snapshots` table exists but is not used for features.

**Implementation type**: `built-in` (add to `ml/race_features.py` or `dataset_builder.py`)

**Files to modify**:
- `backend/ml/race_features.py`
- May need to import `OddsSnapshot` from `app.models.odds`

**Implementation details**:

```python
def _sp_drift(
    db: Session,
    race_id: int | None,
    dog_id: int | None,
) -> dict[str, float | None]:
    """Compute odds drift from earliest to latest snapshot.
    
    Positive drift = price shortened (backed, good sign).
    """
    result = {"sp_drift": None, "sp_drift_pct": None}
    
    if not race_id or not dog_id:
        return result
    
    from app.models.odds import OddsSnapshot
    
    snapshots = (
        db.query(OddsSnapshot.odds_decimal, OddsSnapshot.scraped_at)
        .filter(
            OddsSnapshot.race_id == race_id,
            OddsSnapshot.dog_id == dog_id,
        )
        .order_by(OddsSnapshot.scraped_at)
        .all()
    )
    
    if len(snapshots) < 2:
        return result
    
    early_odds = snapshots[0].odds_decimal
    late_odds = snapshots[-1].odds_decimal
    
    if early_odds and late_odds and early_odds > 0:
        # Positive = price shortened (drift in), negative = price drifted out
        result["sp_drift"] = float(early_odds - late_odds)
        result["sp_drift_pct"] = float((early_odds - late_odds) / early_odds)
    
    return result
```

**Validation**:
1. This feature depends on having multiple odds snapshots per dog per race. First verify that `odds_snapshots` is being populated by checking `SELECT COUNT(*) FROM odds_snapshots`.
2. If the table is empty, this feature cannot be computed — skip until odds scraping is implemented.
3. For a dog whose odds shortened from 5.0 to 3.0: verify `sp_drift = 2.0` and `sp_drift_pct = 0.4`.
4. For a dog whose odds drifted from 3.0 to 5.0: verify `sp_drift = -2.0` (negative = bad sign).

---

### 2.6 Optimal Rest Window

**Rationale**: Research shows greyhounds perform best with 7-14 day rest intervals. Dogs returning from long layoffs (28+ days) or racing on very short rest (<5 days) show measurably worse outcomes.

**Implementation type**: `code` (seed feature)

**Files to modify**:
- `backend/scripts/seed_features.py`

```python
{
    "name": "optimal_rest_window",
    "display_name": "Optimal Rest Window",
    "description": "Binary: 1.0 if 7-14 days since last race (optimal), 0.0 otherwise. Based on research showing greyhounds peak with 7-14 day rest intervals.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    last_race_date = dog_history["race_date"].max()
    current_date = race_context.get("race_date")
    if last_race_date is None or current_date is None:
        return None
    days_rest = (current_date - last_race_date).days
    return 1.0 if 7 <= days_rest <= 14 else 0.0
""",
    "input_columns": ["race_date"],
},
{
    "name": "rest_category",
    "display_name": "Rest Category",
    "description": "Categorized rest: 1=short (<5 days), 2=quick turnaround (5-6), 3=optimal (7-14), 4=freshened (15-28), 5=layoff (29+). Encoded as integer.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    last_race_date = dog_history["race_date"].max()
    current_date = race_context.get("race_date")
    if last_race_date is None or current_date is None:
        return None
    days = (current_date - last_race_date).days
    if days < 5:
        return 1.0
    elif days < 7:
        return 2.0
    elif days <= 14:
        return 3.0
    elif days <= 28:
        return 4.0
    else:
        return 5.0
""",
    "input_columns": ["race_date"],
},
```

**Validation**:
1. Dog with 10 days rest: verify `optimal_rest_window = 1.0` and `rest_category = 3.0`.
2. Dog with 3 days rest: verify `optimal_rest_window = 0.0` and `rest_category = 1.0`.
3. Dog with 45 days rest: verify `optimal_rest_window = 0.0` and `rest_category = 5.0`.
4. Compute win rate by `rest_category` across historical data. Category 3 (7-14 days) should have the highest win rate.

---

### 2.7 Bayesian-Smoothed Win Rates

**Rationale**: Current win rates with small sample sizes are noisy. A dog with 1 win from 1 race shows 100% win rate, which is misleading. Bayesian smoothing adds a prior to regularize these estimates.

**Implementation type**: `code` (seed feature)

**Files to modify**:
- `backend/scripts/seed_features.py`

```python
{
    "name": "bayesian_win_rate",
    "display_name": "Bayesian-Smoothed Win Rate",
    "description": "Win rate smoothed with Bayesian prior (Beta(1,5) prior ~ 17% base rate for 6-runner races). Prevents noisy estimates from small samples.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    positions = dog_history["finish_position"].dropna()
    if positions.empty:
        return None
    wins = (positions == 1).sum()
    total = len(positions)
    # Beta(1, 5) prior = expectation of ~17% (1/6 base rate)
    prior_alpha = 1.0
    prior_beta = 5.0
    smoothed = (wins + prior_alpha) / (total + prior_alpha + prior_beta)
    return float(smoothed)
""",
    "input_columns": ["finish_position"],
},
{
    "name": "bayesian_place_rate",
    "display_name": "Bayesian-Smoothed Place Rate",
    "description": "Place rate (top 3) smoothed with Bayesian prior (Beta(3,3) prior ~ 50% base rate)",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    positions = dog_history["finish_position"].dropna()
    if positions.empty:
        return None
    places = (positions <= 3).sum()
    total = len(positions)
    prior_alpha = 3.0
    prior_beta = 3.0
    smoothed = (places + prior_alpha) / (total + prior_alpha + prior_beta)
    return float(smoothed)
""",
    "input_columns": ["finish_position"],
},
```

**Validation**:
1. Dog with 1 win from 1 race: raw win rate = 100%, Bayesian smoothed = (1+1)/(1+1+5) = 28.6%. Verify.
2. Dog with 5 wins from 30 races: raw = 16.7%, Bayesian = (5+1)/(30+6) = 16.7%. For large samples the prior has minimal effect. Verify.
3. Dog with 0 wins from 0 races (no history): should return `None`.
4. Compare `bayesian_win_rate` vs `win_rate_last10` across all dogs. The Bayesian version should have lower variance, especially for dogs with few races.

---

## TIER 3: Good Evidence, Higher Effort

---

### 3.1 Elo/Glicko Rating System

**Rationale**: Assign each dog a running rating (like chess Elo) that updates after every race based on who they beat and by how much. This captures current form as a single continuous number. Several academic papers on horse racing show Elo outperforms simple win-rate features.

**Implementation type**: `engine` (requires a dedicated computation step, not a simple per-entry feature)

**Files to modify**:
- Create new file: `backend/ml/elo_ratings.py`
- Modify `backend/ml/dataset_builder.py` to incorporate Elo as a feature column

**Implementation details**:

```python
# backend/ml/elo_ratings.py
"""Elo rating system for greyhound performance tracking."""

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.models.race import Race
from app.models.race_entry import RaceEntry


DEFAULT_RATING = 1500.0
K_FACTOR = 32.0


def compute_elo_ratings(db: Session, before_date=None) -> dict[int, float]:
    """Compute Elo ratings for all dogs based on race results.
    
    Process all races chronologically. For each race:
    - Compare each pair of dogs
    - The dog that finished ahead "beat" the other
    - Update ratings using standard Elo formula
    
    Returns: dict mapping dog_id -> current Elo rating
    """
    ratings: dict[int, float] = {}
    
    query = (
        db.query(Race.id, Race.race_date)
        .filter(Race.status == "resulted")
        .order_by(Race.race_date, Race.id)
    )
    if before_date:
        query = query.filter(Race.race_date < before_date)
    
    races = query.all()
    
    for race_id, race_date in races:
        entries = (
            db.query(RaceEntry.dog_id, RaceEntry.finish_position)
            .filter(
                RaceEntry.race_id == race_id,
                RaceEntry.finish_position.isnot(None),
            )
            .all()
        )
        
        if len(entries) < 2:
            continue
        
        # Get current ratings (default for new dogs)
        entry_ratings = {
            dog_id: ratings.get(dog_id, DEFAULT_RATING) 
            for dog_id, _ in entries
        }
        
        # Compare all pairs
        rating_changes: dict[int, float] = {dog_id: 0.0 for dog_id, _ in entries}
        n_comparisons = 0
        
        for i, (dog_a, pos_a) in enumerate(entries):
            for j, (dog_b, pos_b) in enumerate(entries):
                if i >= j:
                    continue
                
                # Expected score for dog_a
                expected_a = 1.0 / (1.0 + 10 ** ((entry_ratings[dog_b] - entry_ratings[dog_a]) / 400.0))
                
                # Actual score: 1 if A beat B, 0 if B beat A, 0.5 if tie
                if pos_a < pos_b:
                    actual_a = 1.0
                elif pos_a > pos_b:
                    actual_a = 0.0
                else:
                    actual_a = 0.5
                
                delta = K_FACTOR * (actual_a - expected_a)
                rating_changes[dog_a] += delta
                rating_changes[dog_b] -= delta
                n_comparisons += 1
        
        # Scale changes by number of opponents (so ratings don't inflate with field size)
        n_runners = len(entries)
        for dog_id in rating_changes:
            ratings[dog_id] = entry_ratings[dog_id] + rating_changes[dog_id] / (n_runners - 1)
    
    return ratings
```

In `dataset_builder.py`, before building the feature matrix, compute Elo ratings and add as a column:

```python
from ml.elo_ratings import compute_elo_ratings

# Compute Elo ratings up to the split point
elo_ratings = compute_elo_ratings(db, before_date=test_cutoff_date)
df["elo_rating"] = df["dog_id"].map(elo_ratings).fillna(DEFAULT_RATING)
```

**Important**: For proper train/val/test separation, Elo ratings should be computed using only data available at the time of each race. The simplest approach is to compute ratings up to the test cutoff date. A more rigorous approach would compute ratings incrementally for each race.

**Validation**:
1. Process a small subset of races manually. Verify Elo updates match the formula.
2. Plot Elo distributions — should be roughly normal around 1500 with known top dogs above 1600.
3. Compute correlation between Elo rating and win rate in test set. Should be strongly positive.
4. Verify Elo improves model accuracy beyond existing features by training with/without it.
5. **Performance note**: Computing Elo for all historical races is O(n * m^2) where n = races, m = runners. This may be slow for large datasets. Consider caching ratings and updating incrementally.

---

### 3.2 Run-Style Classification

**Rationale**: Beyond the binary `is_front_runner`, classifying dogs into full run-style categories (front-runner, prominent, midfield, closer) creates richer tactical features. This interacts with pace shape and trap draw.

**Implementation type**: `code` (seed feature)

**Files to modify**:
- `backend/scripts/seed_features.py`

```python
{
    "name": "run_style_category",
    "display_name": "Run Style Classification",
    "description": "Classified run style: 1=front-runner (led/made all), 2=prominent (disputed/close up), 3=midfield, 4=closer (ran on/came from behind), 5=slow starter. Based on comment parsing.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    recent = dog_history.tail(10)
    comments = recent["comment"].dropna().str.lower()
    if comments.empty:
        return 3.0  # default midfield

    style_scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for c in comments:
        if any(kw in c for kw in ["led", " ld", "made all", "led run in"]):
            style_scores[1] += 1
        elif any(kw in c for kw in ["disp", "2nd early", "close up", "prominent", "tracked"]):
            style_scores[2] += 1
        elif any(kw in c for kw in ["ran on", "chl", "finished well", "late hdwy", "stayed on"]):
            style_scores[4] += 1
        elif any(kw in c for kw in ["slow away", "saway", "slowly away", "dwelt"]):
            style_scores[5] += 1
        else:
            style_scores[3] += 1
    
    # Return the most common style
    return float(max(style_scores, key=style_scores.get))
""",
    "input_columns": ["comment"],
},
{
    "name": "closing_speed_last5",
    "display_name": "Closing Speed (last 5)",
    "description": "Average run-in time (finish_time - sectional_time) over last 5 races. Lower = stronger finisher.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    recent = dog_history.tail(5)
    valid = recent.dropna(subset=["finish_time", "sectional_time"])
    if valid.empty:
        return None
    closing = valid["finish_time"] - valid["sectional_time"]
    return float(closing.mean())
""",
    "input_columns": ["finish_time", "sectional_time"],
},
```

**Validation**:
1. Find a dog whose comments consistently include "led" — should classify as style 1.
2. Find a dog with "ran on" comments — should classify as style 4.
3. Verify `closing_speed_last5` is always positive (finish_time > sectional_time).
4. Create pace interaction: `run_style_category * is_sole_front_runner` — sole front-runners who are classified as style 1 should have the highest win rates.

---

### 3.3 Interaction Features

**Rationale**: Tree-based models (XGBoost, LightGBM) can learn interactions, but explicitly encoding key interactions as features helps them find these patterns faster and with less data.

**Implementation type**: `engine` (add to `dataset_builder.py`)

**Files to modify**:
- `backend/ml/dataset_builder.py` — Add after all individual features are computed

**Implementation details**:

```python
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed interaction features."""
    
    # Class drop after rest = the classic 'plot' (dropping in class + freshened up)
    if "grade_movement" in df.columns and "days_since_last" in df.columns:
        df["class_drop_after_rest"] = df["grade_movement"].clip(lower=0) * (
            df["days_since_last"].clip(lower=0) / 14.0  # normalize to ~1.0 at 2 weeks
        )
    
    # Weight change x improving form
    if "weight_change" in df.columns and "improving_form" in df.columns:
        df["weight_loss_improving"] = (
            (-df["weight_change"]).clip(lower=0) *  # weight loss (positive)
            (-df["improving_form"]).clip(lower=0)    # improving positions (positive)
        )
    
    # Dog age x career races (experience for age)
    if "dog_age_years" in df.columns and "career_races" in df.columns:
        df["age_x_experience"] = df["dog_age_years"] * df["career_races"]
    
    # Trap inside x early speed x same-trap win rate (triple interaction)
    if all(c in df.columns for c in ["early_speed_ratio", "trap_win_rate_at_track"]):
        df["trap_speed_synergy"] = (
            df["early_speed_ratio"] * df["trap_win_rate_at_track"]
        )
    
    return df
```

**Validation**:
1. A dog dropping 2 grades after 14 days rest should have `class_drop_after_rest = 2.0 * 1.0 = 2.0`. Verify.
2. A dog stepping UP in class (negative grade_movement) should have `class_drop_after_rest = 0.0` (clipped). Verify.
3. Train models with and without interaction features. Expect 0.5-1.5% improvement in accuracy.
4. Check SHAP interaction plots — `class_drop_after_rest` should show a clear positive effect on win probability.

---

### 3.4 Field Quality Metric

**Rationale**: A dog's raw time or position matters less than the quality of the field it was achieved against. A 2nd place finish in an OR (open race) is much more impressive than a 2nd in A8.

**Implementation type**: `engine` (add to `dataset_builder.py` in race-relative features)

**Files to modify**:
- `backend/ml/dataset_builder.py`

**Implementation details**:

```python
# Field quality: average mean_finish_time of all runners in this race
if "mean_finish_time_last5" in df.columns:
    df["field_quality"] = df.groupby("race_id")["mean_finish_time_last5"].transform("mean")
    # Dog's time advantage over the field quality
    df["speed_vs_field_quality"] = df["mean_finish_time_last5"] - df["field_quality"]

# Also: number of runners (already available but worth including explicitly)
if "num_runners" in df.columns:
    pass  # Already available from race context
else:
    df["num_runners"] = df.groupby("race_id")["race_id"].transform("count")
```

**Validation**:
1. For a specific race, manually compute the mean of all runners' `mean_finish_time_last5`. Verify `field_quality` matches.
2. Verify that OR/S1 races have lower (faster) `field_quality` than A8/A9 races.
3. `speed_vs_field_quality` should be negative for the fastest dog in the field and positive for the slowest.

---

### 3.5 Recency-Weighted Performance Rating

**Rationale**: A single composite "form" number that accounts for field size, recency, and finishing position. More informative than raw position stats because finishing 2nd in a 6-dog race is better than 2nd in a 4-dog race.

**Implementation type**: `code` (seed feature)

**Files to modify**:
- `backend/scripts/seed_features.py`

```python
{
    "name": "recency_weighted_rating",
    "display_name": "Recency-Weighted Performance Rating",
    "description": "Composite form score (0-1) that weights recent races more heavily and adjusts for field size. Higher = better recent form.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    recent = dog_history.tail(10)
    valid = recent.dropna(subset=["finish_position", "num_runners"])
    if valid.empty:
        return None
    
    n = len(valid)
    total_weight = 0.0
    weighted_score = 0.0
    
    for i, (_, row) in enumerate(valid.iterrows()):
        pos = row["finish_position"]
        runners = row["num_runners"]
        if runners < 2:
            continue
        # Score: 1.0 for winner, 0.0 for last
        score = (runners - pos) / (runners - 1)
        # Exponential decay weight: most recent race = highest weight
        weight = 0.7 ** (n - 1 - i)
        weighted_score += score * weight
        total_weight += weight
    
    if total_weight == 0:
        return None
    return float(weighted_score / total_weight)
""",
    "input_columns": ["finish_position", "num_runners"],
},
```

**Validation**:
1. Dog that won their last 3 races (all 6-runner): score should be close to 1.0. Verify.
2. Dog that finished last in their last 3 races: score should be close to 0.0. Verify.
3. Dog that finished 2nd in a 6-runner race vs 2nd in a 3-runner race: the 6-runner result should score higher (4/5 = 0.8 vs 1/2 = 0.5). Verify.
4. Check correlation with win rate in test data — should be among the most correlated single features.

---

## TIER 4: Additional Opportunities (Lower Priority)

These are worth exploring once the above tiers are implemented and evaluated.

---

### 4.1 Sex-Based Features

```python
{
    "name": "is_bitch",
    "display_name": "Sex (Bitch)",
    "description": "Binary: 1.0 if the dog is a bitch (female), 0.0 if a dog (male). Bitches may have different weight/performance patterns.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    # Note: requires sex to be available in race_context
    sex = race_context.get("sex")
    if sex is None:
        return None
    return 1.0 if sex == "B" else 0.0
""",
    "input_columns": [],
},
```

**Note**: This requires `sex` to be added to `race_context` in `feature_engine.py` / `feature_store.py`. Currently `race_context` only contains race-level data (track, distance, grade, trap, date), not dog-level data. You'll need to add `sex` from the `Dog` model.

---

### 4.2 Temporal Pattern Features

```python
{
    "name": "day_of_week",
    "display_name": "Day of Week",
    "description": "Day of week encoded as 0=Monday to 6=Sunday. Different days may have different field quality.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    race_date = race_context.get("race_date")
    if race_date is None:
        return None
    return float(race_date.weekday())
""",
    "input_columns": [],
},
```

---

### 4.3 Racing Frequency Trend

```python
{
    "name": "racing_frequency_trend",
    "display_name": "Racing Frequency Trend",
    "description": "Change in racing frequency: races in last 14 days minus races in prior 14 days. Positive = increasing campaign intensity.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    from datetime import timedelta
    race_date = race_context.get("race_date")
    if race_date is None:
        return None
    recent_14 = dog_history[dog_history["race_date"] >= (race_date - timedelta(days=14))]
    prior_14 = dog_history[
        (dog_history["race_date"] >= (race_date - timedelta(days=28))) &
        (dog_history["race_date"] < (race_date - timedelta(days=14)))
    ]
    return float(len(recent_14) - len(prior_14))
""",
    "input_columns": ["race_date"],
},
```

---

### 4.4 Weight Trend (Slope)

```python
{
    "name": "weight_trend_last5",
    "display_name": "Weight Trend (last 5)",
    "description": "Slope of weight over last 5 races. Negative = losing weight (often signals getting fitter). Uses linear regression.",
    "feature_type": "visual",
    "config_json": {
        "metric": "weight_kg",
        "aggregation": "trend",
        "window": {"type": "last_n", "n": 5},
        "filters": {}
    },
    "input_columns": ["weight_kg"],
},
```

---

### 4.5 Performance at Prize Money Level

```python
{
    "name": "prize_money_performance",
    "display_name": "Performance at Prize Money Level",
    "description": "Average position in races with similar or higher prize money. Captures class level more precisely than grade letters.",
    "feature_type": "code",
    "code": """
def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    # This requires prize_money in dog_history - check if available
    if "prize_money" not in dog_history.columns:
        return None
    current_prize = race_context.get("prize_money")
    if current_prize is None:
        return None
    # Filter to races at similar or higher prize level (within 80%)
    similar = dog_history[dog_history["prize_money"] >= current_prize * 0.8]
    positions = similar["finish_position"].dropna()
    if positions.empty:
        return None
    return float(positions.mean())
""",
    "input_columns": ["finish_position", "prize_money"],
},
```

**Note**: This requires `prize_money` to be added to the dog history query in `feature_engine.py`. Currently `Race.prize_money` is not queried in `get_dog_history()`.

---

## Implementation Order and Strategy

### Recommended phased approach:

**Phase 1 — Foundation changes** (do first, other features depend on these):
1. Add `adjusted_time` to `get_dog_history()` query in `feature_engine.py`
2. Add `"ewm"` aggregation to `feature_engine.py`
3. Add `prize_money` and `going_allowance` to `get_dog_history()` query (for future features)

**Phase 2 — Tier 1 features** (highest impact):
1. Seed the going-adjusted time features (1.1)
2. Add SP features to dataset_builder (1.2)
3. Add dog age to race_features.py (1.3)
4. Add pace/trap interactions to race_features.py (1.4)
5. Add early speed rank and pace shape to dataset_builder (1.5, 1.6)
6. Seed the ewm features (1.7)

**Phase 3 — Tier 2 features** (strong impact, more code):
1. Add trainer stats to race_features.py (2.1)
2. Add sire/dam stats to race_features.py (2.2)
3. Seed trouble-in-running features (2.3)
4. Add track-normalized speed rating (2.4)
5. Add SP drift if odds data available (2.5)
6. Seed rest window features (2.6)
7. Seed Bayesian-smoothed rates (2.7)

**Phase 4 — Tier 3 features** (good impact, higher effort):
1. Implement Elo rating system (3.1)
2. Seed run-style classification (3.2)
3. Add interaction features to dataset_builder (3.3)
4. Add field quality metric (3.4)
5. Seed recency-weighted rating (3.5)

**Phase 5 — Evaluate and iterate**:
1. Train models with each phase's features added incrementally
2. Use SHAP values to identify which new features are most predictive
3. Drop features that add noise (low importance, high correlation with existing features)
4. Consider feature selection (e.g., mutual information, forward selection) to find the optimal subset

---

## Global Validation Checklist

After implementing any features, run these checks:

1. **No data leakage**: All features use only data from BEFORE the target race date. Verify `before_date` filtering is applied everywhere.
2. **NaN handling**: Features should return `None` (not raise exceptions) when data is insufficient. Verify with dogs that have 0, 1, and 5+ races of history.
3. **Existing tests pass**: Run `pytest backend/` after each phase.
4. **Feature materialization**: Run `python -c "from ml.feature_store import materialize_features; ..."` to verify features can be computed in batch.
5. **Dataset building**: Build a test dataset with the new features and verify no errors.
6. **Baseline comparison**: Always train a model WITHOUT the new features first (baseline), then WITH them. Record the delta in accuracy, log-loss, and calibration.
7. **Feature importance**: After training, check SHAP summary plots. Features with zero importance may be redundant.
8. **Correlation check**: New features that are > 0.95 correlated with existing features add noise, not signal. Drop one of the pair.

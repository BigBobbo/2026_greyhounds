# Autoresearch: Greyhound Prediction Research Directions

This file describes the research directions for the autonomous ML experiment loop.
The autoresearch loop reads this to understand what to explore.

## Primary Objective

Maximize **Kelly criterion ROI** (`betting_kelly_roi`) on the held-out test set.
This is the metric that matters for a real betting system — it combines prediction
accuracy with calibration quality and staking discipline.

## Research Directions

### 1. Hyperparameter Optimization (40% of trials)
- Explore learning rate / n_estimators tradeoffs (lower LR + more trees)
- Regularization strength (reg_alpha, reg_lambda, min_child_weight)
- Tree structure (max_depth, num_leaves, min_child_samples)
- Subsampling rates for rows and columns (bagging/feature dropout)

### 2. Feature Selection (25% of trials)
- Identify and remove noisy features that hurt generalization
- Test minimal feature sets (top 5, top 10 by SHAP importance)
- Try adding/removing individual features to measure marginal value
- Focus on features that capture form vs class vs pace vs draw

### 3. Algorithm Comparison (10% of trials)
- Compare XGBoost, LightGBM, LambdaRank, Random Forest on same features
- LambdaRank may excel since this is fundamentally a ranking problem
  (predict which dog finishes first, not absolute win probability)
- Gradient boosting vs bagging tradeoffs

### 4. Combined Mutations (15% of trials)
- Jointly optimize features + hyperparameters
- Find algorithm-specific feature preferences

### 5. Random Exploration (10% of trials)
- Fully random configs to escape local optima
- Discover unexpectedly good parameter regions

## Domain Knowledge for Feature Engineering

Key predictive signals in greyhound racing:
- **Recent form**: Last 5 finish positions/times (mean, stdev, trend)
- **Track/distance fitness**: Dog's record at specific track+distance
- **Trap position**: Inside traps (1-2) have structural advantage at some tracks
- **Class movement**: Dogs dropping in grade are often value picks
- **Weight changes**: Sudden weight changes can indicate fitness issues
- **Days since last race**: Fresh dogs vs ring-rusty dogs
- **Early speed**: Sectional times predict front-running style
- **Consistency**: Low stdev in finish positions = more predictable

## Success Criteria

- Kelly ROI > 5% on test set is a strong signal
- Top-pick strike rate > 20% (random baseline ~16.7% for 6-dog races)
- Value betting ROI > 0% indicates calibrated probabilities
- Model should beat favourite-backing baseline on all P&L metrics

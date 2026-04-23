"""Monotonic constraints for tree-based trainers.

LightGBM and XGBoost support per-feature monotonicity constraints — a
free-lunch form of regularization that encodes domain knowledge: a higher
ELO rating must not decrease the predicted win probability, a longer
starting price must not increase it, etc.  On a noisy dataset like
race-level outcomes this meaningfully reduces overfitting without
sacrificing any legitimate non-linear behaviour in other features.

`FEATURE_DIRECTIONS` gives the canonical direction **relative to the dog's
chance of winning** (higher feature value = higher win prob is +1, lower
is -1, no opinion is 0).

`build_monotone_constraints(feature_names, target)` translates this into
the per-column constraint vector that LightGBM/XGBoost expect, flipping
signs automatically when the target is `finish_position` or
`finish_time` (lower = better) so constraints stay aligned with the
actual optimization direction.
"""

from __future__ import annotations

# Direction relative to WIN probability for each known feature.
#   +1: higher feature value → higher P(win)
#   -1: higher feature value → lower  P(win)
# Features absent from this dict are left unconstrained (0).
#
# Only add a feature here if the relationship is genuinely monotonic in
# the domain.  Features with U-shaped relationships (e.g. days_since_last
# is bad at 1-3 days, good at 7-14, bad at 60+) must stay at 0 to let
# the trees capture the non-monotonic pattern.
FEATURE_DIRECTIONS: dict[str, int] = {
    # ---- ELO ratings: higher skill → higher win prob ----
    "dog_elo": 1,
    "dog_elo_at_distance": 1,
    "dog_elo_at_track": 1,
    "dog_elo_races": 0,  # sample-size proxy — ambiguous
    # Field-level ELO: higher field strength → lower THIS dog's win prob
    "field_avg_elo": -1,
    "field_max_elo": -1,
    "elo_rank_in_field": -1,  # rank 1 = best
    "elo_gap_to_best": -1,
    "elo_gap_to_avg": 1,
    # ---- Speed figures (Beyer-style: higher = faster) ----
    "speed_figure_best_last10": 1,
    "speed_figure_mean_last5": 1,
    "speed_figure_ewm_last10": 1,
    "speed_figure_trend_last5": 1,
    "career_peak_speed_figure": 1,
    "recent_vs_peak_speed_figure": 1,
    # ---- Win / place rates ----
    "win_rate_last10": 1,
    "place_rate_last10": 1,
    "bayesian_win_rate": 1,
    "bayesian_place_rate": 1,
    "win_rate_same_track": 1,
    "clean_run_win_rate_last10": 1,
    "clear_win_rate_last10": 1,
    # ---- Finish-position aggregates (lower position is better) ----
    "mean_position_last5": -1,
    "ewm_position_last10": -1,
    "clean_run_mean_position_last10": -1,
    "trouble_run_mean_position_last10": -1,
    "trouble_recovery_ratio_last10": -1,
    # ---- Times: lower finish time is better ----
    "mean_finish_time_last5": -1,
    "min_finish_time_last10": -1,
    "mean_finish_time_last5_same_dist": -1,
    "mean_adjusted_time_last5": -1,
    "mean_adjusted_time_last5_same_dist": -1,
    "best_adjusted_time_last10": -1,
    "best_adjusted_time_last10_same_dist": -1,
    "ewm_finish_time_last10": -1,
    "ewm_adjusted_time_last10": -1,
    "mean_beaten_dist_last5": -1,
    # ---- Market / SP (shorter price = market favours = better) ----
    "current_sp_decimal": -1,
    "mean_sp_last5": -1,
    "current_sp_implied_prob": 1,
    "current_sp_log_odds": -1,
    "current_sp_devigged_prob": 1,
    "sp_rank_in_field": -1,
    "is_favorite": 1,
    "is_second_favorite": 0,  # ambiguous
    "fav_gap": -1,  # larger gap = further from favourite
    # ---- Trouble ----
    "trouble_rate_last10": -1,
    "first_bend_trouble_rate": -1,
    "trouble_bend1_rate_last10": -1,
    "trouble_bend2_rate_last10": -1,
    "trouble_bend3_rate_last10": -1,
    "trouble_bend4_rate_last10": -1,
    "faded_rate_last10": -1,
    # ---- Trainer / sire ----
    "trainer_win_rate": 1,
    "trainer_place_rate": 1,
    "trainer_win_rate_at_track": 1,
    "sire_progeny_win_rate": 1,
    # ---- Stamina / finish profile ----
    "finishing_speed_ratio_last5": 1,
    "finishing_speed_ewm_last10": 1,
    "finishing_speed_trend_last5": 1,
    "finish_well_rate_last10": 1,
    # ---- Break quality ----
    "quick_away_rate_last10": 1,
    "slow_away_rate_last10": -1,
    "awkward_start_rate_last10": -1,
    # ---- Head-to-head ----
    "h2h_win_rate_vs_field": 1,
    "h2h_wins_vs_field": 1,
    "h2h_losses_vs_field": -1,
    "h2h_avg_beaten_length_vs_field": -1,
    "best_opponent_beaten_count": 1,
    # ---- Position consistency (lower stdev = more predictable) ----
    # Consistency is ambiguous for win prob — consistent placer vs
    # consistent winner look the same here — leave at 0.
    # ---- Days since last / rest: non-monotonic (U-shape) — leave at 0 ----
    # ---- Weight features: ambiguous — leave at 0 ----
}


def _auto_extend_for_race_relative_variants(
    base: dict[str, int],
) -> dict[str, int]:
    """For each base feature with a known direction, infer directions for
    its race-relative variants (__vs_field, __rank, __z_in_field,
    __gap_to_best, __is_field_best).

    Sign conventions:
      * __vs_field      — same direction as base (your value vs field mean)
      * __rank          — always decreasing (rank 1 = best → higher win prob)
      * __z_in_field    — signed so positive = better (implementation in
                          dataset_builder already flips by direction);
                          therefore always +1 regardless of base sign
      * __gap_to_best   — always decreasing (gap 0 means you ARE the best)
      * __is_field_best — always +1 (binary flag)
    """
    out = dict(base)
    for name, direction in list(base.items()):
        if direction == 0:
            continue
        out[f"{name}__vs_field"] = direction
        out[f"{name}__rank"] = -1
        out[f"{name}__z_in_field"] = 1
        out[f"{name}__gap_to_best"] = -1
        out[f"{name}__is_field_best"] = 1
    return out


FEATURE_DIRECTIONS = _auto_extend_for_race_relative_variants(FEATURE_DIRECTIONS)


def build_monotone_constraints(
    feature_names: list[str],
    target: str,
) -> list[int]:
    """Build a per-column monotone constraint vector.

    Args:
        feature_names: column names of X in the order seen by the trainer.
        target: the training target.  For "win_prob" constraints are used
            as-is (higher target = winning).  For "finish_position" and
            "finish_time" signs are flipped because lower target is better.

    Returns:
        A list of int constraints (-1, 0, or +1), one per feature.
    """
    # LambdaRank internally uses finish_position as relevance label but
    # relevance SCORE is "higher = better", so we use the win-prob
    # convention.  The training_service passes target="finish_position"
    # to build_dataset but the trainer optimizes higher = better. We
    # identify this case via the caller providing target="rank".
    if target in ("finish_position", "finish_time"):
        sign_flip = -1
    else:
        sign_flip = 1

    return [
        sign_flip * FEATURE_DIRECTIONS.get(name, 0)
        for name in feature_names
    ]

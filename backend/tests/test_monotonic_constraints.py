"""Unit tests for the monotonic-constraints helper."""

from ml.monotonic_constraints import (
    FEATURE_DIRECTIONS,
    build_monotone_constraints,
)


def test_known_directions_for_win_target():
    cols = ["dog_elo", "current_sp_decimal", "trainer_win_rate", "unknown_feature"]
    constraints = build_monotone_constraints(cols, target="win_prob")
    assert constraints == [1, -1, 1, 0]


def test_signs_flip_for_finish_position_target():
    # finish_position: lower position is better.  A feature that
    # increases win prob (e.g. dog_elo) must DECREASE predicted position.
    cols = ["dog_elo", "current_sp_decimal", "unknown_feature"]
    constraints = build_monotone_constraints(cols, target="finish_position")
    assert constraints == [-1, 1, 0]


def test_signs_flip_for_finish_time_target():
    # finish_time: lower time is better — same logic as finish_position
    cols = ["speed_figure_best_last10", "mean_finish_time_last5"]
    win_constraints = build_monotone_constraints(cols, target="win_prob")
    time_constraints = build_monotone_constraints(cols, target="finish_time")
    assert time_constraints == [-c for c in win_constraints]


def test_race_relative_variants_inferred():
    # __vs_field shares direction; __rank is decreasing; __z_in_field is
    # always +1 (signed by direction at construction time); __gap_to_best
    # is decreasing; __is_field_best is +1.
    assert FEATURE_DIRECTIONS["dog_elo__vs_field"] == 1
    assert FEATURE_DIRECTIONS["dog_elo__rank"] == -1
    assert FEATURE_DIRECTIONS["dog_elo__z_in_field"] == 1
    assert FEATURE_DIRECTIONS["dog_elo__gap_to_best"] == -1
    assert FEATURE_DIRECTIONS["dog_elo__is_field_best"] == 1


def test_constraint_vector_aligns_with_column_order():
    cols = ["unknown_a", "dog_elo", "unknown_b", "current_sp_decimal"]
    constraints = build_monotone_constraints(cols, target="win_prob")
    assert len(constraints) == 4
    assert constraints[0] == 0
    assert constraints[1] == 1
    assert constraints[2] == 0
    assert constraints[3] == -1


def test_unknown_target_treated_like_win_prob():
    cols = ["dog_elo"]
    out = build_monotone_constraints(cols, target="something_unexpected")
    assert out == [1]


def test_h2h_features_directionally_correct():
    cols = [
        "h2h_win_rate_vs_field",
        "h2h_losses_vs_field",
        "best_opponent_beaten_count",
    ]
    constraints = build_monotone_constraints(cols, target="win_prob")
    assert constraints == [1, -1, 1]


def test_speed_figure_consistency():
    # All speed-figure aggregates should be +1 for win prob (higher = better)
    sf_cols = [
        "speed_figure_best_last10",
        "speed_figure_mean_last5",
        "speed_figure_ewm_last10",
        "career_peak_speed_figure",
    ]
    constraints = build_monotone_constraints(sf_cols, target="win_prob")
    assert constraints == [1, 1, 1, 1]

"""The post-race leakage guard must cover every known post-race feature name."""

from ml.feature_availability import POST_RACE_FEATURE_NAMES, post_race_features_in_use


def test_known_escapees_are_guarded():
    # Regression: these three previously slipped past the guard. current_race_sp
    # trains on the closing market price; the other two read fields only
    # populated on the night / with the results.
    for name in (
        "current_race_sp",
        "weight_pct_of_career_avg",
        "trap_bias_deviation_going",
    ):
        assert name in POST_RACE_FEATURE_NAMES, name


def test_post_race_features_in_use_filters():
    used = post_race_features_in_use(
        ["speed_figure_mean_last5", "current_race_sp", "trainer_win_rate"]
    )
    assert set(used) == {"current_race_sp"}

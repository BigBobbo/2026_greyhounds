"""Tests for field-relative feature construction."""

import numpy as np
import pandas as pd

from ml.dataset_builder import add_race_relative_features


def _make_df():
    # Two races: race 10 (lower-is-better example) and race 20 (higher-is-better example)
    # win_rate_last10 is higher-is-better; mean_finish_time_last5 is lower-is-better.
    data = {
        "mean_finish_time_last5": [29.5, 29.1, 30.0, 28.8, 28.8, 29.2],
        "win_rate_last10": [0.10, 0.30, 0.20, 0.40, 0.40, 0.25],
        "dog_elo": [1500.0, 1600.0, 1550.0, 1700.0, 1700.0, 1550.0],
    }
    X = pd.DataFrame(data, index=[101, 102, 103, 201, 202, 203])
    race_ids = pd.Series([10, 10, 10, 20, 20, 20], index=X.index)
    return X, race_ids


def test_vs_field_and_rank():
    X, race_ids = _make_df()
    out = add_race_relative_features(X, race_ids)

    # Race 10: times = [29.5, 29.1, 30.0], mean = 29.5333...
    mean_r10 = np.mean([29.5, 29.1, 30.0])
    assert abs(out.loc[101, "mean_finish_time_last5__vs_field"] - (29.5 - mean_r10)) < 1e-9
    # Race 10: rank by ascending time, 29.1 is best
    assert out.loc[102, "mean_finish_time_last5__rank"] == 1
    # Race 20: win_rate_last10 rank — higher is better, tied 0.4 both rank 1
    assert out.loc[201, "win_rate_last10__rank"] == 1
    assert out.loc[202, "win_rate_last10__rank"] == 1
    assert out.loc[203, "win_rate_last10__rank"] == 3


def test_gap_to_best_and_is_field_best():
    X, race_ids = _make_df()
    out = add_race_relative_features(X, race_ids)

    # Race 10 lower-is-better, best time = 29.1
    assert abs(out.loc[101, "mean_finish_time_last5__gap_to_best"] - (29.5 - 29.1)) < 1e-9
    assert abs(out.loc[102, "mean_finish_time_last5__gap_to_best"] - 0.0) < 1e-9
    assert out.loc[102, "mean_finish_time_last5__is_field_best"] == 1.0
    assert out.loc[101, "mean_finish_time_last5__is_field_best"] == 0.0

    # Race 20 higher-is-better ELO, best = 1700; ties both flagged as best
    assert out.loc[201, "dog_elo__is_field_best"] == 1.0
    assert out.loc[202, "dog_elo__is_field_best"] == 1.0
    assert out.loc[203, "dog_elo__is_field_best"] == 0.0
    assert abs(out.loc[203, "dog_elo__gap_to_best"] - (1700.0 - 1550.0)) < 1e-9


def test_z_score_sign_follows_higher_is_better():
    X, race_ids = _make_df()
    out = add_race_relative_features(X, race_ids)

    # Race 10: best time (29.1 at index 102) should yield positive z (better than field).
    assert out.loc[102, "mean_finish_time_last5__z_in_field"] > 0
    # Slowest time (30.0 at 103) should be negative.
    assert out.loc[103, "mean_finish_time_last5__z_in_field"] < 0

    # Race 20: highest elo dogs should have positive z.
    assert out.loc[201, "dog_elo__z_in_field"] > 0
    assert out.loc[203, "dog_elo__z_in_field"] < 0


def test_num_runners():
    X, race_ids = _make_df()
    out = add_race_relative_features(X, race_ids)
    assert (out["num_runners"] == 3.0).all()


def test_empty_frame_returns_empty():
    X = pd.DataFrame()
    race_ids = pd.Series([], dtype=int)
    out = add_race_relative_features(X, race_ids)
    assert out.empty


def test_missing_columns_are_skipped_gracefully():
    # If a KEY_FEATURES column isn't present, the function should not error
    X = pd.DataFrame({"some_other_feature": [1.0, 2.0]}, index=[1, 2])
    race_ids = pd.Series([10, 10], index=X.index)
    out = add_race_relative_features(X, race_ids)
    # Still gets num_runners at minimum
    assert "num_runners" in out.columns
    assert "some_other_feature" in out.columns

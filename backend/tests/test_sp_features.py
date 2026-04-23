"""Tests for SP-derived market features in dataset_builder."""

import numpy as np
import pandas as pd

from ml.dataset_builder import _add_sp_features


def _make_frames():
    # Two 3-dog races with realistic SPs
    # Race 10: fav 2.0, 4.0, 8.0 — classic front-runner scenario
    # Race 20: tight — 3.0, 3.5, 4.0
    X = pd.DataFrame({"_placeholder": [0.0] * 6}, index=[101, 102, 103, 201, 202, 203])
    entries_df = pd.DataFrame(
        {
            "sp_decimal": [2.0, 4.0, 8.0, 3.0, 3.5, 4.0],
            "race_id": [10, 10, 10, 20, 20, 20],
        },
        index=[101, 102, 103, 201, 202, 203],
    )
    return X, entries_df


def test_basic_sp_columns_added():
    X, entries_df = _make_frames()
    out = _add_sp_features(X, entries_df)
    for col in (
        "current_sp_decimal",
        "current_sp_implied_prob",
        "current_sp_log_odds",
        "sp_rank_in_field",
        "market_overround",
        "current_sp_devigged_prob",
        "is_favorite",
        "is_second_favorite",
        "fav_gap",
        "second_fav_gap",
        "sp_vs_field_mean",
    ):
        assert col in out.columns


def test_devigged_probs_sum_to_one_per_race():
    X, entries_df = _make_frames()
    out = _add_sp_features(X, entries_df)
    for race_id in (10, 20):
        mask = entries_df["race_id"] == race_id
        # Per-race de-vigged probs should sum to 1.0 after rescaling by overround
        assert abs(out.loc[mask, "current_sp_devigged_prob"].sum() - 1.0) < 1e-9


def test_favorite_flags():
    X, entries_df = _make_frames()
    out = _add_sp_features(X, entries_df)
    # Race 10: 101 is favourite (2.0), 102 is second fav (4.0)
    assert out.loc[101, "is_favorite"] == 1.0
    assert out.loc[102, "is_favorite"] == 0.0
    assert out.loc[101, "is_second_favorite"] == 0.0
    assert out.loc[102, "is_second_favorite"] == 1.0


def test_fav_gap_zero_for_favorite_positive_for_others():
    X, entries_df = _make_frames()
    out = _add_sp_features(X, entries_df)
    assert out.loc[101, "fav_gap"] == 0.0
    assert out.loc[102, "fav_gap"] > 0.0
    assert out.loc[103, "fav_gap"] > out.loc[102, "fav_gap"]  # further out
    # Favourites have negative or zero gap to second favourite
    assert out.loc[101, "second_fav_gap"] < 0.0
    # Second favourite has gap 0 to second favourite
    assert abs(out.loc[102, "second_fav_gap"]) < 1e-9


def test_log_odds_monotonic_with_sp():
    X, entries_df = _make_frames()
    out = _add_sp_features(X, entries_df)
    # Higher SP -> higher log odds
    assert out.loc[101, "current_sp_log_odds"] < out.loc[102, "current_sp_log_odds"]
    assert out.loc[102, "current_sp_log_odds"] < out.loc[103, "current_sp_log_odds"]


def test_handles_missing_sp():
    X = pd.DataFrame({"_placeholder": [0.0] * 3}, index=[1, 2, 3])
    entries_df = pd.DataFrame(
        {"sp_decimal": [None, None, None], "race_id": [10, 10, 10]},
        index=[1, 2, 3],
    )
    out = _add_sp_features(X, entries_df)
    # Gracefully returns X unchanged
    assert "current_sp_decimal" not in out.columns

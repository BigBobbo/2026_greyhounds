"""Tests for expanded pace-shape features."""

import pandas as pd

from ml.dataset_builder import _add_pace_shape_features


def _make(fr, es, race_ids):
    X = pd.DataFrame(
        {"is_front_runner": fr, "early_speed_ratio": es},
        index=list(range(len(fr))),
    )
    race_ids = pd.Series(race_ids, index=X.index)
    return X, race_ids


def test_lone_speed_scenario():
    # One clear front-runner in a 4-dog race
    X, rids = _make(
        fr=[0.8, 0.2, 0.1, 0.0],
        es=[0.25, 0.30, 0.32, 0.35],
        race_ids=[1, 1, 1, 1],
    )
    out = _add_pace_shape_features(X, rids)
    assert (out["pace_scenario_lone_speed"] == 1.0).all()
    assert (out["pace_scenario_duel"] == 0.0).all()
    assert out.loc[0, "is_sole_front_runner"] == 1.0


def test_duel_scenario():
    X, rids = _make(
        fr=[0.9, 0.8, 0.1, 0.0],
        es=[0.24, 0.25, 0.32, 0.35],
        race_ids=[1, 1, 1, 1],
    )
    out = _add_pace_shape_features(X, rids)
    assert (out["pace_scenario_duel"] == 1.0).all()
    assert (out["pace_scenario_lone_speed"] == 0.0).all()
    # Neither dog is the sole front-runner
    assert (out["is_sole_front_runner"] == 0.0).all()


def test_no_speed_scenario_flags():
    X, rids = _make(
        fr=[0.0, 0.1, 0.2, 0.3],
        es=[0.30, 0.32, 0.34, 0.35],
        race_ids=[1, 1, 1, 1],
    )
    out = _add_pace_shape_features(X, rids)
    assert (out["pace_scenario_no_speed"] == 1.0).all()
    assert out["num_front_runners_in_race"].iloc[0] == 0


def test_expected_lead_probability_sums_to_one():
    X, rids = _make(
        fr=[0.8, 0.2, 0.1],
        es=[0.25, 0.30, 0.33],
        race_ids=[1, 1, 1],
    )
    out = _add_pace_shape_features(X, rids)
    assert abs(out["expected_lead_probability"].sum() - 1.0) < 1e-9
    # Fastest breaker should have the highest lead probability
    assert out.loc[0, "expected_lead_probability"] > out.loc[1, "expected_lead_probability"]


def test_avg_opponent_early_speed():
    X, rids = _make(
        fr=[0.5, 0.5, 0.5],
        es=[0.20, 0.30, 0.40],
        race_ids=[1, 1, 1],
    )
    out = _add_pace_shape_features(X, rids)
    # Dog 0 opponents avg: (0.30 + 0.40)/2 = 0.35
    assert abs(out.loc[0, "avg_opponent_early_speed"] - 0.35) < 1e-9


def test_running_style_mismatch_rewards_lone_front_runner():
    X, rids = _make(
        fr=[0.9, 0.1, 0.1, 0.1],
        es=[0.25, 0.30, 0.32, 0.33],
        race_ids=[1, 1, 1, 1],
    )
    out = _add_pace_shape_features(X, rids)
    # The lone front-runner should have a negative mismatch score (favourable)
    assert out.loc[0, "running_style_mismatch"] < 0.0

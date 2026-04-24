"""Smoke test for the Plackett-Luce trainer end-to-end on synthetic data."""

import numpy as np
import pandas as pd
import pytest

# LightGBM may not be installed in lightweight test environments.
lgb = pytest.importorskip("lightgbm")

from ml.trainers.plackett_luce_trainer import (
    PlackettLuceTrainer,
    _plackett_luce_grad_hess,
    _plackett_luce_nll,
)


def _make_synthetic_races(n_races: int, dogs_per_race: int = 6, seed: int = 0):
    """Generate races where finish position correlates with feature x0."""
    rng = np.random.default_rng(seed)
    rows = []
    positions = []
    groups = []
    for _ in range(n_races):
        # True scores: dog i's score is a noisy function of x0
        x0 = rng.normal(size=dogs_per_race)
        x1 = rng.normal(size=dogs_per_race)
        x2 = rng.normal(size=dogs_per_race)
        true_score = 2.0 * x0 + 0.5 * x1 + rng.normal(scale=0.5, size=dogs_per_race)
        # Positions: 1st = highest true_score
        order = np.argsort(-true_score)
        pos = np.empty(dogs_per_race, dtype=float)
        for rank, dog in enumerate(order):
            pos[dog] = rank + 1
        for i in range(dogs_per_race):
            rows.append([x0[i], x1[i], x2[i]])
            positions.append(pos[i])
        groups.append(dogs_per_race)
    X = pd.DataFrame(rows, columns=["x0", "x1", "x2"])
    y = pd.Series(positions, name="finish_position")
    return X, y, groups


def test_grad_hess_zero_when_perfectly_predicted():
    """If predictions exactly equal true scores and only one race exists,
    the gradient at the winner should be negative (push score up) and
    positive at the losers (push them down)."""
    preds = np.array([3.0, 1.0, -1.0])
    labels = np.array([1.0, 2.0, 3.0])  # dog 0 wins
    groups = np.array([3])
    grad, hess = _plackett_luce_grad_hess(preds, labels, groups, top_k=1)
    # Winner's gradient should be (p1[0] - 1) which is negative
    assert grad[0] < 0
    # Losers' gradients should be positive (p1[i] for i!=0)
    assert grad[1] > 0
    assert grad[2] > 0
    # Hessian should be strictly positive everywhere
    assert (hess > 0).all()


def test_nll_is_lower_when_predictions_match_truth():
    """Aligned predictions should give lower NLL than uniform predictions."""
    preds_good = np.array([2.0, 0.0, -2.0])
    preds_bad = np.array([0.0, 0.0, 0.0])
    labels = np.array([1.0, 2.0, 3.0])
    groups = np.array([3])

    nll_good = _plackett_luce_nll(preds_good, labels, groups, top_k=3)
    nll_bad = _plackett_luce_nll(preds_bad, labels, groups, top_k=3)
    assert nll_good < nll_bad


def test_trainer_smoke_end_to_end():
    """Train, predict, and read out position distributions on synthetic data."""
    X_train, y_train, g_train = _make_synthetic_races(120, seed=1)
    X_val, y_val, g_val = _make_synthetic_races(40, seed=2)

    trainer = PlackettLuceTrainer({
        "n_estimators": 30,
        "learning_rate": 0.1,
        "num_leaves": 8,
        "min_child_samples": 5,
    })
    result = trainer.train(X_train, y_train, X_val, y_val,
                           group_train=g_train, group_val=g_val)

    # Basic sanity on the result object
    assert "top1_accuracy" in result.metrics
    # Synthetic data is highly learnable; expect well above the 1/6 random rate
    assert result.metrics["top1_accuracy"] > 0.30

    # Henery lambdas were fitted (within the grid bounds we set)
    assert 0.3 <= trainer.henery.lambda_2 <= 1.1
    assert 0.3 <= trainer.henery.lambda_3 <= 1.1

    # predict + position_distributions
    scores = trainer.predict(X_val)
    assert scores.shape == (len(X_val),)

    dists = trainer.position_distributions(scores, group_sizes=g_val)
    assert len(dists) == len(g_val)
    for race_dist, g_size in zip(dists, g_val):
        assert race_dist.shape == (g_size, 4)
        # Each row sums to 1
        assert np.allclose(race_dist.sum(axis=1), 1.0, atol=1e-6)
        # Cols 0/1/2 each sum to 1 across the race
        assert race_dist[:, 0].sum() == pytest.approx(1.0, abs=1e-6)
        assert race_dist[:, 1].sum() == pytest.approx(1.0, abs=1e-6)
        assert race_dist[:, 2].sum() == pytest.approx(1.0, abs=1e-6)

    # scores_to_proba should give per-race-normalised win probabilities
    win_probs = trainer.scores_to_proba(scores, group_sizes=g_val)
    idx = 0
    for g in g_val:
        race_probs = win_probs[idx : idx + g]
        assert race_probs.sum() == pytest.approx(1.0, abs=1e-6)
        idx += g

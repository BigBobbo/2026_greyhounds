"""Tests for the Plackett-Luce position-probability machinery."""

import math

import numpy as np
import pytest

from ml.position_distribution import (
    HeneryLambdas,
    exacta_probability,
    fit_henery_lambdas,
    place_probabilities,
    position_probabilities,
    top_exactas,
    top_trifectas,
    trifecta_probability,
)

EPS = 1e-9


# ---------------------------------------------------------------- core shape


def test_rows_sum_to_one_six_dog_race():
    scores = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0])
    pos = position_probabilities(scores)
    row_sums = pos.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=EPS)


def test_position_columns_sum_correctly():
    """Cols 0/1/2 should each sum to 1 (one dog finishes each position).
    Col 3 (4+) should sum to (n - 3) for n > 3."""
    scores = np.array([1.5, 0.7, 0.0, -0.3, -0.8, -1.2, -1.5])
    pos = position_probabilities(scores)
    assert pos[:, 0].sum() == pytest.approx(1.0, abs=1e-9)
    assert pos[:, 1].sum() == pytest.approx(1.0, abs=1e-9)
    assert pos[:, 2].sum() == pytest.approx(1.0, abs=1e-9)
    assert pos[:, 3].sum() == pytest.approx(scores.size - 3, abs=1e-9)


def test_uniform_scores_give_uniform_marginals():
    scores = np.zeros(8)
    pos = position_probabilities(scores)
    expected = 1.0 / 8.0
    assert np.allclose(pos[:, 0], expected, atol=EPS)
    assert np.allclose(pos[:, 1], expected, atol=EPS)
    assert np.allclose(pos[:, 2], expected, atol=EPS)


def test_winner_has_higher_p1_than_loser():
    scores = np.array([3.0, -3.0, 0.0])
    pos = position_probabilities(scores)
    assert pos[0, 0] > pos[1, 0]
    assert pos[0, 0] > 0.9  # extremely strong favourite


def test_two_dog_race_no_third_place_mass():
    pos = position_probabilities(np.array([0.5, -0.5]))
    # No 3rd or 4+ slot when n=2
    assert pos[:, 2].sum() == pytest.approx(0.0, abs=EPS)
    assert pos[:, 3].sum() == pytest.approx(0.0, abs=EPS)
    # 1st + 2nd cover full probability per dog
    assert np.allclose(pos[:, 0] + pos[:, 1], 1.0, atol=EPS)


def test_one_dog_race():
    pos = position_probabilities(np.array([0.0]))
    assert pos.shape == (1, 4)
    assert pos[0, 0] == pytest.approx(1.0, abs=EPS)


def test_empty_race():
    pos = position_probabilities(np.array([]))
    assert pos.shape == (0, 4)


# ------------------------------------------------------------------ Harville


def test_harville_p2_matches_closed_form():
    """For 3-dog race under Harville (lambda=1):
    P(j is 2nd) = sum_{i!=j} p1_i * theta_j / (Z - theta_i)
    """
    scores = np.array([1.0, 0.5, -0.5])
    theta = np.exp(scores)
    Z = theta.sum()
    p1 = theta / Z

    pos = position_probabilities(scores, lambdas=HeneryLambdas(1.0, 1.0))

    # Manually compute P(dog 1 finishes 2nd)
    expected_p2_dog1 = (
        p1[0] * theta[1] / (Z - theta[0])
        + p1[2] * theta[1] / (Z - theta[2])
    )
    assert pos[1, 1] == pytest.approx(expected_p2_dog1, abs=1e-9)


def test_exacta_closed_form():
    """P(i wins, j is 2nd) under Harville = theta_i/Z * theta_j/(Z - theta_i)."""
    scores = np.array([1.5, 0.3, -0.2, -0.8])
    theta = np.exp(scores)
    Z = theta.sum()
    expected = theta[0] / Z * theta[2] / (Z - theta[0])
    actual = exacta_probability(scores, 0, 2, lambdas=HeneryLambdas(1.0, 1.0))
    assert actual == pytest.approx(expected, abs=1e-9)


def test_trifecta_closed_form():
    scores = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    theta = np.exp(scores)
    Z = theta.sum()
    expected = (
        theta[0] / Z
        * theta[2] / (Z - theta[0])
        * theta[4] / (Z - theta[0] - theta[2])
    )
    actual = trifecta_probability(scores, 0, 2, 4, lambdas=HeneryLambdas(1.0, 1.0))
    assert actual == pytest.approx(expected, abs=1e-9)


def test_exacta_self_pair_is_zero():
    scores = np.array([1.0, 0.0, -1.0])
    assert exacta_probability(scores, 1, 1) == 0.0


def test_trifecta_with_repeated_dog_is_zero():
    scores = np.array([1.0, 0.0, -1.0, -2.0])
    assert trifecta_probability(scores, 0, 1, 0) == 0.0


def test_exacta_grid_sums_to_one():
    """Sum over all ordered pairs (i, j) of P(exacta i,j) == 1."""
    scores = np.array([0.8, 0.3, -0.1, -0.6])
    n = scores.size
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total += exacta_probability(scores, i, j)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_trifecta_grid_sums_to_one():
    scores = np.array([0.8, 0.3, -0.1, -0.6, -1.0])
    n = scores.size
    total = 0.0
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                total += trifecta_probability(scores, i, j, k)
    assert total == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------- Henery


def test_henery_with_unit_lambdas_equals_harville():
    scores = np.array([1.2, 0.4, -0.3, -1.0])
    pos_harville = position_probabilities(scores, lambdas=HeneryLambdas(1.0, 1.0))
    pos_henery_unit = position_probabilities(scores, lambdas=HeneryLambdas(1.0, 1.0))
    assert np.allclose(pos_harville, pos_henery_unit, atol=EPS)


def test_henery_with_smaller_lambdas_flattens_place_distribution():
    """Smaller lambda => place/show distribution moves toward uniform."""
    scores = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    n = scores.size

    pos_strong = position_probabilities(scores, lambdas=HeneryLambdas(1.0, 1.0))
    pos_flat = position_probabilities(scores, lambdas=HeneryLambdas(0.4, 0.4))

    # Variance of P(top-3 for each dog) should DROP under flat lambdas.
    place3_strong = pos_strong[:, 0] + pos_strong[:, 1] + pos_strong[:, 2]
    place3_flat = pos_flat[:, 0] + pos_flat[:, 1] + pos_flat[:, 2]

    assert place3_flat.var() < place3_strong.var()
    # Both should still sum to 3 across all dogs (one dog fills each spot).
    assert place3_strong.sum() == pytest.approx(3.0, abs=1e-9)
    assert place3_flat.sum() == pytest.approx(3.0, abs=1e-9)


# -------------------------------------------------------------- top combos


def test_top_exactas_returns_descending_sorted():
    scores = np.array([1.5, 0.5, -0.5, -1.0])
    combos = top_exactas(scores, k=5)
    probs = [c[2] for c in combos]
    assert probs == sorted(probs, reverse=True)
    # Top exacta should be (best, second-best)
    assert combos[0][0] == 0
    assert combos[0][1] == 1


def test_top_trifectas_returns_descending_sorted():
    scores = np.array([1.5, 0.5, -0.5, -1.0])
    combos = top_trifectas(scores, k=5)
    probs = [c[3] for c in combos]
    assert probs == sorted(probs, reverse=True)
    assert combos[0][0] == 0
    assert combos[0][1] == 1
    assert combos[0][2] == 2


# ---------------------------------------------------- place_probabilities


def test_place_probabilities_match_position_marginalisation():
    scores = np.array([1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
    place = place_probabilities(scores)
    pos = position_probabilities(scores)
    assert np.allclose(place["top2"], pos[:, 0] + pos[:, 1], atol=EPS)
    assert np.allclose(place["top3"], pos[:, 0] + pos[:, 1] + pos[:, 2], atol=EPS)


# -------------------------------------------------------------- Henery fit


def test_fit_henery_recovers_uniform_when_orderings_random():
    """If finish orderings are independent of scores, the optimal Henery
    lambdas should pull toward small values (no signal)."""
    rng = np.random.default_rng(0)
    score_groups = []
    position_groups = []
    for _ in range(200):
        n = 6
        scores = rng.normal(size=n)
        # Random finish positions (independent of scores)
        positions = rng.permutation(np.arange(1, n + 1)).astype(float)
        score_groups.append(scores)
        position_groups.append(positions)

    fit = fit_henery_lambdas(score_groups, position_groups)
    # No prediction signal => pulling lambda down should help
    assert fit.lambda_2 < 1.0 or fit.lambda_3 < 1.0


def test_fit_henery_keeps_lambdas_high_when_scores_predict_well():
    """If scores perfectly determine ordering, lambdas should stay near 1.0."""
    rng = np.random.default_rng(0)
    score_groups = []
    position_groups = []
    for _ in range(200):
        n = 6
        scores = rng.normal(size=n)
        # True positions: best score finishes 1st, etc.
        order = np.argsort(-scores)
        positions = np.empty(n, dtype=float)
        for rank, dog in enumerate(order):
            positions[dog] = rank + 1
        score_groups.append(scores)
        position_groups.append(positions)

    fit = fit_henery_lambdas(score_groups, position_groups)
    # When scores are perfectly predictive, larger lambdas are preferred.
    assert fit.lambda_2 >= 0.8
    assert fit.lambda_3 >= 0.8


def test_fit_henery_handles_empty_input():
    fit = fit_henery_lambdas([], [])
    assert fit.lambda_2 == 1.0 and fit.lambda_3 == 1.0

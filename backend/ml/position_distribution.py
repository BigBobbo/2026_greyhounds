"""Plackett-Luce position-probability machinery.

Given a per-dog scoring function (e.g. from a learned ranker), this module
produces:

    * P(dog i finishes 1st, 2nd, 3rd) for every dog in a race
    * P(top-2) and P(top-3) marginals (for place / show betting)
    * Top-K exacta and trifecta combinations with their probabilities
    * Expected-value helpers for ordered combination bets

The base model is Plackett-Luce / Harville:

    theta_i = exp(score_i)
    P(dog i finishes 1st) = theta_i / sum_k theta_k
    P(dog j finishes 2nd | i was 1st) = theta_j / sum_{k != i} theta_k
    ...

Vanilla Harville is well known to over-estimate the place / show probabilities
of favourites and under-estimate longshots.  To correct this we apply the
Henery (1981) extension, which uses position-specific score discounts:

    P(dog j finishes 2nd | i was 1st) propto exp(lambda_2 * score_j)
    P(dog k finishes 3rd | i, j)       propto exp(lambda_3 * score_k)

With (lambda_2, lambda_3) = (1, 1) we recover Harville exactly.  Empirical
fits typically give 0.6 < lambda_2 < 0.95 and 0.5 < lambda_3 < 0.85.

The trainer fits the lambdas on a validation set; this module just consumes
them.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class HeneryLambdas:
    """Position-specific Henery score discounts.

    lambda_1 is fixed at 1.0 by definition (the model is trained on win
    likelihood, so scores are already calibrated for 1st-place ranking).
    """

    lambda_2: float = 1.0
    lambda_3: float = 1.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (1.0, self.lambda_2, self.lambda_3)


def _stable_softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D score vector."""
    if scores.size == 0:
        return scores
    shifted = scores - scores.max()
    exp = np.exp(np.clip(shifted, -100.0, 0.0))
    total = exp.sum()
    if total <= 0:
        return np.full_like(scores, 1.0 / scores.size, dtype=float)
    return exp / total


def position_probabilities(
    scores: np.ndarray,
    lambdas: HeneryLambdas | None = None,
) -> np.ndarray:
    """Return the per-dog probability of finishing 1st, 2nd, 3rd, or 4+.

    Args:
        scores: 1-D array of shape (n,) — raw scores for each dog in a race.
        lambdas: Henery discount parameters.  Defaults to Harville (1,1,1).

    Returns:
        Array of shape (n, 4) where column k is P(dog finishes position k+1)
        for k in {0, 1, 2}, and column 3 is P(finishes outside the top 3).
        Each row sums to 1.0.  Each column sums to 1.0 for k in {0, 1, 2};
        column 3 sums to (n - 3) for n > 3 and to 0 otherwise.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return np.zeros((0, 4))

    lam = lambdas or HeneryLambdas()
    lam1, lam2, lam3 = lam.as_tuple()

    out = np.zeros((n, 4), dtype=float)

    # Position 1: softmax over the full field with score = lam1 * s.
    p1 = _stable_softmax(lam1 * scores)
    out[:, 0] = p1

    if n == 1:
        return out

    # Position 2: marginalise over which dog won.
    # P(j is 2nd) = sum_{i != j} P(i is 1st) * P(j is 2nd | i was 1st)
    #            = sum_{i != j} p1_i * softmax(lam2 * s; field \ {i})_j
    s2 = lam2 * scores
    for winner in range(n):
        # Softmax of s2 over field excluding `winner`
        mask = np.ones(n, dtype=bool)
        mask[winner] = False
        cond = _stable_softmax(s2[mask])
        # Distribute cond probabilities into out[:, 1] for non-winner dogs
        out[mask, 1] += p1[winner] * cond

    if n == 2:
        # All remaining mass is in 4+ slot, but n=2 means there is no 3rd or 4+.
        return out

    # Position 3: marginalise over (winner, runner-up) pairs.
    # This is O(n^2) but n is at most ~8 for greyhound racing, so trivial.
    s3 = lam3 * scores
    for winner in range(n):
        # P(winner = i) = p1[i]
        # Conditional on winner, P(runner-up = j) uses the softmax of s2 over
        # the field excluding the winner.
        mask_w = np.ones(n, dtype=bool)
        mask_w[winner] = False
        cond_2 = _stable_softmax(s2[mask_w])  # length n-1
        # Map cond_2 back to full-length index space
        cond_2_full = np.zeros(n)
        cond_2_full[mask_w] = cond_2

        for runner in range(n):
            if runner == winner:
                continue
            mask_wr = mask_w.copy()
            mask_wr[runner] = False
            if not mask_wr.any():
                continue
            cond_3 = _stable_softmax(s3[mask_wr])
            joint = p1[winner] * cond_2_full[runner]
            out[mask_wr, 2] += joint * cond_3

    # Position 4+: residual mass per dog.
    if n > 3:
        residual = 1.0 - out[:, :3].sum(axis=1)
        out[:, 3] = np.clip(residual, 0.0, 1.0)
    else:
        out[:, 3] = 0.0

    return out


def place_probabilities(
    scores: np.ndarray,
    lambdas: HeneryLambdas | None = None,
) -> dict[str, np.ndarray]:
    """Per-dog top-2 and top-3 probabilities (for place betting).

    Returns a dict with keys ``top2`` and ``top3``, each a 1-D array of length n.
    """
    pos = position_probabilities(scores, lambdas=lambdas)
    return {
        "top2": pos[:, 0] + pos[:, 1],
        "top3": pos[:, 0] + pos[:, 1] + pos[:, 2],
    }


def exacta_probability(
    scores: np.ndarray,
    first: int,
    second: int,
    lambdas: HeneryLambdas | None = None,
) -> float:
    """P(`first` wins AND `second` is runner-up).

    Closed form under Henery: p1[first] * softmax(lam2 * s; field \\ {first})[second].
    """
    if first == second:
        return 0.0
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if first < 0 or second < 0 or first >= n or second >= n:
        raise ValueError("first/second must be valid dog indices")

    lam = lambdas or HeneryLambdas()
    p1 = _stable_softmax(lam.as_tuple()[0] * scores)
    mask = np.ones(n, dtype=bool)
    mask[first] = False
    cond = _stable_softmax(lam.lambda_2 * scores[mask])
    # Position of `second` inside the masked array
    cond_full = np.zeros(n)
    cond_full[mask] = cond
    return float(p1[first] * cond_full[second])


def trifecta_probability(
    scores: np.ndarray,
    first: int,
    second: int,
    third: int,
    lambdas: HeneryLambdas | None = None,
) -> float:
    """P(`first` wins AND `second` is 2nd AND `third` is 3rd)."""
    if len({first, second, third}) < 3:
        return 0.0
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    for idx in (first, second, third):
        if idx < 0 or idx >= n:
            raise ValueError("indices must be valid dog indices")

    lam = lambdas or HeneryLambdas()
    _, lam2, lam3 = lam.as_tuple()

    p1 = _stable_softmax(scores)[first]

    mask2 = np.ones(n, dtype=bool)
    mask2[first] = False
    cond_2 = _stable_softmax(lam2 * scores[mask2])
    cond_2_full = np.zeros(n)
    cond_2_full[mask2] = cond_2
    p2 = cond_2_full[second]

    mask3 = mask2.copy()
    mask3[second] = False
    cond_3 = _stable_softmax(lam3 * scores[mask3])
    cond_3_full = np.zeros(n)
    cond_3_full[mask3] = cond_3
    p3 = cond_3_full[third]

    return float(p1 * p2 * p3)


def top_exactas(
    scores: np.ndarray,
    k: int = 10,
    lambdas: HeneryLambdas | None = None,
) -> list[tuple[int, int, float]]:
    """Top-K (first, second, probability) exacta combinations sorted descending."""
    n = np.asarray(scores).size
    combos: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = exacta_probability(scores, i, j, lambdas=lambdas)
            combos.append((i, j, p))
    combos.sort(key=lambda t: t[2], reverse=True)
    return combos[:k]


def top_trifectas(
    scores: np.ndarray,
    k: int = 10,
    lambdas: HeneryLambdas | None = None,
) -> list[tuple[int, int, int, float]]:
    """Top-K (first, second, third, probability) trifecta combinations."""
    n = np.asarray(scores).size
    combos: list[tuple[int, int, int, float]] = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for m in range(n):
                if m == i or m == j:
                    continue
                p = trifecta_probability(scores, i, j, m, lambdas=lambdas)
                combos.append((i, j, m, p))
    combos.sort(key=lambda t: t[3], reverse=True)
    return combos[:k]


def expected_value(probability: float, decimal_odds: float | None) -> float | None:
    """Return EV per unit stake, or None if odds are missing."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    return probability * (decimal_odds - 1.0) - (1.0 - probability)


# ---------------------------------------------------------------------------
# Henery lambda fitting
# ---------------------------------------------------------------------------


def _negative_log_likelihood_top3(
    lambdas: HeneryLambdas,
    score_groups: Sequence[np.ndarray],
    position_groups: Sequence[np.ndarray],
) -> float:
    """Negative log-likelihood of observed top-3 finishing orders under Henery.

    Args:
        score_groups: list of per-race score vectors (one per race).
        position_groups: list of per-race finish position vectors (1-based).
    """
    nll = 0.0
    eps = 1e-12

    _, lam2, lam3 = lambdas.as_tuple()

    for scores, positions in zip(score_groups, position_groups):
        scores = np.asarray(scores, dtype=float)
        positions = np.asarray(positions, dtype=float)
        n = scores.size
        if n < 2:
            continue

        # Identify dogs that finished 1st, 2nd, 3rd (by smallest finish positions).
        # Skip races where positions aren't fully observed.
        if not np.isfinite(positions).all():
            continue

        order = np.argsort(positions)
        winner = int(order[0])

        # 1st-place factor (lam1 == 1)
        p1 = _stable_softmax(scores)
        nll -= float(np.log(max(p1[winner], eps)))

        if n < 2 or order.size < 2:
            continue

        # 2nd-place factor under Henery
        runner = int(order[1])
        mask = np.ones(n, dtype=bool)
        mask[winner] = False
        cond2 = _stable_softmax(lam2 * scores[mask])
        # Find runner's index in the masked array
        masked_indices = np.where(mask)[0]
        runner_pos = int(np.where(masked_indices == runner)[0][0])
        nll -= float(np.log(max(cond2[runner_pos], eps)))

        if n < 3 or order.size < 3:
            continue

        # 3rd-place factor under Henery
        third = int(order[2])
        mask[runner] = False
        cond3 = _stable_softmax(lam3 * scores[mask])
        masked_indices = np.where(mask)[0]
        third_pos = int(np.where(masked_indices == third)[0][0])
        nll -= float(np.log(max(cond3[third_pos], eps)))

    return nll


def fit_henery_lambdas(
    score_groups: Sequence[np.ndarray],
    position_groups: Sequence[np.ndarray],
    grid: Iterable[float] = tuple(round(0.3 + 0.05 * i, 2) for i in range(0, 17)),
) -> HeneryLambdas:
    """Fit (lambda_2, lambda_3) by grid search over (0.3, 1.1).

    Grid search is cheap (default 17x17 = 289 evaluations) and avoids a
    SciPy dependency.  Each evaluation is O(sum of race sizes), so the
    whole fit is dominated by the dataset size, not the grid.

    Returns the best HeneryLambdas; falls back to (1.0, 1.0) if fitting
    fails or there is insufficient data.
    """
    grid_list = list(grid)
    if not grid_list or not score_groups:
        return HeneryLambdas(1.0, 1.0)

    best = HeneryLambdas(1.0, 1.0)
    best_nll = _negative_log_likelihood_top3(best, score_groups, position_groups)

    for l2 in grid_list:
        for l3 in grid_list:
            cand = HeneryLambdas(lambda_2=l2, lambda_3=l3)
            nll = _negative_log_likelihood_top3(cand, score_groups, position_groups)
            if nll < best_nll:
                best_nll = nll
                best = cand

    return best


def split_by_groups(
    arr: np.ndarray, group_sizes: Sequence[int]
) -> list[np.ndarray]:
    """Slice a flat array into per-race chunks given group sizes."""
    arr = np.asarray(arr)
    out: list[np.ndarray] = []
    idx = 0
    for g in group_sizes:
        out.append(arr[idx : idx + g])
        idx += g
    return out

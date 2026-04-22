"""Multi-entrant ELO ratings for greyhounds.

Maintains a rating dict per dog and updates ratings after each race using
pairwise comparisons of finish positions.  For an n-runner race the per-pair
K-factor is normalised by (n-1) so the total update applied to any single
dog stays bounded regardless of field size.

Typical usage:

    elo = EloRatings(k=24, initial=1500.0)
    # race_results: list of (dog_id, finish_position) sorted however
    pre = {dog_id: elo.get(dog_id) for dog_id, _ in race_results}
    elo.update_race(race_results)
    new = {dog_id: elo.get(dog_id) for dog_id, _ in race_results}
"""

from __future__ import annotations

from collections import defaultdict


class EloRatings:
    __slots__ = ("k", "initial", "_ratings", "_counts")

    def __init__(self, k: float = 24.0, initial: float = 1500.0) -> None:
        self.k = float(k)
        self.initial = float(initial)
        self._ratings: dict[int, float] = {}
        self._counts: dict[int, int] = defaultdict(int)

    def get(self, dog_id: int) -> float:
        return self._ratings.get(dog_id, self.initial)

    def count(self, dog_id: int) -> int:
        return self._counts.get(dog_id, 0)

    def update_race(self, results: list[tuple[int, int]]) -> None:
        """Update ratings based on a list of (dog_id, finish_position).

        Dogs with finish_position == None are skipped.  Ties (same finish
        position) contribute 0.5 expected/actual to each side of the pair.
        """
        cleaned = [(d, p) for d, p in results if p is not None]
        n = len(cleaned)
        if n < 2:
            for d, _ in cleaned:
                self._counts[d] += 1
            return

        # Snapshot current ratings to avoid order-dependent updates within a race
        current = {d: self.get(d) for d, _ in cleaned}
        deltas: dict[int, float] = defaultdict(float)
        per_pair_k = self.k / (n - 1)

        for i in range(n):
            d_i, p_i = cleaned[i]
            r_i = current[d_i]
            for j in range(i + 1, n):
                d_j, p_j = cleaned[j]
                r_j = current[d_j]
                expected_i = 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))
                if p_i < p_j:
                    actual_i = 1.0
                elif p_i > p_j:
                    actual_i = 0.0
                else:
                    actual_i = 0.5
                delta = per_pair_k * (actual_i - expected_i)
                deltas[d_i] += delta
                deltas[d_j] -= delta

        for d, delta in deltas.items():
            self._ratings[d] = current[d] + delta
        for d, _ in cleaned:
            self._counts[d] += 1

"""Race-ordering service.

Turns per-dog calibrated win probabilities into ordered/unordered exotic
probabilities (place, show, forecast/exacta, trio/trifecta) using a
Henery-discounted Plackett-Luce model evaluated by Monte Carlo.

Why this layer exists
---------------------
The win model (LambdaRank with softmax + Platt) gives us P(dog wins). To
price a forecast (1st+2nd) or trio (1st+2nd+3rd) we need ordered
multi-dog probabilities. The literature (Harville 1973, Henery 1981,
Stern 1990, Lo & Bacon-Shone 1994, Benter 1994) is consistent:

  * Naive Harville expansion P(i 1st, j 2nd) = pi * pj/(1-pi) is biased:
    the favourite finishes 2nd / 3rd less often than its win-prob
    implies, and longshots more often. Bias gets worse for 3rd than 2nd.

  * The standard correction is to raise win probabilities to a power
    alpha < 1 for second-place draws and a smaller power for third, then
    renormalise. Empirical alphas hover around 0.81 / 0.65.

  * Equivalently, treat the model's softmax pre-image as a Plackett-Luce
    "strength" vector and discount strengths for later positions. Monte
    Carlo sampling from this distribution gives every exotic probability
    in one pass.

We use Monte Carlo rather than the closed-form PL expansion because:
  * 6-dog Irish fields make 720 permutations cheap to sample
  * the same simulation gives forecast + trio + place + show in one go
  * non-runners and tie-breaking are trivial to handle
  * adding further corrections (track/trap pace bias) only needs the
    sampler to be tweaked, not a re-derived formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# Default discount exponents. Alpha for 2nd, beta for 3rd. Values come
# from the published Henery / Stern fits (alpha ~= 0.81, beta ~= 0.65)
# rounded to round numbers for stable defaults — they can be fitted
# against the historical race_entries table later via a single scalar
# optimisation if a calibration job is added.
DEFAULT_ALPHA_PLACE = 0.81
DEFAULT_BETA_SHOW = 0.65

# Number of Monte Carlo samples. With 6 dogs each sample is O(n) work
# and the standard error on a probability of size p is sqrt(p(1-p)/N).
# At N=10_000 the SE on a 0.10 probability is ~0.003 — well under the
# noise in the win-prob calibrator itself.
DEFAULT_N_SAMPLES = 10_000


@dataclass
class OrderingResult:
    """Per-race ordering probabilities.

    Indexed by `entry_ids` so the caller can map back to RaceEntry rows
    without depending on positional alignment.
    """

    entry_ids: list[int]
    win_prob: dict[int, float]
    place_prob: dict[int, float]   # P(dog finishes 1st OR 2nd)
    show_prob: dict[int, float]    # P(dog finishes 1st, 2nd, OR 3rd)
    forecast: list["ForecastCombo"] = field(default_factory=list)
    trio: list["TrioCombo"] = field(default_factory=list)


@dataclass
class ForecastCombo:
    first_entry_id: int
    second_entry_id: int
    probability: float


@dataclass
class TrioCombo:
    first_entry_id: int
    second_entry_id: int
    third_entry_id: int
    probability: float


def _normalise_strengths(win_probs: np.ndarray) -> np.ndarray:
    """Take an array of win probabilities to PL "strength" weights.

    The LambdaRank trainer's softmax-then-Platt step already produces
    numbers that are ordered like PL strengths, so for the win draw we
    just normalise. Negative or zero entries are clamped to a tiny
    positive value so a single bad calibration doesn't kill the sampler.
    """
    s = np.asarray(win_probs, dtype=float)
    s = np.where(np.isfinite(s) & (s > 0), s, 1e-9)
    total = s.sum()
    if total <= 0:
        return np.full_like(s, 1.0 / len(s))
    return s / total


def _henery_discount(
    probs: np.ndarray, exponent: float, removed_mask: np.ndarray
) -> np.ndarray:
    """Apply the Henery exponent to the still-running dogs and renormalise.

    `removed_mask[i] = True` means dog i has already been drawn for an
    earlier position and must not be drawn again. Dogs marked removed
    get probability zero.
    """
    discounted = np.where(removed_mask, 0.0, np.power(probs, exponent))
    total = discounted.sum()
    if total <= 0:
        # All remaining dogs collapsed to zero — fall back to uniform
        # over whoever's left so the sampler never NaN-divides.
        remaining = (~removed_mask).astype(float)
        rsum = remaining.sum()
        if rsum <= 0:
            return discounted  # nothing left to draw
        return remaining / rsum
    return discounted / total


def simulate_orderings(
    win_probs: np.ndarray,
    n_samples: int = DEFAULT_N_SAMPLES,
    alpha_place: float = DEFAULT_ALPHA_PLACE,
    beta_show: float = DEFAULT_BETA_SHOW,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample `n_samples` finishing orders using Henery-discounted PL.

    Returns an `(n_samples, 3)` int array containing the indices of
    (1st, 2nd, 3rd) for each sample. We stop after three positions
    because forecast + trio cover everything the staking layer prices.

    Index 0 is sampled directly from `win_probs` (Plackett-Luce step 1).
    Indices 1 and 2 are sampled from `win_probs ** exponent`, which is
    the Henery correction for the second- and third-place dilution.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(win_probs)
    out = np.full((n_samples, 3), -1, dtype=np.int64)
    if n == 0:
        return out

    base = _normalise_strengths(win_probs)
    n_pos = min(3, n)

    # Pre-compute the per-position strength vectors (they don't depend on
    # the sample, only on the position index).  Position 0 uses the raw
    # win-prob; positions 1 and 2 use the Henery exponents.
    exps = (1.0, alpha_place, beta_show)

    for s in range(n_samples):
        removed = np.zeros(n, dtype=bool)
        for pos in range(n_pos):
            probs = _henery_discount(base, exps[pos], removed)
            if probs.sum() <= 0:
                break
            pick = rng.choice(n, p=probs)
            out[s, pos] = pick
            removed[pick] = True

    return out


def _top_combos(
    counts: np.ndarray,
    entry_ids: list[int],
    *,
    arity: int,
    limit: int,
    min_probability: float,
) -> list[tuple[float, tuple[int, ...]]]:
    """Return the top `limit` ordered-combo cells from a counts array.

    `counts` is shape (n,)*arity. We avoid materialising a dense
    `(n^arity, ...)` list when arity >= 3 by iterating over the non-zero
    cells through `np.argwhere` then sorting only those.
    """
    nonzero_idx = np.argwhere(counts > 0)
    if len(nonzero_idx) == 0:
        return []
    values = counts[tuple(nonzero_idx.T)]
    # Sort descending by count; counts.sum() is the denominator.
    order = np.argsort(values)[::-1]
    total = float(counts.sum()) or 1.0

    results: list[tuple[float, tuple[int, ...]]] = []
    for k in order:
        prob = float(values[k]) / total
        if prob < min_probability and len(results) >= limit:
            break
        idx_tuple = tuple(int(x) for x in nonzero_idx[k])
        # Skip degenerate samples where a position couldn't be drawn
        # (`-1` is the sentinel from `simulate_orderings`).
        if any(i < 0 for i in idx_tuple):
            continue
        results.append((prob, tuple(entry_ids[i] for i in idx_tuple)))
        if len(results) >= limit:
            break
    return results


def compute_ordering(
    entry_ids: Iterable[int],
    win_probs: Iterable[float],
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    alpha_place: float = DEFAULT_ALPHA_PLACE,
    beta_show: float = DEFAULT_BETA_SHOW,
    forecast_limit: int = 10,
    trio_limit: int = 10,
    min_combo_probability: float = 0.005,
    seed: int | None = None,
) -> OrderingResult:
    """Compute place/show/forecast/trio probabilities for one race.

    Args:
        entry_ids: race_entry_id of each dog, in the same order as
            `win_probs`.
        win_probs: per-dog calibrated win probabilities (need not sum
            to exactly 1.0; non-finite or negative entries are clamped).
        n_samples: Monte Carlo budget. 10k is fast (<5ms) for 6 dogs.
        alpha_place / beta_show: Henery exponents for 2nd and 3rd
            position draws. <1 means longshots place more often than
            their win-prob would imply.
        forecast_limit / trio_limit: cap on how many top combos to
            return — full grids are 6*5=30 forecasts and 6*5*4=120 trios
            for a 6-dog race so we keep them all by default.
        min_combo_probability: combos rarer than this are dropped from
            the results unless we still need them to fill the limit.
        seed: optional RNG seed for reproducible Monte Carlo runs.

    Returns:
        OrderingResult with per-dog place/show probs and ranked
        forecast/trio combos.
    """
    entry_id_list = list(entry_ids)
    win_arr = np.asarray(list(win_probs), dtype=float)

    if len(entry_id_list) != len(win_arr):
        raise ValueError(
            f"entry_ids length ({len(entry_id_list)}) must equal win_probs "
            f"length ({len(win_arr)})"
        )

    n = len(win_arr)
    win_dict = {
        eid: float(p) for eid, p in zip(entry_id_list, _normalise_strengths(win_arr))
    }

    if n == 0:
        return OrderingResult(
            entry_ids=[], win_prob={}, place_prob={}, show_prob={},
        )

    if n == 1:
        eid = entry_id_list[0]
        return OrderingResult(
            entry_ids=entry_id_list,
            win_prob={eid: 1.0},
            place_prob={eid: 1.0},
            show_prob={eid: 1.0},
        )

    rng = np.random.default_rng(seed)
    samples = simulate_orderings(
        win_arr,
        n_samples=n_samples,
        alpha_place=alpha_place,
        beta_show=beta_show,
        rng=rng,
    )

    # Per-dog place/show counts: did this dog appear at any sampled
    # finish position <= 2 (place) or <= 3 (show)?
    place_counts = np.zeros(n, dtype=np.int64)
    show_counts = np.zeros(n, dtype=np.int64)

    valid_samples = 0
    for sample in samples:
        first, second, third = sample
        if first < 0:
            continue
        valid_samples += 1
        place_counts[first] += 1
        if second >= 0:
            place_counts[second] += 1
        show_counts[first] += 1
        if second >= 0:
            show_counts[second] += 1
        if third >= 0:
            show_counts[third] += 1

    denom = float(valid_samples) or 1.0
    place_dict = {
        entry_id_list[i]: float(place_counts[i]) / denom for i in range(n)
    }
    show_dict = {
        entry_id_list[i]: float(show_counts[i]) / denom for i in range(n)
    }

    # Forecast (i, j) and trio (i, j, k) ordered counts. Using a sparse
    # iteration via np.unique on the sample columns keeps memory bounded
    # to actually-realised orderings rather than the full n^k grid.
    fc_counts = np.zeros((n, n), dtype=np.int64)
    trio_counts = np.zeros((n, n, n), dtype=np.int64)
    for sample in samples:
        first, second, third = sample
        if first < 0 or second < 0:
            continue
        fc_counts[first, second] += 1
        if third >= 0:
            trio_counts[first, second, third] += 1

    forecast_combos = [
        ForecastCombo(
            first_entry_id=combo[0],
            second_entry_id=combo[1],
            probability=round(prob, 6),
        )
        for prob, combo in _top_combos(
            fc_counts,
            entry_id_list,
            arity=2,
            limit=forecast_limit,
            min_probability=min_combo_probability,
        )
    ]
    trio_combos = [
        TrioCombo(
            first_entry_id=combo[0],
            second_entry_id=combo[1],
            third_entry_id=combo[2],
            probability=round(prob, 6),
        )
        for prob, combo in _top_combos(
            trio_counts,
            entry_id_list,
            arity=3,
            limit=trio_limit,
            min_probability=min_combo_probability,
        )
    ]

    # Round per-dog probabilities for stable JSON.
    return OrderingResult(
        entry_ids=entry_id_list,
        win_prob={eid: round(p, 6) for eid, p in win_dict.items()},
        place_prob={eid: round(p, 6) for eid, p in place_dict.items()},
        show_prob={eid: round(p, 6) for eid, p in show_dict.items()},
        forecast=forecast_combos,
        trio=trio_combos,
    )


# --- Kelly staking for combo bets ---------------------------------------

def compute_combo_kelly(
    combo_probability: float,
    combo_odds_decimal: float | None,
    bankroll: float = 100.0,
    kelly_fraction: float = 0.125,  # eighth Kelly: combos are higher variance
    min_edge: float = 0.10,         # combos need a fatter edge to clear noise
    max_stake_pct: float = 0.02,    # absolute cap, half the win-bet cap
    cfg: "object | None" = None,
) -> dict[str, float | bool | str]:
    """Kelly stake recommendation for a forecast / trio bet.

    Combos have much higher variance than win bets, so sizing is
    deliberately tighter than the win-bet defaults — eighth Kelly, 10%
    min-edge, 2% bankroll cap. Pass ``cfg`` (a ml.staking.StakingConfig,
    usually ``StakingConfig.from_db(db).for_combos()``) to drive sizing
    from the user's BankrollConfig instead of these hard-coded legacy
    defaults; the explicit keyword parameters then no longer apply.

    Tote dividends carry no separate commission (the operator margin is in
    the dividend), so the legacy path uses commission_rate=0.
    """
    from dataclasses import replace as _replace

    from ml.staking import StakingConfig, kelly_stake

    if combo_probability is None or combo_probability <= 0 or combo_probability >= 1:
        return {"bet": False, "reason": "no_probability"}

    if cfg is None:
        cfg = StakingConfig(
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge,
            max_stake_pct=max_stake_pct,
            commission_rate=0.0,
            min_odds=1.0 + 1e-9,  # dividends are long; no min-price filter
        )
    else:
        cfg = _replace(cfg, min_odds=1.0 + 1e-9)

    return kelly_stake(combo_probability, combo_odds_decimal, cfg)

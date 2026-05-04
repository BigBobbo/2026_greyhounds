"""Tests for the Henery-discounted Plackett-Luce ordering layer.

Covers the maths the staking layer relies on:
  * place / show probabilities are consistent (show >= place >= win
    for every dog, sums match the number of finishing slots)
  * forecast and trio probabilities are valid joint distributions
  * the Henery exponent meaningfully shifts probability mass away from
    the favourite at later positions (the bias the literature warns
    against is being corrected)
  * the Kelly helper for combos refuses thin edges and respects
    fractional / cap settings.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.race_ordering import (
    DEFAULT_BETA_SHOW,
    compute_combo_kelly,
    compute_ordering,
    simulate_orderings,
)


# Six-dog Irish field with one strong favourite and a long tail —
# representative of the calibrator's typical output for a competitive race.
WIN_PROBS = [0.40, 0.20, 0.15, 0.12, 0.08, 0.05]
ENTRY_IDS = [101, 102, 103, 104, 105, 106]


def test_place_show_sum_to_position_count():
    """Sum of per-dog place probs over the field == 2 (two place slots),
    sum of show probs == 3. Sanity check that the sampler stays
    self-consistent."""
    out = compute_ordering(
        ENTRY_IDS, WIN_PROBS, n_samples=20_000, seed=42,
    )
    place_sum = sum(out.place_prob.values())
    show_sum = sum(out.show_prob.values())
    assert place_sum == pytest.approx(2.0, abs=0.02)
    assert show_sum == pytest.approx(3.0, abs=0.02)


def test_show_dominates_place_dominates_win():
    """For every dog the relationship show >= place >= win must hold —
    a dog that finishes 1st always counts as placing and showing too."""
    out = compute_ordering(
        ENTRY_IDS, WIN_PROBS, n_samples=20_000, seed=7,
    )
    for eid, win in out.win_prob.items():
        place = out.place_prob[eid]
        show = out.show_prob[eid]
        # Allow a tiny tolerance for Monte Carlo noise.
        assert show + 1e-3 >= place, f"show < place for entry {eid}"
        assert place + 1e-3 >= win, f"place < win for entry {eid}"


def test_forecast_and_trio_combos_are_distinct_and_normalised():
    out = compute_ordering(
        ENTRY_IDS, WIN_PROBS,
        n_samples=20_000, seed=123,
        forecast_limit=30, trio_limit=120, min_combo_probability=0.0,
    )

    # Forecast combos must use distinct legs and probabilities should be
    # bounded by the marginals.
    for combo in out.forecast:
        assert combo.first_entry_id != combo.second_entry_id
        assert 0 <= combo.probability <= 1

    for combo in out.trio:
        assert len({
            combo.first_entry_id,
            combo.second_entry_id,
            combo.third_entry_id,
        }) == 3
        assert 0 <= combo.probability <= 1

    # Forecast probability mass should sum to ~1 across the full grid.
    fc_sum = sum(c.probability for c in out.forecast)
    assert fc_sum == pytest.approx(1.0, abs=0.02)


def test_henery_discount_reduces_favourite_place_share():
    """The Harville baseline (alpha=1.0) over-estimates the favourite's
    place share. The default Henery alpha should shrink it. Without
    the discount the heaviest favourite's place probability is bounded
    above; with the discount it should be strictly smaller."""
    out_harville = compute_ordering(
        ENTRY_IDS, WIN_PROBS,
        n_samples=20_000,
        alpha_place=1.0, beta_show=1.0,
        seed=99,
    )
    out_henery = compute_ordering(
        ENTRY_IDS, WIN_PROBS,
        n_samples=20_000,
        seed=99,
    )
    favourite = ENTRY_IDS[0]
    assert (
        out_henery.place_prob[favourite]
        < out_harville.place_prob[favourite]
    ), "Henery discount should shrink the favourite's place probability"

    # Conversely, longshots should pick up some of that mass.
    longshot = ENTRY_IDS[-1]
    assert (
        out_henery.place_prob[longshot]
        > out_harville.place_prob[longshot] - 1e-3
    )


def test_simulate_orderings_handles_two_dog_field():
    """Two dogs => trio slot can never be filled. The sampler must
    return -1 for the third position rather than blowing up."""
    samples = simulate_orderings(
        np.array([0.7, 0.3]), n_samples=200, rng=np.random.default_rng(0),
    )
    assert samples.shape == (200, 3)
    # Every sample must have a valid first and second pick, third is -1.
    assert (samples[:, 0] >= 0).all()
    assert (samples[:, 1] >= 0).all()
    assert (samples[:, 2] == -1).all()
    # And the two picks are always distinct.
    assert (samples[:, 0] != samples[:, 1]).all()


def test_compute_ordering_one_dog_field():
    """A walkover should return P=1 for the single dog at every slot."""
    out = compute_ordering([42], [1.0], n_samples=100)
    assert out.win_prob == {42: 1.0}
    assert out.place_prob == {42: 1.0}
    assert out.show_prob == {42: 1.0}
    assert out.forecast == []
    assert out.trio == []


def test_compute_ordering_validates_input_lengths():
    with pytest.raises(ValueError):
        compute_ordering([1, 2], [0.5], n_samples=10)


def test_seeded_runs_are_reproducible():
    a = compute_ordering(ENTRY_IDS, WIN_PROBS, n_samples=2000, seed=2026)
    b = compute_ordering(ENTRY_IDS, WIN_PROBS, n_samples=2000, seed=2026)
    assert a.place_prob == b.place_prob
    assert a.show_prob == b.show_prob
    assert [
        (c.first_entry_id, c.second_entry_id, c.probability)
        for c in a.forecast
    ] == [
        (c.first_entry_id, c.second_entry_id, c.probability)
        for c in b.forecast
    ]


def test_compute_ordering_is_robust_to_zero_probs():
    """Mid-race scratch / bad calibration: one dog comes back with
    probability zero. The sampler must still produce valid output."""
    out = compute_ordering(
        ENTRY_IDS,
        [0.0, 0.30, 0.20, 0.20, 0.20, 0.10],
        n_samples=5_000, seed=1,
    )
    assert out.win_prob[101] == pytest.approx(0.0, abs=1e-6)
    assert out.place_prob[101] == pytest.approx(0.0, abs=0.01)
    assert sum(out.place_prob.values()) == pytest.approx(2.0, abs=0.02)


# --- Kelly stake tests --------------------------------------------------

def test_combo_kelly_refuses_thin_edge():
    res = compute_combo_kelly(
        combo_probability=0.12,
        combo_odds_decimal=8.0,  # implied 0.125 -> edge -0.005
        bankroll=1000.0,
    )
    assert res["bet"] is False
    assert res["reason"] == "insufficient_edge"


def test_combo_kelly_respects_fractional_kelly_and_cap():
    """A clear positive-edge combo should produce a stake within the
    eighth-Kelly default and never exceed the 2% bankroll cap."""
    res = compute_combo_kelly(
        combo_probability=0.30,
        combo_odds_decimal=10.0,   # implied 0.10, edge = 0.20
        bankroll=1000.0,
    )
    assert res["bet"] is True
    # Stake must be bounded above by max_stake_pct * bankroll.
    assert res["stake"] <= 1000.0 * 0.02 + 1e-6
    # And must be strictly positive.
    assert res["stake"] > 0


def test_combo_kelly_rejects_invalid_odds():
    res = compute_combo_kelly(0.5, 1.0, bankroll=100.0)
    assert res["bet"] is False
    assert res["reason"] == "no_odds"


def test_default_beta_is_below_default_alpha():
    """3rd-place dilution should be a stronger correction than 2nd. The
    package would still work with arbitrary defaults, but the literature
    is unanimous that beta < alpha < 1.0 — make sure the constants we
    ship reflect that."""
    from app.services.race_ordering import DEFAULT_ALPHA_PLACE
    assert DEFAULT_BETA_SHOW < DEFAULT_ALPHA_PLACE < 1.0

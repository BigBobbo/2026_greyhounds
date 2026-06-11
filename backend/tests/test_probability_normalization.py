"""Probability-normalization parity tests (audit task C5).

Serving normalizes win probabilities per race; the backtest previously did
not, so reported ROI was computed on numbers nobody ever bet with. These
tests pin the unified convention.
"""

import numpy as np
import pytest

from ml.evaluation import compute_betting_metrics, normalize_probs_per_race


def test_normalize_sums_to_one_per_race():
    probs = np.array([0.2, 0.1, 0.1, 0.3, 0.3, 0.2])
    ids = np.array([1, 1, 1, 2, 2, 2])
    out = normalize_probs_per_race(probs, ids)
    assert out[:3].sum() == pytest.approx(1.0)
    assert out[3:].sum() == pytest.approx(1.0)
    # monotonic: ordering within each race unchanged
    assert np.argmax(out[:3]) == np.argmax(probs[:3])
    assert out[0] == pytest.approx(0.5)  # 0.2 / 0.4


def test_normalize_is_idempotent():
    probs = np.array([0.5, 0.3, 0.2])
    ids = np.array([1, 1, 1])
    once = normalize_probs_per_race(probs, ids)
    twice = normalize_probs_per_race(once, ids)
    np.testing.assert_allclose(once, twice)


def _toy_inputs(scale: float):
    """Two 3-dog races; model prob mass scaled by `scale` (sums != 1)."""
    y_true = np.array([1, 0, 0, 0, 1, 0])
    base = np.array([0.5, 0.3, 0.2, 0.25, 0.5, 0.25])
    sp = np.array([2.5, 3.0, 5.0, 4.0, 2.2, 4.5])
    ids = np.array([10, 10, 10, 20, 20, 20])
    return y_true, base * scale, sp, ids


def test_betting_metrics_invariant_to_probability_scale():
    """Same model, different (un)normalization, identical betting results."""
    y, p1, sp, ids = _toy_inputs(1.0)
    _, p2, _, _ = _toy_inputs(0.6)  # sums to 0.6 per race, like raw Platt output
    m1 = compute_betting_metrics(y, p1, sp, ids)
    m2 = compute_betting_metrics(y, p2, sp, ids)
    for key in ("top_pick_pnl", "value_bet_pnl", "value_bet_count", "kelly_pnl"):
        assert m1[key] == m2[key], key


def test_lambdarank_scores_to_proba_sums_to_one_after_calibration():
    from ml.trainers.lambdarank_trainer import LambdaRankTrainer

    trainer = LambdaRankTrainer({}, "classification")

    class FakeCalibrator:
        def predict_proba(self, x):
            # squashes values so the group no longer sums to 1 pre-renorm
            p = 1.0 / (1.0 + np.exp(-x[:, 0] * 0.3))
            return np.column_stack([1 - p, p])

    trainer.calibrator = FakeCalibrator()
    scores = np.array([2.0, 1.0, 0.5, 3.0, 0.0, 0.0])
    groups = [3, 3]
    proba = trainer.scores_to_proba(scores, group_sizes=groups)
    assert proba[:3].sum() == pytest.approx(1.0)
    assert proba[3:].sum() == pytest.approx(1.0)
    # ordering preserved: highest score stays the top pick in each group
    assert np.argmax(proba[:3]) == 0
    assert np.argmax(proba[3:]) == 0


def test_kelly_sim_compounds_with_commission():
    """Audit C10: hand-computed 3-race compounding sequence.

    Each race has a single dominant pick with prob 0.5 at SP 3.0
    (implied 1/3, edge ~0.167 >= 0.05). f* = (2*0.5 - 0.5)/2 = 0.25;
    quarter Kelly = 0.0625 -> capped at 0.05. Commission 2%.

      race 1 (won):  stake 5.00 -> profit 5*2*0.98 = 9.80 -> bankroll 109.80
      race 2 (lost): stake 5.49 -> bankroll 104.31
      race 3 (won):  stake 5.2155 -> profit 10.2224 -> bankroll 114.53
    """
    probs = np.array([0.5, 0.25, 0.5, 0.25, 0.5, 0.25])
    sp = np.array([3.0, 4.0, 3.0, 4.0, 3.0, 4.0])
    ids = np.array([1, 1, 2, 2, 3, 3])
    # outcomes: race1 top pick (idx0) wins; race2 top pick (idx2) loses;
    # race3 top pick (idx4) wins
    y = np.array([1, 0, 0, 1, 1, 0])

    m = compute_betting_metrics(y, probs, sp, ids, commission=0.02)
    assert m["kelly_races"] == 3
    assert m["kelly_final_bankroll"] == pytest.approx(114.53, abs=0.02)
    assert m["kelly_total_staked"] == pytest.approx(5.00 + 5.49 + 5.22, abs=0.02)
    assert m["assumptions"]["commission"] == 0.02
    assert "compounding" in m["assumptions"]["staking"]


def test_zero_commission_reproduces_frictionless():
    y = np.array([1, 0])
    probs = np.array([0.5, 0.5])
    sp = np.array([3.0, 3.0])
    ids = np.array([1, 1])
    m = compute_betting_metrics(y, probs, sp, ids, commission=0.0)
    # tie in probs -> idxmax takes the first, which won: stake 5, profit 10
    assert m["kelly_final_bankroll"] == pytest.approx(110.0, abs=0.01)

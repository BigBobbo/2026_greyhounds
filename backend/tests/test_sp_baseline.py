"""Beat-the-SP gate tests (audit task C11)."""

import numpy as np
import pytest

from ml.evaluation import compute_sp_baseline_metrics


def _market(n_races=200, seed=3):
    """Synthetic races where SP reflects true probabilities with overround."""
    rng = np.random.default_rng(seed)
    y, sp, ids, true_p = [], [], [], []
    for r in range(n_races):
        p = rng.dirichlet(np.ones(6) * 2)
        winner = rng.choice(6, p=p)
        overround = 1.15
        for i in range(6):
            y.append(1.0 if i == winner else 0.0)
            sp.append(1.0 / (p[i] * overround))
            ids.append(r)
            true_p.append(p[i])
    return (np.array(y), np.array(sp), np.array(ids), np.array(true_p))


def test_model_equal_to_devigged_sp_has_zero_delta():
    y, sp, ids, true_p = _market()
    # model probabilities identical to the de-vigged market
    out = compute_sp_baseline_metrics(y, true_p, sp, ids)
    assert "error" not in out
    assert out["log_loss_vs_sp"] == pytest.approx(0.0, abs=1e-3)
    assert out["brier_vs_sp"] == pytest.approx(0.0, abs=1e-3)


def test_better_model_beats_sp_and_noise_does_not():
    y, sp, ids, true_p = _market()
    rng = np.random.default_rng(11)

    # A model that sharpens toward the realized outcome beats the market
    sharper = np.clip(true_p * 0.7 + y * 0.3, 1e-4, 1)
    better = compute_sp_baseline_metrics(y, sharper, sp, ids)
    assert better["beats_sp"] is True
    assert better["log_loss_vs_sp"] < 0
    assert better["model_blend_coef"] is not None and better["model_blend_coef"] > 0

    # A degraded (noisy) model loses to the market
    noisy = np.clip(true_p + rng.normal(0, 0.15, len(true_p)), 1e-4, 1)
    worse = compute_sp_baseline_metrics(y, noisy, sp, ids)
    assert worse["beats_sp"] is False
    assert worse["log_loss_vs_sp"] > 0


def test_handles_missing_sp_gracefully():
    y = np.array([1.0, 0.0, 0.0])
    p = np.array([0.5, 0.3, 0.2])
    sp = np.array([np.nan, np.nan, np.nan])
    ids = np.array([1, 1, 1])
    out = compute_sp_baseline_metrics(y, p, sp, ids)
    assert "error" in out

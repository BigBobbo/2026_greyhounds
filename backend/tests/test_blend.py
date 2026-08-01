"""Tests for the Benter model/market blend."""

import numpy as np
import pytest

from ml.blend import BlendModel, devig_market_probs, fit_blend


def _synthetic_races(n_races=300, seed=7):
    """True probs known; model sees a noisy version, market a less noisy
    one — the classic setting where the blend should weight both."""
    rng = np.random.default_rng(seed)
    model, market, won, rids = [], [], [], []
    for r in range(n_races):
        true = rng.dirichlet(np.ones(6) * 2.0)
        model_p = np.clip(true + rng.normal(0, 0.08, 6), 0.01, None)
        model_p /= model_p.sum()
        market_p = np.clip(true + rng.normal(0, 0.03, 6), 0.01, None)
        market_p /= market_p.sum()
        winner = rng.choice(6, p=true)
        for i in range(6):
            model.append(model_p[i])
            market.append(market_p[i])
            won.append(1 if i == winner else 0)
            rids.append(r)
    return (np.array(model), np.array(market), np.array(won), np.array(rids))


def test_devig_normalizes_within_race():
    odds = np.array([2.0, 4.0, 4.0, 8.0, 8.0, 8.0])
    rids = np.zeros(6)
    p = devig_market_probs(odds, rids)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx((1 / 2.0) / (1 / 2.0 + 2 / 4.0 + 3 / 8.0))


def test_devig_refuses_partial_books():
    odds = np.array([2.0, np.nan, 4.0])
    p = devig_market_probs(odds, np.zeros(3))
    assert np.isnan(p).all()


def test_fit_weights_market_when_market_is_sharper():
    model, market, won, rids = _synthetic_races()
    b = fit_blend(model, market, won, rids)
    assert b.beta > 0.5  # market carries real weight
    assert b.alpha > 0   # model still contributes


def test_blend_probs_sum_to_one_and_fall_back():
    model, market, won, rids = _synthetic_races(n_races=5)
    b = BlendModel(alpha=0.6, beta=0.8)
    out = b.blend(model, market, rids)
    for r in np.unique(rids):
        assert out[rids == r].sum() == pytest.approx(1.0)
    # missing market data for one race -> model-only fallback, still sums to 1
    market2 = market.copy()
    market2[rids == 0] = np.nan
    out2 = b.blend(model, market2, rids)
    assert out2[rids == 0].sum() == pytest.approx(1.0)


def test_blend_improves_log_loss_over_model_alone():
    model, market, won, rids = _synthetic_races(n_races=400, seed=11)
    half = rids < 200
    b = fit_blend(model[half], market[half], won[half], rids[half])
    test = ~half
    blended = b.blend(model[test], market[test], rids[test])
    eps = 1e-12
    ll_model = -np.mean(won[test] * np.log(model[test] + eps))
    ll_blend = -np.mean(won[test] * np.log(blended + eps))
    assert ll_blend < ll_model  # sharper market info must help

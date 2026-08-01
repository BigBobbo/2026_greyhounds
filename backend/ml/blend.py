"""Benter-style model/market probability blend.

Benter's published result — replicated repeatedly since — is that a
fundamental model's probabilities are biased exactly on the bets the model
likes, and that the fix is a second-stage conditional logit combining the
model's log-probability with the market's log-probability:

    strength_i = alpha * log(p_model_i) + beta * log(p_market_i)
    p_blend_i  = softmax(strength) within the race

alpha/beta are fit by maximum likelihood on races where the winner is
known (each race is one multinomial observation). beta > 0 concedes the
market knows things the model doesn't; the model earns whatever alpha
says it deserves. Fitting uses only the validation window, never test.

The market probability source is whatever price column is supplied —
historical SP today, live exchange prices once the odds feed runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BlendModel:
    alpha: float  # weight on log model probability
    beta: float   # weight on log market probability

    def blend(
        self,
        model_probs: np.ndarray,
        market_probs: np.ndarray,
        race_ids: np.ndarray,
    ) -> np.ndarray:
        """Blend and re-normalize within each race. Entries lacking a
        market probability fall back to the model probability."""
        mp = np.clip(np.asarray(model_probs, dtype=float), 1e-9, 1.0)
        kp = np.asarray(market_probs, dtype=float)
        out = np.array(mp)

        strength = np.full(len(mp), np.nan)
        ok = np.isfinite(kp) & (kp > 0)
        strength[ok] = self.alpha * np.log(mp[ok]) + self.beta * np.log(kp[ok])

        for rid in np.unique(race_ids):
            m = race_ids == rid
            s = strength[m]
            if np.isnan(s).any():
                # incomplete market data for the race: keep model probs,
                # renormalized
                p = mp[m]
                out[m] = p / p.sum() if p.sum() > 0 else p
                continue
            e = np.exp(s - s.max())
            out[m] = e / e.sum()
        return out


def devig_market_probs(odds: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Overround-corrected market probabilities from decimal odds: within a
    race, p_i = (1/odds_i) / sum(1/odds). NaN where odds are missing, or
    where a race is missing odds for any runner (a partial book cannot be
    de-vigged honestly)."""
    inv = np.where(
        np.isfinite(np.asarray(odds, dtype=float)) & (np.asarray(odds, dtype=float) > 1.0),
        1.0 / np.asarray(odds, dtype=float),
        np.nan,
    )
    out = np.full(len(inv), np.nan)
    for rid in np.unique(race_ids):
        m = race_ids == rid
        if np.isnan(inv[m]).any():
            continue
        total = inv[m].sum()
        if total > 0:
            out[m] = inv[m] / total
    return out


def fit_blend(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    won: np.ndarray,
    race_ids: np.ndarray,
) -> BlendModel:
    """Fit alpha/beta by conditional-logit maximum likelihood.

    Only races with complete market data and exactly one recorded winner
    contribute. Falls back to alpha=1, beta=0 (pure model) if fitting is
    impossible."""
    from scipy.optimize import minimize

    mp = np.clip(np.asarray(model_probs, dtype=float), 1e-9, 1.0)
    kp = np.asarray(market_probs, dtype=float)
    y = np.asarray(won, dtype=int)
    rids = np.asarray(race_ids)

    races: list[tuple[np.ndarray, np.ndarray, int]] = []
    for rid in np.unique(rids):
        m = rids == rid
        if not np.isfinite(kp[m]).all() or (kp[m] <= 0).any():
            continue
        winners = np.where(y[m] == 1)[0]
        if len(winners) != 1:
            continue
        races.append((np.log(mp[m]), np.log(kp[m]), int(winners[0])))

    if len(races) < 50:
        logger.warning(
            "Blend fit skipped: only %d usable races (need >= 50)", len(races),
        )
        return BlendModel(alpha=1.0, beta=0.0)

    def neg_ll(params: np.ndarray) -> float:
        a, b = params
        total = 0.0
        for lm, lk, w in races:
            s = a * lm + b * lk
            s = s - s.max()
            total += s[w] - np.log(np.exp(s).sum())
        return -total

    res = minimize(neg_ll, x0=np.array([0.7, 0.5]), method="Nelder-Mead")
    a, b = float(res.x[0]), float(res.x[1])
    logger.info(
        "Blend fit on %d races: alpha=%.3f (model), beta=%.3f (market)",
        len(races), a, b,
    )
    return BlendModel(alpha=a, beta=b)

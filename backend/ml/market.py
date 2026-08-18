"""Live market layer: turn captured exchange prices into race-level
market probabilities, and blend them with the model's.

The backtest is unambiguous about where this system's edge lives: the
fitted Benter blend leans on the market (beta ≈ 1.12 against alpha ≈
0.71). Until the Betfair feed existed there was no market probability to
blend at serving time, so every sheet shipped model-only. This module is
the serving-time counterpart of ``ml.blend``, which is fitted offline.

Two deliberate refusals:

  * A partial book is never de-vigged. If any runner in a race lacks a
    price, the overround cannot be computed honestly, and the race falls
    back to model-only rather than silently blending against a market
    probability that is wrong by whatever the missing runners were worth.
  * Stale prices are dropped, not used. A price captured an hour before
    the off is not the price you can bet, and a blend that treats it as
    current will size stakes against a market that has already moved.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ml.blend import BlendModel

logger = logging.getLogger(__name__)

EXCHANGE_BOOKMAKER = "betfair_exchange"
SP_BOOKMAKER = "betfair_sp"

# A snapshot older than this is treated as no price at all.
DEFAULT_MAX_AGE_MINUTES = 45


def latest_prices_for_races(
    db: Any,
    race_ids: list[int],
    bookmaker: str = EXCHANGE_BOOKMAKER,
    as_of: datetime | None = None,
    max_age_minutes: int | None = DEFAULT_MAX_AGE_MINUTES,
) -> dict[int, dict[int, dict[str, Any]]]:
    """Most recent price per (race, dog) from odds_snapshots.

    Returns ``{race_id: {dog_id: {"odds": float, "scraped_at": dt}}}``.
    ``as_of`` bounds the snapshot time so a backtest can ask what was
    showing at sheet-generation time rather than what is showing now.
    """
    from app.models.odds import OddsSnapshot

    if not race_ids:
        return {}

    as_of = as_of or datetime.utcnow()
    q = (
        db.query(OddsSnapshot.race_id, OddsSnapshot.dog_id,
                 OddsSnapshot.odds_decimal, OddsSnapshot.scraped_at)
        .filter(OddsSnapshot.race_id.in_(race_ids))
        .filter(OddsSnapshot.bookmaker == bookmaker)
        .filter(OddsSnapshot.odds_decimal > 1.0)
        .filter(OddsSnapshot.scraped_at <= as_of)
    )
    if max_age_minutes is not None:
        q = q.filter(
            OddsSnapshot.scraped_at >= as_of - timedelta(minutes=max_age_minutes)
        )

    out: dict[int, dict[int, dict[str, Any]]] = {}
    for race_id, dog_id, odds, scraped_at in q.all():
        per_race = out.setdefault(race_id, {})
        current = per_race.get(dog_id)
        if current is None or scraped_at > current["scraped_at"]:
            per_race[dog_id] = {"odds": float(odds), "scraped_at": scraped_at}
    return out


def devig_book(prices: dict[int, float],
               expected_runners: int | None = None) -> dict[int, float] | None:
    """Overround-corrected win probabilities for one race's book.

    Returns None when the book is incomplete — either a runner priced at
    or below evens-on-certainty (1.0), or fewer prices than the race has
    runners. A partial book cannot be de-vigged honestly: the missing
    runners' share of the overround is unknown.
    """
    usable = {d: p for d, p in prices.items() if p and p > 1.0}
    if not usable:
        return None
    if expected_runners is not None and len(usable) < expected_runners:
        return None
    inv = {d: 1.0 / p for d, p in usable.items()}
    total = sum(inv.values())
    if total <= 0:
        return None
    return {d: v / total for d, v in inv.items()}


def blend_race(
    model_probs: dict[int, float],
    market_probs: dict[int, float] | None,
    alpha: float,
    beta: float,
) -> tuple[dict[int, float], bool]:
    """Blend one race's model probabilities with the market's.

    Returns ``(probabilities, blended)`` — ``blended`` is False when the
    market side was unusable and the model's own (renormalised)
    probabilities were passed through untouched, so callers can label the
    sheet honestly instead of implying a blend that did not happen.
    """
    keys = [k for k, v in model_probs.items() if v and v > 0]
    if not keys:
        return dict(model_probs), False

    mp = np.array([model_probs[k] for k in keys], dtype=float)
    if not market_probs or any(
        k not in market_probs or not market_probs[k] > 0 for k in keys
    ):
        renorm = mp / mp.sum() if mp.sum() > 0 else mp
        return dict(zip(keys, renorm.tolist())), False

    kp = np.array([market_probs[k] for k in keys], dtype=float)
    blended = BlendModel(alpha=alpha, beta=beta).blend(
        mp, kp, np.zeros(len(keys), dtype=int),
    )
    return dict(zip(keys, blended.tolist())), True

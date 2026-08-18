"""Serving-time market layer: de-vigging a live book and blending it with
the model's probabilities (ml/market.py). No database, no network."""

import math

from ml.market import blend_race, devig_book


def test_devig_normalises_to_one():
    # A 6-dog book with ~110% overround.
    prices = {1: 3.0, 2: 4.0, 3: 6.0, 4: 8.0, 5: 12.0, 6: 15.0}
    probs = devig_book(prices, expected_runners=6)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    # Ordering is preserved and the shortest price is the most likely.
    assert max(probs, key=probs.get) == 1


def test_devig_refuses_a_partial_book():
    """A missing runner's share of the overround is unknowable, so a
    partial book yields nothing rather than a confidently wrong market
    probability."""
    prices = {1: 3.0, 2: 4.0, 3: 6.0}
    assert devig_book(prices, expected_runners=6) is None
    # Same prices, and the race really is a 3-dog field: fine.
    assert devig_book(prices, expected_runners=3) is not None


def test_devig_drops_impossible_prices():
    assert devig_book({1: 2.0, 2: 1.0}, expected_runners=2) is None
    assert devig_book({}, expected_runners=None) is None


def test_blend_moves_towards_the_market():
    """With beta > alpha the blend should sit closer to the market than to
    the model — the fitted parameters say the market knows more."""
    model = {1: 0.50, 2: 0.30, 3: 0.20}
    market = {1: 0.20, 2: 0.30, 3: 0.50}
    probs, blended = blend_race(model, market, alpha=0.71, beta=1.12)
    assert blended is True
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs[1] < model[1]
    assert probs[3] > model[3]
    assert abs(probs[1] - market[1]) < abs(probs[1] - model[1])


def test_blend_matches_the_closed_form():
    model = {1: 0.6, 2: 0.4}
    market = {1: 0.3, 2: 0.7}
    alpha, beta = 0.71, 1.12
    probs, _ = blend_race(model, market, alpha, beta)
    strength = {
        k: alpha * math.log(model[k]) + beta * math.log(market[k])
        for k in model
    }
    total = sum(math.exp(v) for v in strength.values())
    for k in model:
        assert abs(probs[k] - math.exp(strength[k]) / total) < 1e-9


def test_blend_falls_back_to_model_when_market_is_missing():
    model = {1: 0.5, 2: 0.3, 3: 0.2}
    probs, blended = blend_race(model, None, alpha=0.71, beta=1.12)
    assert blended is False
    assert probs == model

    # A market missing one of the runners is equally unusable: blending
    # only the priced dogs would renormalise against a different field.
    probs, blended = blend_race(model, {1: 0.6, 2: 0.4}, 0.71, 1.12)
    assert blended is False
    assert abs(sum(probs.values()) - 1.0) < 1e-9


# --- database-backed: which snapshot the serving layer actually picks ---

from datetime import datetime, timedelta  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.database import Base  # noqa: E402
from app.models.odds import OddsSnapshot  # noqa: E402
from ml.market import latest_prices_for_races  # noqa: E402


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _snap(db, race_id, dog_id, odds, minutes_ago, bookmaker="betfair_exchange"):
    db.add(OddsSnapshot(
        race_id=race_id, dog_id=dog_id, bookmaker=bookmaker,
        odds_decimal=odds, implied_prob=1 / odds,
        scraped_at=datetime(2026, 8, 18, 20, 0) - timedelta(minutes=minutes_ago),
        is_sp=bookmaker.endswith("_sp"),
    ))


def test_latest_price_wins_and_stale_prices_are_dropped(db):
    as_of = datetime(2026, 8, 18, 20, 0)
    _snap(db, 1, 501, 5.0, minutes_ago=90)   # stale
    _snap(db, 1, 501, 4.0, minutes_ago=30)   # the one we should get
    _snap(db, 1, 502, 6.0, minutes_ago=120)  # stale only
    _snap(db, 1, 503, 3.0, minutes_ago=200)  # stale only
    db.commit()

    prices = latest_prices_for_races(db, [1], as_of=as_of, max_age_minutes=45)
    assert prices[1][501]["odds"] == 4.0
    # 502 and 503 had nothing inside the window at all.
    assert set(prices[1]) == {501}

    # Widen the window and the older prices come back, still latest-wins.
    prices = latest_prices_for_races(db, [1], as_of=as_of, max_age_minutes=300)
    assert prices[1][501]["odds"] == 4.0
    assert prices[1][502]["odds"] == 6.0


def test_prices_after_as_of_are_invisible(db):
    """A backtest asking what was showing at sheet time must not see prices
    captured afterwards — that is lookahead, straight into the stake size."""
    _snap(db, 1, 501, 4.0, minutes_ago=30)
    _snap(db, 1, 501, 2.5, minutes_ago=-10)  # captured 10 min later
    db.commit()

    prices = latest_prices_for_races(
        db, [1], as_of=datetime(2026, 8, 18, 20, 0), max_age_minutes=45,
    )
    assert prices[1][501]["odds"] == 4.0


def test_sp_rows_are_not_mistaken_for_live_prices(db):
    """betfair_sp rows are the post-race settlement price; the serving path
    asks for the exchange book and must not pick them up."""
    _snap(db, 1, 501, 4.0, minutes_ago=10)
    _snap(db, 1, 501, 3.2, minutes_ago=-60, bookmaker="betfair_sp")
    db.commit()

    prices = latest_prices_for_races(
        db, [1], as_of=datetime(2026, 8, 18, 21, 30), max_age_minutes=None,
    )
    assert prices[1][501]["odds"] == 4.0

    sp = latest_prices_for_races(
        db, [1], bookmaker="betfair_sp",
        as_of=datetime(2026, 8, 18, 21, 30), max_age_minutes=None,
    )
    assert sp[1][501]["odds"] == 3.2

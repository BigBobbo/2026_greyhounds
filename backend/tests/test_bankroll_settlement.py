"""API-level tests for bet settlement — the money-handling path."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models.bankroll import BankrollConfig, BetRecord


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    yield TestClient(app), Session
    app.dependency_overrides.clear()


def _seed_bet(Session, bet_type="place", odds=4.0, stake=10.0):
    db = Session()
    cfg = BankrollConfig(initial_bankroll=100.0, current_bankroll=90.0)
    db.add(cfg)
    rec = BetRecord(
        race_entry_id=1, experiment_id=1, dog_name="TESTER",
        bet_type=bet_type, odds_decimal=odds, stake=stake,
        outcome="pending",
    )
    db.add(rec)
    db.commit()
    bet_id = rec.id
    db.close()
    return bet_id


class TestSettlement:
    def test_lost_place_bet_is_a_loss(self, client):
        """Regression: the old UI posted actual_position=2 for 'Lost', which
        the place rule (pos <= 2) settled as a WIN."""
        c, Session = client
        bet_id = _seed_bet(Session, bet_type="place")
        r = c.post(f"/api/bankroll/bets/{bet_id}/settle", json={"result": "lost"})
        assert r.status_code == 200
        assert r.json()["outcome"] == "lost"
        assert r.json()["profit"] == -10.0
        assert r.json()["bankroll"] == 90.0  # nothing returned

    def test_won_place_bet_pays_place_terms(self, client):
        c, Session = client
        bet_id = _seed_bet(Session, bet_type="place", odds=4.0, stake=10.0)
        r = c.post(f"/api/bankroll/bets/{bet_id}/settle", json={"result": "won"})
        assert r.status_code == 200
        # quarter odds: 10 * 3 * 0.25 = 7.50, not the full 30
        assert r.json()["profit"] == pytest.approx(7.5)
        assert r.json()["bankroll"] == pytest.approx(90.0 + 10.0 + 7.5)

    def test_void_refunds_stake(self, client):
        c, Session = client
        bet_id = _seed_bet(Session)
        r = c.post(f"/api/bankroll/bets/{bet_id}/settle", json={"result": "void"})
        assert r.json()["outcome"] == "void"
        assert r.json()["bankroll"] == 100.0

    def test_win_without_odds_rejected_not_booked_as_loss(self, client):
        """Regression: NULL-odds winners were silently booked as LOSSES."""
        c, Session = client
        bet_id = _seed_bet(Session, bet_type="win", odds=None)
        r = c.post(f"/api/bankroll/bets/{bet_id}/settle", json={"result": "won"})
        assert r.status_code == 422
        assert "odds" in r.json()["detail"].lower()

    def test_position_path_still_works_for_win_bets(self, client):
        c, Session = client
        bet_id = _seed_bet(Session, bet_type="win", odds=3.0, stake=10.0)
        r = c.post(f"/api/bankroll/bets/{bet_id}/settle", json={"actual_position": 1})
        assert r.json()["outcome"] == "won"
        assert r.json()["profit"] == pytest.approx(20.0)

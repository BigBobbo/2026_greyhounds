"""Bankroll API flow tests (audit tasks D3/J7)."""

from datetime import date, time

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models.bankroll import BankrollConfig, BetRecord
from app.models.dog import Dog
from app.models.experiment import Experiment
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track

client = TestClient(app)


@pytest.fixture
def seeded(request):
    """One track/race/dog/entry + experiment + fresh bankroll of 100."""
    db = SessionLocal()
    created = []
    try:
        track = Track(name="BetTown", code=f"BT{request.node.name[-3:]}"[:5], active=True)
        db.add(track)
        db.flush()
        race = Race(
            track_id=track.id, race_date=date(2026, 6, 1), race_time=time(19, 0),
            race_number=1, distance_m=525, grade="A3", race_type="flat",
            num_runners=2, status="resulted",
        )
        db.add(race)
        db.flush()
        dog = Dog(name=f"Bet Dog {request.node.name}"[:40])
        db.add(dog)
        db.flush()
        entry = RaceEntry(race_id=race.id, dog_id=dog.id, trap=1, finish_position=1)
        db.add(entry)
        exp = Experiment(
            name=f"exp-{request.node.name}", algorithm="xgboost", target="win_prob",
            status="completed", hyperparameters={}, feature_set=[],
        )
        db.add(exp)
        db.query(BetRecord).delete()
        db.query(BankrollConfig).delete()
        db.add(BankrollConfig(initial_bankroll=100.0, current_bankroll=100.0,
                              kelly_fraction=0.25, min_edge=0.05, max_stake_pct=0.05))
        db.commit()
        created = [entry.id, exp.id]
        yield {"entry_id": entry.id, "experiment_id": exp.id, "db": db}
    finally:
        db.query(BetRecord).delete()
        db.query(BankrollConfig).delete()
        db.commit()
        db.close()


def _place(seeded, stake, odds=3.5):
    return client.post("/api/bankroll/bets", json={
        "race_entry_id": seeded["entry_id"],
        "experiment_id": seeded["experiment_id"],
        "win_probability": 0.4,
        "odds_decimal": odds,
        "stake": stake,
    })


def test_place_bet_deducts_stake(seeded):
    resp = _place(seeded, 10.0)
    assert resp.status_code == 200, resp.text
    assert resp.json()["bankroll"] == pytest.approx(90.0)


def test_rejects_nonpositive_and_oversized_stakes(seeded):
    assert _place(seeded, 0).status_code == 422
    assert _place(seeded, -5).status_code == 422
    resp = _place(seeded, 150.0)
    assert resp.status_code == 422
    assert "exceeds" in resp.json()["detail"]
    # bankroll untouched after rejections
    db = seeded["db"]
    db.expire_all()
    cfg = db.query(BankrollConfig).first()
    assert cfg.current_bankroll == pytest.approx(100.0)


def test_settle_win_credits_and_refuses_without_odds(seeded):
    bet_id = _place(seeded, 10.0, odds=3.0).json()["id"]

    # Strip the odds, then settling as a win must 422 instead of
    # silently recording a loss
    db = seeded["db"]
    record = db.query(BetRecord).filter(BetRecord.id == bet_id).first()
    record.odds_decimal = None
    db.commit()
    resp = client.post(f"/api/bankroll/bets/{bet_id}/settle", json={"actual_position": 1})
    assert resp.status_code == 422

    # Restore odds and settle properly: stake 10 at 3.0 -> profit 20,
    # bankroll 90 + 10 + 20 = 120
    record = db.query(BetRecord).filter(BetRecord.id == bet_id).first()
    record.odds_decimal = 3.0
    db.commit()
    resp = client.post(f"/api/bankroll/bets/{bet_id}/settle", json={"actual_position": 1})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["outcome"] == "won"
    assert body["profit"] == pytest.approx(20.0)
    assert body["bankroll"] == pytest.approx(120.0)


def test_concurrent_placements_lose_no_decrements(seeded):
    import threading

    results = []

    def worker():
        results.append(_place(seeded, 5.0).status_code)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(code == 200 for code in results), results
    db = seeded["db"]
    db.expire_all()
    cfg = db.query(BankrollConfig).first()
    assert cfg.current_bankroll == pytest.approx(100.0 - 8 * 5.0)

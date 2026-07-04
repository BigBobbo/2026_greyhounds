"""Scraping API validation tests (audit tasks I2, I6, E11)."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models.scrape_log import ScrapeLog
from app.models.track import Track

client = TestClient(app)


@pytest.fixture
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.query(ScrapeLog).delete()
        s.commit()
        s.close()


@pytest.fixture
def track(db):
    t = db.query(Track).filter(Track.code == "SCR").first()
    if t is None:
        t = Track(name="ScrapeTown", code="SCR", active=True)
        db.add(t)
        db.commit()
    return t


def test_backfill_rejects_invalid_ranges(track):
    r = client.post("/api/scraping/backfill", json={
        "start_date": "2026-06-10", "end_date": "2026-06-01",
    })
    assert r.status_code == 422

    r = client.post("/api/scraping/backfill", json={
        "start_date": "2020-01-01", "end_date": "2026-06-01",
    })
    assert r.status_code == 422
    assert "366" in r.json()["detail"]

    r = client.post("/api/scraping/backfill", json={
        "start_date": "not-a-date", "end_date": "2026-06-01",
    })
    assert r.status_code == 422


def test_backfill_unknown_track_is_404(track):
    r = client.post("/api/scraping/backfill", json={
        "start_date": "2026-06-01", "end_date": "2026-06-02",
        "track_codes": ["ZZZ"],
    })
    assert r.status_code == 404
    assert "ZZZ" in r.json()["detail"]


def test_backfill_refuses_while_job_running(db, track):
    log = ScrapeLog(
        spider_name="gri", source="test running job", status="running",
        started_at=datetime.utcnow(), heartbeat_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    r = client.post("/api/scraping/backfill", json={
        "start_date": "2026-06-01", "end_date": "2026-06-02",
        "track_codes": [track.code],
    })
    assert r.status_code == 409

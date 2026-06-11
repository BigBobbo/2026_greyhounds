"""Daily failure-visibility digest tests (audit task E12)."""

from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models.scrape_log import ScrapeLog
from app.tasks.scheduler import daily_digest, reap_stale_jobs

client = TestClient(app)


@pytest.fixture
def db():
    s = SessionLocal()
    # Start from a clean scrape_logs table so the counts are deterministic,
    # and reap any stale 'running' experiments left by earlier modules so
    # stale_experiments is ours to control.
    s.query(ScrapeLog).delete()
    s.commit()
    reap_stale_jobs(s)
    try:
        yield s
    finally:
        s.rollback()
        s.query(ScrapeLog).delete()
        s.commit()
        s.close()


def test_daily_digest_shape_and_counts(db):
    now = datetime.utcnow()
    db.add_all([
        ScrapeLog(spider_name="gri", status="success", records_scraped=40, started_at=now),
        ScrapeLog(spider_name="gri", status="partial", records_scraped=10, started_at=now),
        ScrapeLog(
            spider_name="gri", status="failed", track_code="SPK",
            race_date=date(2026, 6, 10), error_message="boom", started_at=now,
        ),
        # Outside the 24h window — ignored entirely.
        ScrapeLog(
            spider_name="gri", status="failed", track_code="GLY",
            race_date=date(2026, 5, 1), started_at=now - timedelta(days=3),
        ),
    ])
    db.commit()

    digest = daily_digest(db)

    assert digest["window_hours"] == 24
    assert digest["scrape_jobs_by_status"] == {"success": 1, "partial": 1, "failed": 1}
    assert digest["total_failed_pairs"] == 1
    assert digest["failed_pairs"][0]["track_code"] == "SPK"
    assert digest["failed_pairs"][0]["race_date"] == "2026-06-10"
    assert digest["failed_pairs"][0]["error"] == "boom"
    assert digest["races_scraped"] == 50
    assert digest["anything_failed"] is True
    assert isinstance(digest["stale_experiments"], int)
    assert "generated_at" in digest


def test_daily_digest_quiet_day(db):
    db.add(
        ScrapeLog(
            spider_name="gri", status="success", records_scraped=12,
            started_at=datetime.utcnow(),
        )
    )
    db.commit()

    digest = daily_digest(db)
    assert digest["scrape_jobs_by_status"] == {"success": 1}
    assert digest["total_failed_pairs"] == 0
    assert digest["stale_experiments"] == 0
    assert digest["anything_failed"] is False


def test_digest_endpoint_returns_same_shape(db):
    r = client.get("/api/scraping/digest")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "generated_at", "window_hours", "scrape_jobs_by_status",
        "total_failed_pairs", "failed_pairs", "races_scraped",
        "stale_experiments", "anything_failed",
    ):
        assert key in body

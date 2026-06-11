"""Shared scrape-job runner lifecycle tests (audit tasks E8 + E7.3).

Uses fake scrape/upsert functions — no network, no Track rows needed. Runs
against the migrated temp test DB; every ScrapeLog row is deleted in
teardown.
"""

from datetime import date, datetime

import pytest

from app.database import SessionLocal
from app.models.scrape_log import ScrapeLog
from scraping.job_runner import run_scrape_job

D1 = date(2026, 6, 1)
D2 = date(2026, 6, 2)


@pytest.fixture
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.rollback()
        s.query(ScrapeLog).delete()
        s.commit()
        s.close()


def _stats(races_new=1):
    return {
        "races_new": races_new,
        "races_updated": 0,
        "entries_new": 6 * races_new,
        "entries_updated": 0,
        "dogs_new": 0,
    }


def test_success_lifecycle(db):
    seen = []

    async def scrape_fn(tc, d, client):
        seen.append((tc, d))
        return [{"race_number": 1}]

    def upsert_fn(session, races, scrape_log_id=None):
        assert scrape_log_id is not None
        return _stats()

    result = run_scrape_job(
        SessionLocal, ["AAA", "BBB"], [D1, D2], scrape_fn, upsert_fn,
        source_desc="runner-test success", delay=0,
    )

    assert result["status"] == "success"
    assert result["races_new"] == 4
    assert result["races_scraped"] == 4
    assert result["entries_new"] == 24
    assert result["failed_pairs"] == []
    assert result["failed_tracks"] == []
    assert seen == [("AAA", D1), ("AAA", D2), ("BBB", D1), ("BBB", D2)]

    log = db.get(ScrapeLog, result["log_id"])
    assert log.status == "success"
    assert log.spider_name == "gri"
    assert log.source == "runner-test success"
    assert log.records_scraped == 4
    assert log.records_new == 4
    assert log.heartbeat_at is not None
    assert log.completed_at is not None
    assert log.error_message is None


def test_single_pair_job_stamps_structured_columns(db):
    async def scrape_fn(tc, d, client):
        return []

    def upsert_fn(session, races, scrape_log_id=None):
        return _stats()

    result = run_scrape_job(
        SessionLocal, ["SPK"], [D1], scrape_fn, upsert_fn,
        source_desc="runner-test single", delay=0,
    )
    log = db.get(ScrapeLog, result["log_id"])
    assert log.track_code == "SPK"
    assert log.race_date == D1


def test_partial_records_failure_rows_and_continues_after_db_error(db):
    async def scrape_fn(tc, d, client):
        if tc == "BAD" and d == D1:
            raise RuntimeError("fetch kaboom")
        return [{"race_number": 1}]

    upserts = []

    def upsert_fn(session, races, scrape_log_id=None):
        upserts.append(1)
        if len(upserts) == 2:
            raise RuntimeError("db kaboom")  # DB error -> rollback + continue
        return _stats()

    result = run_scrape_job(
        SessionLocal, ["BAD", "OKK"], [D1, D2], scrape_fn, upsert_fn,
        source_desc="runner-test partial", delay=0,
    )

    # Iterations: BAD/D1 scrape fails; BAD/D2 upsert #1 ok; OKK/D1 upsert #2
    # raises; OKK/D2 upsert #3 ok — the job kept going after both failures.
    assert result["status"] == "partial"
    assert len(upserts) == 3
    assert result["failed_pairs"] == [("BAD", D1), ("OKK", D1)]
    assert result["races_new"] == 2

    parent = db.get(ScrapeLog, result["log_id"])
    assert parent.status == "partial"
    assert "Failed (track, date)" in parent.error_message
    assert "BAD 2026-06-01" in parent.error_message

    # One ScrapeLog row per failed (track, date) pair (E7.3), with the
    # structured columns set for retry tooling.
    failure_rows = (
        db.query(ScrapeLog)
        .filter(ScrapeLog.status == "failed", ScrapeLog.track_code.isnot(None))
        .all()
    )
    assert {(r.track_code, r.race_date) for r in failure_rows} == {
        ("BAD", D1), ("OKK", D1),
    }
    for row in failure_rows:
        assert row.spider_name == "gri"
        assert "kaboom" in row.error_message
        assert row.completed_at is not None


def test_adopts_existing_log_created_by_endpoint(db):
    pre = ScrapeLog(
        spider_name="gri", source="pre-created", status="running",
        started_at=datetime.utcnow(),
    )
    db.add(pre)
    db.commit()

    async def scrape_fn(tc, d, client):
        return []

    def upsert_fn(session, races, scrape_log_id=None):
        return _stats()

    result = run_scrape_job(
        SessionLocal, ["SPK"], [D1], scrape_fn, upsert_fn,
        source_desc="ignored (log pre-created)", delay=0, log_id=pre.id,
    )
    assert result["log_id"] == pre.id

    db.expire_all()
    assert pre.status == "success"
    assert pre.completed_at is not None
    # Single-pair jobs get the structured columns even on adopted logs.
    assert pre.track_code == "SPK"
    assert pre.race_date == D1

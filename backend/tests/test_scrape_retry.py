"""Retry/backoff + retry-failed endpoint tests (audit task E7).

The transient-retry policy lives in scraping.gri_scraper._fetch_page; the
POST /scraping/retry-failed endpoint re-scrapes the per-(track, date)
failure rows the job runner records.
"""

import asyncio
from datetime import date, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

import scraping.gri_scraper as gri
from app.database import SessionLocal
from app.main import app
from app.models.scrape_log import ScrapeLog
from scraping.gri_scraper import ParseStructureError, ScrapeFetchError, scrape_results

client = TestClient(app)

RACE_DATE = date(2026, 6, 5)

# A valid-but-empty GRI page: carries the track-dropdown anchor, so the
# parser treats it as a quiet no-racing day and returns [].
EMPTY_OK_HTML = (
    "<html><body><form>"
    "<select name='ctl00$ContentPlaceHolder1$ddlTrack'><option>SPK</option></select>"
    "</form></body></html>"
)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(gri, "RETRY_BACKOFF_S", (0.0, 0.0, 0.0))


def _scrape_with(handler):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await scrape_results("SPK", RACE_DATE, c)

    return asyncio.run(go())


# ---------------------------------------------------------------------------
# Backoff behaviour (E7.1)
# ---------------------------------------------------------------------------


def test_retries_after_transient_503_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text=EMPTY_OK_HTML)

    assert _scrape_with(handler) == []
    assert calls["n"] == 2


def test_retries_network_error_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < gri.MAX_FETCH_ATTEMPTS:
            raise httpx.ConnectError("flaky", request=request)
        return httpx.Response(200, text=EMPTY_OK_HTML)

    assert _scrape_with(handler) == []
    assert calls["n"] == gri.MAX_FETCH_ATTEMPTS


def test_404_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404, text="nope")

    with pytest.raises(ScrapeFetchError):
        _scrape_with(handler)
    assert calls["n"] == 1


def test_5xx_exhausts_attempts_then_raises():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="busy")

    with pytest.raises(ScrapeFetchError):
        _scrape_with(handler)
    assert calls["n"] == gri.MAX_FETCH_ATTEMPTS


def test_parse_structure_error_is_not_retried():
    calls = {"n": 0}
    # 200 with race-like text but no parseable markup -> ParseStructureError,
    # raised AFTER the fetch — must not trigger another request.
    broken = "<html><body><p>Race 1 results here, honest</p></body></html>"

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, text=broken)

    with pytest.raises(ParseStructureError):
        _scrape_with(handler)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# POST /scraping/retry-failed (E7.4)
# ---------------------------------------------------------------------------


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


def _failed_row(s, tc, rd, spider="gri", when=None):
    when = when or datetime.utcnow()
    log = ScrapeLog(
        spider_name=spider,
        source="retry-test",
        status="failed",
        track_code=tc,
        race_date=rd,
        error_message="boom",
        started_at=when,
        completed_at=when,
    )
    s.add(log)
    s.commit()
    return log


def test_retry_failed_validates_days(db):
    assert client.post("/api/scraping/retry-failed?days=0").status_code == 422
    assert client.post("/api/scraping/retry-failed?days=999").status_code == 422


def test_retry_failed_404_when_nothing_to_retry(db):
    r = client.post("/api/scraping/retry-failed?days=7")
    assert r.status_code == 404


def test_retry_failed_rescrapes_pairs_and_marks_rows(db, monkeypatch):
    rd = date(2026, 6, 1)
    _failed_row(db, "SPK", rd)
    _failed_row(db, "SPK", rd)  # duplicate pair -> one attempt, both marked
    # Outside the window: must not be attempted or marked.
    _failed_row(db, "GLY", rd, when=datetime.utcnow() - timedelta(days=30))

    seen = []

    async def fake_scrape(tc, race_date, client=None):
        seen.append((tc, race_date))
        return []

    monkeypatch.setattr("scraping.gri_scraper.scrape_results", fake_scrape)

    r = client.post("/api/scraping/retry-failed?days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["pairs_attempted"] == 1
    assert body["pairs_succeeded"] == 1
    assert body["pairs_failed"] == []
    assert body["rows_marked_retried"] == 2
    assert seen == [("SPK", rd)]

    db.expire_all()
    spk_statuses = [
        row.status
        for row in db.query(ScrapeLog).filter(ScrapeLog.track_code == "SPK").all()
    ]
    assert spk_statuses == ["retried", "retried"]
    old = db.query(ScrapeLog).filter(ScrapeLog.track_code == "GLY").one()
    assert old.status == "failed"


def test_retry_failed_keeps_rows_failed_when_rescrape_fails(db, monkeypatch):
    rd = date(2026, 6, 2)
    _failed_row(db, "CRK", rd)

    async def fake_scrape(tc, race_date, client=None):
        raise ScrapeFetchError("still down")

    monkeypatch.setattr("scraping.gri_scraper.scrape_results", fake_scrape)

    r = client.post("/api/scraping/retry-failed?days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["pairs_succeeded"] == 0
    assert body["pairs_failed"] == [f"CRK {rd}"]
    assert body["rows_marked_retried"] == 0

    db.expire_all()
    row = db.query(ScrapeLog).filter(ScrapeLog.track_code == "CRK").one()
    assert row.status == "failed"

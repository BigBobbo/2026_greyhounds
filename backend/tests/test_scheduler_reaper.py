"""Scheduler robustness tests (audit task E9).

Uses the migrated temp test DB via app.database.SessionLocal; every row
created here is deleted again in teardown.
"""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import func

from app.database import SessionLocal
from app.models.experiment import Experiment
from app.models.race import Race
from app.models.scrape_log import ScrapeLog
from app.tasks.scheduler import (
    DUBLIN_TZ,
    _dublin_today,
    reap_stale_jobs,
    scheduler,
    start_scheduler,
    warn_if_results_stale,
)

STALE = datetime.utcnow() - timedelta(hours=2)
FRESH = datetime.utcnow()


@pytest.fixture
def db():
    s = SessionLocal()
    created = {"scrape_logs": [], "experiments": []}
    yield s, created
    s.rollback()
    if created["scrape_logs"]:
        s.query(ScrapeLog).filter(ScrapeLog.id.in_(created["scrape_logs"])).delete(
            synchronize_session=False
        )
    if created["experiments"]:
        s.query(Experiment).filter(Experiment.id.in_(created["experiments"])).delete(
            synchronize_session=False
        )
    s.commit()
    s.close()


def _mk_scrape_log(s, created, **kw):
    log = ScrapeLog(spider_name="gri", source="reaper-test", **kw)
    s.add(log)
    s.commit()
    created["scrape_logs"].append(log.id)
    return log


def _mk_experiment(s, created, **kw):
    exp = Experiment(
        name="reaper-test",
        algorithm="xgboost",
        target="win_prob",
        hyperparameters={},
        feature_set=[],
        **kw,
    )
    s.add(exp)
    s.commit()
    created["experiments"].append(exp.id)
    return exp


def test_reap_stale_jobs_marks_dead_rows_failed_and_leaves_fresh(db):
    s, created = db

    # Clear any stale rows left behind by earlier test modules so the
    # reaped count below is exactly ours.
    reap_stale_jobs(s)

    stale_log = _mk_scrape_log(
        s, created, status="running", started_at=STALE, heartbeat_at=None
    )
    fresh_log = _mk_scrape_log(
        s, created, status="running", started_at=FRESH, heartbeat_at=FRESH
    )
    # Started long ago but heartbeating recently — alive, must survive.
    beating_log = _mk_scrape_log(
        s, created, status="running", started_at=STALE, heartbeat_at=FRESH
    )
    done_log = _mk_scrape_log(
        s, created, status="success", started_at=STALE, completed_at=STALE
    )

    stale_exp = _mk_experiment(
        s, created, status="running", created_at=STALE, heartbeat_at=STALE
    )
    null_hb_exp = _mk_experiment(
        s, created, status="running", created_at=STALE, heartbeat_at=None
    )
    fresh_exp = _mk_experiment(
        s, created, status="running", created_at=FRESH, heartbeat_at=FRESH
    )

    reaped = reap_stale_jobs(s)
    assert reaped == 3  # stale_log, stale_exp, null_hb_exp

    s.expire_all()
    assert stale_log.status == "failed"
    assert "killed by restart or stalled" in stale_log.error_message
    assert stale_log.completed_at is not None

    assert stale_exp.status == "failed"
    assert "killed by restart or stalled" in stale_exp.error_message
    assert null_hb_exp.status == "failed"

    # Fresh / heartbeating / finished rows untouched
    assert fresh_log.status == "running"
    assert beating_log.status == "running"
    assert done_log.status == "success"
    assert fresh_exp.status == "running"

    # Idempotent: second pass reaps nothing.
    assert reap_stale_jobs(s) == 0


def test_start_scheduler_respects_enable_scheduler_setting(monkeypatch, caplog):
    from app.config import settings

    monkeypatch.setattr(settings, "enable_scheduler", False)
    with caplog.at_level("INFO"):
        start_scheduler()
    assert not scheduler.running
    assert any("ENABLE_SCHEDULER" in r.message for r in caplog.records)


def test_warn_if_results_stale_logs_when_behind(db, caplog):
    s, _ = db
    last = (
        s.query(func.max(Race.race_date))
        .filter(Race.status == "resulted")
        .scalar()
    )
    yesterday = _dublin_today() - timedelta(days=1)
    should_warn = last is None or last < yesterday

    with caplog.at_level("WARNING"):
        warn_if_results_stale(s)

    warned = any(
        "scrape-since-last-race-date" in r.message or "backfill" in r.message
        for r in caplog.records
    )
    assert warned == should_warn


def test_dublin_today_uses_irish_timezone():
    assert _dublin_today() == datetime.now(DUBLIN_TZ).date()

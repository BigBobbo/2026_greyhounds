"""Backup service tests (audit task A1)."""

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.config import settings
from app.services.backup_service import (
    _keep_dates,
    apply_retention,
    create_db_snapshot,
    create_models_archive,
    run_backup,
)


@pytest.fixture
def source_db(tmp_path):
    path = tmp_path / "source.db"
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE races (id INTEGER PRIMARY KEY, name TEXT)")
    con.executemany("INSERT INTO races (name) VALUES (?)", [("a",), ("b",), ("c",)])
    con.commit()
    con.close()
    return path


def test_snapshot_is_consistent_and_passes_integrity(source_db, tmp_path):
    snap = tmp_path / "snap.db"
    create_db_snapshot(snap, db_path=source_db)
    con = sqlite3.connect(snap)
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert con.execute("SELECT count(*) FROM races").fetchone()[0] == 3
    con.close()


def test_models_archive_round_trip(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "exp_1.pkl").write_bytes(b"artifact-bytes")
    archive = tmp_path / "models.tar.gz"
    assert create_models_archive(archive, models_dir=models) is True
    import tarfile

    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "models/exp_1.pkl" in names


def test_models_archive_skips_when_empty(tmp_path):
    empty = tmp_path / "none"
    assert create_models_archive(tmp_path / "x.tar.gz", models_dir=empty) is False


def test_retention_keeps_recent_daily_and_weekly():
    # 20 consecutive days ending on a Sunday (2026-06-07)
    from datetime import timedelta

    dates = [date(2026, 5, 19) + timedelta(days=i) for i in range(20)]
    keep = _keep_dates(dates, daily=7, weekly=4)
    # newest 7 days always kept
    for d in dates[-7:]:
        assert d in keep
    # Mondays within the range kept up to 4
    mondays = sorted([d for d in dates if d.weekday() == 0], reverse=True)
    for d in mondays[:4]:
        assert d in keep
    # an old non-Monday is pruned
    assert date(2026, 5, 20) not in keep


class FakeS3:
    """Captures uploads/deletes; serves a canned listing."""

    def __init__(self, keys=()):
        self.keys = list(keys)
        self.uploaded = []
        self.deleted = []

    def upload_file(self, filename, bucket, key):
        assert Path(filename).exists()
        self.uploaded.append(key)
        self.keys.append(key)

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in self.keys if k.startswith(Prefix)]}

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)
        self.keys.remove(Key)


def test_apply_retention_deletes_only_expired(monkeypatch):
    monkeypatch.setattr(settings, "backup_retention_daily", 2)
    monkeypatch.setattr(settings, "backup_retention_weekly", 1)
    keys = [
        "backups/db-20260601.db",      # Monday — kept (weekly)
        "backups/db-20260603.db",      # old Wednesday — pruned
        "backups/db-20260610.db",      # kept (daily)
        "backups/db-20260611.db",      # kept (daily)
        "backups/models-20260603.tar.gz",  # same old date — pruned
    ]
    s3 = FakeS3(keys)
    deleted = apply_retention(s3, "bucket", "backups")
    assert sorted(deleted) == ["backups/db-20260603.db", "backups/models-20260603.tar.gz"]


def test_run_backup_skips_without_credentials(monkeypatch):
    monkeypatch.setattr(settings, "backup_s3_bucket", "")
    summary = run_backup()
    assert summary["skipped"] is True


def test_run_backup_uploads_db_and_applies_retention(monkeypatch, source_db, tmp_path):
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{source_db}")
    monkeypatch.setattr(settings, "backup_s3_bucket", "bucket")
    monkeypatch.setattr(settings, "model_artifacts_dir", str(tmp_path / "no-models"))
    s3 = FakeS3()
    summary = run_backup(s3=s3)
    stamp = date.today().strftime("%Y%m%d")
    assert summary["uploaded"] == [f"backups/db-{stamp}.db"]
    assert summary["db_size_bytes"] > 0
    assert s3.uploaded == [f"backups/db-{stamp}.db"]

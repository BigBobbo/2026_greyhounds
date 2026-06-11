"""Nightly off-site backups of the SQLite database and model artifacts.

Snapshots are produced with ``VACUUM INTO`` (atomic and consistent — never a
raw file copy of a live WAL database) and uploaded to S3-compatible object
storage (Cloudflare R2 / Backblaze B2 / AWS S3). Retention keeps the last
``backup_retention_daily`` daily snapshots plus ``backup_retention_weekly``
weekly (Monday) ones.

Configuration (see .env.example): BACKUP_S3_BUCKET, BACKUP_S3_ENDPOINT_URL,
BACKUP_S3_ACCESS_KEY, BACKUP_S3_SECRET_KEY. When unset, the scheduled job
logs a warning and skips — it never crashes the app.
"""

import logging
import re
import sqlite3
import tarfile
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"(db|models)-(\d{8})\.")


def _db_path() -> Path:
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        raise RuntimeError(f"Backups support sqlite URLs only, got: {url}")
    return Path(url[len("sqlite:///"):])


def create_db_snapshot(snapshot_path: Path, db_path: Path | None = None) -> None:
    """Write a consistent snapshot of the database to ``snapshot_path``."""
    src = db_path or _db_path()
    snapshot_path = Path(snapshot_path)
    if snapshot_path.exists():
        snapshot_path.unlink()
    con = sqlite3.connect(src)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.execute("VACUUM INTO ?", (str(snapshot_path),))
    finally:
        con.close()
    check = sqlite3.connect(snapshot_path)
    try:
        result = check.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        check.close()
    if result != "ok":
        raise RuntimeError(f"Snapshot failed integrity check: {result}")


def create_models_archive(archive_path: Path, models_dir: Path | None = None) -> bool:
    """Tar the model artifacts directory. Returns False when there is nothing to back up."""
    src = models_dir or Path(settings.model_artifacts_dir)
    if not src.is_dir() or not any(src.iterdir()):
        return False
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(src, arcname="models")
    return True


def backup_configured() -> bool:
    return bool(
        settings.backup_s3_bucket
        and settings.backup_s3_access_key
        and settings.backup_s3_secret_key
    )


def _make_s3_client():
    import boto3

    kwargs = {
        "aws_access_key_id": settings.backup_s3_access_key,
        "aws_secret_access_key": settings.backup_s3_secret_key,
    }
    if settings.backup_s3_endpoint_url:
        kwargs["endpoint_url"] = settings.backup_s3_endpoint_url
    if settings.backup_s3_region:
        kwargs["region_name"] = settings.backup_s3_region
    return boto3.client("s3", **kwargs)


def _keep_dates(dates: list[date], daily: int, weekly: int) -> set[date]:
    """Retention: newest ``daily`` dates plus the newest ``weekly`` Mondays."""
    ordered = sorted(set(dates), reverse=True)
    keep = set(ordered[:daily])
    mondays = [d for d in ordered if d.weekday() == 0]
    keep.update(mondays[:weekly])
    return keep


def apply_retention(s3, bucket: str, prefix: str) -> list[str]:
    """Delete objects older than the retention policy. Returns deleted keys."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = resp.get("Contents", [])
    dated: dict[str, date] = {}
    for obj in objects:
        m = _DATE_RE.search(obj["Key"])
        if m:
            dated[obj["Key"]] = datetime.strptime(m.group(2), "%Y%m%d").date()
    keep = _keep_dates(
        list(dated.values()),
        settings.backup_retention_daily,
        settings.backup_retention_weekly,
    )
    deleted = [key for key, d in dated.items() if d not in keep]
    for key in deleted:
        s3.delete_object(Bucket=bucket, Key=key)
    return deleted


def run_backup(s3=None) -> dict:
    """Snapshot DB + models, upload, apply retention. Returns a summary dict."""
    if s3 is None:
        if not backup_configured():
            logger.warning(
                "Backup skipped: BACKUP_S3_BUCKET / BACKUP_S3_ACCESS_KEY / "
                "BACKUP_S3_SECRET_KEY are not configured."
            )
            return {"skipped": True, "reason": "not configured"}
        s3 = _make_s3_client()

    bucket = settings.backup_s3_bucket
    prefix = settings.backup_s3_prefix.rstrip("/")
    stamp = date.today().strftime("%Y%m%d")
    summary: dict = {"skipped": False, "uploaded": [], "deleted": []}

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        snapshot = tmpdir / f"db-{stamp}.db"
        create_db_snapshot(snapshot)
        db_key = f"{prefix}/db-{stamp}.db"
        s3.upload_file(str(snapshot), bucket, db_key)
        summary["uploaded"].append(db_key)
        summary["db_size_bytes"] = snapshot.stat().st_size

        archive = tmpdir / f"models-{stamp}.tar.gz"
        if create_models_archive(archive):
            models_key = f"{prefix}/models-{stamp}.tar.gz"
            s3.upload_file(str(archive), bucket, models_key)
            summary["uploaded"].append(models_key)

    summary["deleted"] = apply_retention(s3, bucket, prefix)
    logger.info(
        "Backup complete: uploaded %s, pruned %d old object(s)",
        summary["uploaded"],
        len(summary["deleted"]),
    )
    return summary


def run_backup_job() -> None:
    """Scheduler entrypoint: run the backup and record the outcome."""
    from app.database import SessionLocal
    from app.models.scrape_log import ScrapeLog

    db = SessionLocal()
    log = ScrapeLog(
        spider_name="backup",
        source="nightly backup",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    try:
        summary = run_backup()
        if summary.get("skipped"):
            log.status = "failed"
            log.error_message = f"skipped: {summary.get('reason')}"
        else:
            log.status = "success"
            log.records_new = len(summary["uploaded"])
            log.error_message = None
        log.completed_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        logger.error("Backup failed: %s", e, exc_info=True)
        try:
            log.status = "failed"
            log.error_message = str(e)[:2000]
            log.completed_at = datetime.utcnow()
            db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()

"""Admin endpoints: database backup and one-off maintenance jobs over HTTPS.

The production host has no reachable shell (SSH is blocked from many
client networks), so anything operational that would normally be a
one-off script run gets a token-gated endpoint here instead:

    curl -H "Authorization: Bearer $ADMIN_BACKUP_TOKEN" \
         https://<host>/api/admin/backup -o backup.db
    curl -X POST -H "Authorization: Bearer $ADMIN_BACKUP_TOKEN" \
         https://<host>/api/admin/backfill-weather

All endpoints are disabled entirely (404) unless ADMIN_BACKUP_TOKEN is
configured. The backup uses SQLite's online backup API, so the copy is
transactionally consistent even while the app is writing.
"""

import logging
import os
import sqlite3
import tempfile
import threading
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_token(authorization: str | None) -> None:
    token = settings.admin_backup_token
    if not token:
        # Endpoints disabled — indistinguishable from absent.
        raise HTTPException(status_code=404, detail="Not found")
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Invalid token")


def _db_path() -> str:
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        raise HTTPException(
            status_code=501,
            detail="Backup endpoint supports SQLite databases only",
        )
    return url[len("sqlite:///"):]


@router.get("/backup")
def download_backup(
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
):
    """Stream a consistent snapshot of the SQLite database."""
    _require_token(authorization)

    src_path = _db_path()
    if not os.path.exists(src_path):
        raise HTTPException(status_code=500, detail="Database file not found")

    fd, tmp_path = tempfile.mkstemp(suffix=".db", prefix="backup_")
    os.close(fd)
    try:
        src = sqlite3.connect(src_path)
        dst = sqlite3.connect(tmp_path)
        with dst:
            src.backup(dst)
        dst.close()
        src.close()
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    background_tasks.add_task(os.remove, tmp_path)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    logger.info("Admin backup downloaded (%d bytes)", os.path.getsize(tmp_path))
    return FileResponse(
        tmp_path,
        media_type="application/octet-stream",
        filename=f"greyhound-backup-{stamp}.db",
    )


# --- Weather backfill (one-off, runs in a background thread) ---

_weather_state: dict = {"status": "idle"}
_weather_lock = threading.Lock()


def _run_weather_backfill() -> None:
    from app.database import SessionLocal
    from ml.weather import backfill_archive

    db = SessionLocal()
    try:
        def progress(msg: str) -> None:
            logger.info("weather backfill: %s", msg)
            _weather_state["last_message"] = msg

        result = backfill_archive(db, log=progress)
        _weather_state.update(status="done", result=result,
                              finished_at=datetime.utcnow().isoformat())
        logger.info("weather backfill finished: %s", result)
    except Exception as e:
        logger.exception("weather backfill failed")
        _weather_state.update(status="failed", error=str(e)[:2000],
                              finished_at=datetime.utcnow().isoformat())
    finally:
        db.close()


@router.post("/backfill-weather", status_code=202)
def start_weather_backfill(authorization: str | None = Header(default=None)):
    """Kick off the historical weather backfill (idempotent upserts).

    Returns immediately; poll GET /admin/backfill-weather for progress.
    Refuses to start a second run while one is in flight.
    """
    _require_token(authorization)
    with _weather_lock:
        if _weather_state.get("status") == "running":
            raise HTTPException(status_code=409, detail="Backfill already running")
        _weather_state.clear()
        _weather_state.update(status="running",
                              started_at=datetime.utcnow().isoformat())
    thread = threading.Thread(target=_run_weather_backfill,
                              name="weather-backfill", daemon=True)
    thread.start()
    return {"status": "running", "poll": "/api/admin/backfill-weather"}


@router.get("/backfill-weather")
def weather_backfill_status(authorization: str | None = Header(default=None)):
    """Progress/result of the last (or current) weather backfill run."""
    _require_token(authorization)
    return dict(_weather_state)

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


# --- Dog-profile enrichment (long-running subprocess job) ---

_dogs_state: dict = {"status": "idle"}
_dogs_lock = threading.Lock()

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _run_dog_backfill() -> None:
    import subprocess
    import sys

    cmd = [sys.executable, os.path.join("scripts", "backfill_dog_profiles.py"),
           "--concurrency", "2"]
    try:
        proc = subprocess.Popen(
            cmd, cwd=_BACKEND_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            line = line.strip()
            if line:
                logger.info("dog backfill: %s", line)
                _dogs_state["last_message"] = line
        code = proc.wait()
        _dogs_state.update(
            status="done" if code == 0 else "failed",
            exit_code=code, finished_at=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.exception("dog backfill crashed")
        _dogs_state.update(status="failed", error=str(e)[:2000],
                           finished_at=datetime.utcnow().isoformat())


@router.post("/backfill-dogs", status_code=202)
def start_dog_backfill(authorization: str | None = Header(default=None)):
    """Enrich every dog missing profile data from its GRI profile page
    (sectionals, running positions, birth date, trainer, pedigree fields
    on entries). Resume-safe; a full first run takes hours, nightly
    top-ups take minutes. Poll GET /admin/backfill-dogs."""
    _require_token(authorization)
    with _dogs_lock:
        if _dogs_state.get("status") == "running":
            raise HTTPException(status_code=409, detail="Backfill already running")
        _dogs_state.clear()
        _dogs_state.update(status="running",
                           started_at=datetime.utcnow().isoformat())
    threading.Thread(target=_run_dog_backfill, name="dog-backfill",
                     daemon=True).start()
    return {"status": "running", "poll": "/api/admin/backfill-dogs"}


@router.get("/backfill-dogs")
def dog_backfill_status(authorization: str | None = Header(default=None)):
    """Progress/result of the last (or current) dog-profile backfill."""
    _require_token(authorization)
    return dict(_dogs_state)


@router.get("/betfair-check")
def betfair_check(authorization: str | None = Header(default=None)):
    """Verify Betfair credentials end to end without returning any of them.

    Reports config presence, login success, market counts, and how many
    markets match scheduled races — enough to confirm the integration is
    live, safe to read over the wire.
    """
    _require_token(authorization)
    from app.database import SessionLocal
    from scraping.betfair_odds import diagnose

    db = SessionLocal()
    try:
        return diagnose(db)
    finally:
        db.close()


@router.post("/capture-odds")
def capture_odds_now(authorization: str | None = Header(default=None)):
    """Run one odds-capture pass immediately (the cron does this on a
    schedule; this is for verifying the first one by hand)."""
    _require_token(authorization)
    from app.database import SessionLocal
    from scraping.betfair_odds import capture_from_settings

    db = SessionLocal()
    try:
        written = capture_from_settings(db)
        if written < 0:
            raise HTTPException(status_code=400,
                                detail="Betfair credentials not configured")
        return {"snapshots_written": written}
    finally:
        db.close()


@router.post("/register-model")
def register_model(authorization: str | None = Header(default=None)):
    """Register the committed retrain model as an experiment (idempotent).
    Runs scripts/register_retrain_model.py and returns its output."""
    import subprocess
    import sys

    _require_token(authorization)
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "register_retrain_model.py")],
        cwd=_BACKEND_ROOT, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(proc.stderr or proc.stdout or "registration failed")[-2000:],
        )
    return {"output": proc.stdout.strip()}

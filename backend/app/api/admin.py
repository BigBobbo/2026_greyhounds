"""Admin endpoints: consistent database backup over HTTPS.

The production database lives on a single hosting volume with no other
export path (the host's SSH is not reachable from every client network),
so this endpoint is the supported way to pull a full, consistent copy:

    curl -H "Authorization: Bearer $ADMIN_BACKUP_TOKEN" \
         https://<host>/api/admin/backup -o backup.db

Disabled entirely unless ADMIN_BACKUP_TOKEN is configured. Uses SQLite's
online backup API, so the copy is transactionally consistent even while
the app is writing.
"""

import logging
import os
import sqlite3
import tempfile
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


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
    token = settings.admin_backup_token
    if not token:
        # Endpoint disabled — indistinguishable from absent.
        raise HTTPException(status_code=404, detail="Not found")
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Invalid token")

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

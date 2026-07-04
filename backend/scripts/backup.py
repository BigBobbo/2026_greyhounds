"""Run an off-site backup right now (same code path as the nightly job).

    python scripts/backup.py

Requires BACKUP_S3_* environment variables — see .env.example.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from app.services.backup_service import backup_configured, run_backup

if __name__ == "__main__":
    if not backup_configured():
        print("BACKUP_S3_* not configured — see .env.example.")
        sys.exit(1)
    summary = run_backup()
    print(summary)
    sys.exit(0 if not summary.get("skipped") else 1)

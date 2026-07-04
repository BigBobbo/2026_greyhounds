"""Restore a database snapshot (and optionally model artifacts) from backup.

    python scripts/restore_backup.py --list
    python scripts/restore_backup.py --key backups/db-20260611.db --out ./data/greyhound.db
    python scripts/restore_backup.py --latest --out ./data/greyhound.db

Restoring over a live database is unsafe: stop the API first, restore, then
start it (alembic will no-op if the snapshot is at head).
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.backup_service import _make_s3_client, backup_configured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--key", help="Object key to restore")
    parser.add_argument("--latest", action="store_true", help="Restore the newest db snapshot")
    parser.add_argument("--out", help="Destination path for the restored file")
    args = parser.parse_args()

    if not backup_configured():
        print("BACKUP_S3_* not configured — see .env.example.")
        sys.exit(1)

    s3 = _make_s3_client()
    bucket = settings.backup_s3_bucket
    prefix = settings.backup_s3_prefix.rstrip("/")

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = sorted(resp.get("Contents", []), key=lambda o: o["Key"])

    if args.list or (not args.key and not args.latest):
        for obj in objects:
            print(f"{obj['Key']}\t{obj['Size'] // 1024} KB\t{obj['LastModified']}")
        return

    key = args.key
    if args.latest:
        db_keys = [o["Key"] for o in objects if "/db-" in o["Key"]]
        if not db_keys:
            print("No db snapshots found.")
            sys.exit(1)
        key = db_keys[-1]

    if not args.out:
        print("--out is required when restoring.")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {key} -> {out}")
    s3.download_file(bucket, key, str(out))

    if out.suffix == ".db":
        con = sqlite3.connect(out)
        try:
            result = con.execute("PRAGMA integrity_check").fetchone()[0]
            n_tables = con.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
        finally:
            con.close()
        print(f"integrity_check: {result}; tables: {n_tables}")
        if result != "ok":
            sys.exit(1)
    print("Restore complete. Start the API against the restored file.")


if __name__ == "__main__":
    main()

"""Reclaim space from the computed_features table — safe, offline admin tool.

Replaces the removed `DELETE /api/features/computed/all` endpoint, which
rebuilt and swapped the live database file and deleted its own backup.

Run this with the API stopped (or accept that writes during the run may be
rejected while VACUUM holds the write lock):

    python scripts/cleanup_computed_features.py            # unversioned rows
    python scripts/cleanup_computed_features.py --all      # every row
    python scripts/cleanup_computed_features.py --vacuum   # also VACUUM after

Never deletes backups; never swaps database files.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.database import engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL computed features (default: only unversioned rows, "
        "preserving version snapshots experiments depend on)",
    )
    parser.add_argument(
        "--vacuum", action="store_true", help="Run VACUUM after deleting to reclaim disk space"
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    scope = "ALL computed features" if args.all else "unversioned computed features"
    if not args.yes:
        answer = input(f"Delete {scope}? Type 'delete' to confirm: ")
        if answer.strip().lower() != "delete":
            print("Aborted.")
            return

    with engine.begin() as conn:
        if args.all:
            result = conn.execute(text("DELETE FROM computed_features"))
        else:
            result = conn.execute(text("DELETE FROM computed_features WHERE version_id IS NULL"))
        print(f"Deleted {result.rowcount} rows.")

    if args.vacuum:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
            conn.execute(text("VACUUM"))
        print("VACUUM complete.")


if __name__ == "__main__":
    main()

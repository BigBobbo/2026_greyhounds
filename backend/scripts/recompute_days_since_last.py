"""Recompute RaceEntry.days_since_last for the whole database.

Heals values corrupted by out-of-order inserts — most commonly track-by-track
backfills, where a dog's races at track B were inserted after its later races
at track A had already had days_since_last computed.

Usage (from backend/):
    python scripts/recompute_days_since_last.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal  # noqa: E402
import app.models  # noqa: F401, E402
from scraping.db_pipeline import recompute_days_since_last  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    db = SessionLocal()
    try:
        changed = recompute_days_since_last(db)
        logger.info("Done: %d entries corrected", changed)
    finally:
        db.close()


if __name__ == "__main__":
    main()

"""Backfill daily weather for every track from the Open-Meteo archive.

Thin CLI wrapper around ``ml.weather.backfill_archive`` (also exposed as a
token-gated admin endpoint for production, where there is no shell).

Usage (from backend/):
    DATABASE_URL=sqlite:///./data/greyhound_local.db \
        python3 scripts/backfill_weather.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, engine, Base  # noqa: E402
from ml.weather import backfill_archive  # noqa: E402


def main() -> None:
    Base.metadata.create_all(engine)  # ensure track_weather exists locally
    db = SessionLocal()
    try:
        backfill_archive(db, log=print)
    finally:
        db.close()


if __name__ == "__main__":
    main()

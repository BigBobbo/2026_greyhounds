"""Backfill daily weather for every track from the Open-Meteo archive.

One archive call per distinct town covers the full race-date range
(multi-year ranges are supported), then rows are stored per (track_id,
date) for the dates that actually have races plus a 2-day lead-in for the
trailing-rain feature.

Usage (from backend/):
    DATABASE_URL=sqlite:///./data/greyhound_local.db \
        python3 scripts/backfill_weather.py
"""

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func, text  # noqa: E402

from app.database import SessionLocal, engine, Base  # noqa: E402
from app.models.race import Race  # noqa: E402
from app.models.track import Track  # noqa: E402
from ml.weather import coords_for_track, fetch_archive, upsert_weather  # noqa: E402


def main() -> None:
    Base.metadata.create_all(engine)  # ensure track_weather exists locally
    db = SessionLocal()

    lo, hi = db.query(func.min(Race.race_date), func.max(Race.race_date)).one()
    if lo is None:
        print("No races in DB; nothing to backfill")
        return
    if isinstance(lo, str):
        lo, hi = date.fromisoformat(lo), date.fromisoformat(hi)
    start = lo - timedelta(days=2)
    end = min(hi, date.today() - timedelta(days=3))  # archive lags a few days
    print(f"Backfilling weather {start} .. {end}")

    tracks = db.query(Track).all()
    by_coords: dict[tuple, list[Track]] = {}
    for t in tracks:
        c = coords_for_track(t)
        if c is None:
            print(f"  ! no coordinates for track {t.code} {t.name!r} (loc={t.location!r}) — skipped")
            continue
        by_coords.setdefault(c, []).append(t)

    # Only store rows for dates each track actually raced (plus lead-in days
    # feeding the trailing-rain sums — handled inside fetch rows already).
    race_dates: dict[int, set] = {}
    for tid, rd in db.query(Race.track_id, Race.race_date).distinct():
        if isinstance(rd, str):
            rd = date.fromisoformat(rd)
        race_dates.setdefault(tid, set()).add(rd)

    total = 0
    for coords, town_tracks in by_coords.items():
        print(f"fetching {coords} for {[t.code for t in town_tracks]}")
        rows = fetch_archive(coords[0], coords[1], start, end)
        for t in town_tracks:
            dates = race_dates.get(t.id)
            if not dates:
                continue
            n = upsert_weather(db, t.id, rows, only_dates=dates)
            total += n
        db.commit()

    cnt = db.execute(text("SELECT COUNT(*) FROM track_weather")).scalar()
    print(f"DONE: inserted {total} new rows; table now {cnt} rows")
    db.close()


if __name__ == "__main__":
    main()

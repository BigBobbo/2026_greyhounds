"""Backfill dead columns from GRI dog-profile pages.

One request per dog fetches its full career form (sectional times, running
positions, weights, going allowances) plus birth date, sex, colour,
trainer and owner. Writes via scraping.dog_profile_scraper.apply_profile.

Resume-safe: a dog with ``sex`` already set is considered done (sex is
always present on a profile page). Dogs are processed most-recently-active
first so the runners we'll actually predict on get their sectionals first.

Usage (from backend/):
    DATABASE_URL=sqlite:///./data/greyhound_local.db \
        python3 scripts/backfill_dog_profiles.py [--concurrency 3] [--limit N]
"""

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
from sqlalchemy import func, text  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.models.dog import Dog  # noqa: E402
from app.models.race import Race  # noqa: E402
from app.models.race_entry import RaceEntry  # noqa: E402
from scraping.dog_profile_scraper import apply_profile, scrape_dog_profile  # noqa: E402
from scraping.gri_scraper import DEFAULT_HEADERS, ScrapeError  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("backfill_dog_profiles")


async def main(concurrency: int, limit: int | None, delay: float) -> None:
    db = SessionLocal()

    # Ensure the running_positions column exists even on DBs created via
    # metadata before the migration ran.
    try:
        db.execute(text("ALTER TABLE race_entries ADD COLUMN running_positions VARCHAR"))
        db.commit()
    except Exception:
        db.rollback()  # already exists

    # Most-recently-active first; skip dogs already done (sex set).
    rows = (
        db.query(Dog.id, Dog.name, func.max(Race.race_date).label("last_race"))
        .join(RaceEntry, RaceEntry.dog_id == Dog.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(Dog.sex.is_(None))
        .group_by(Dog.id)
        .order_by(func.max(Race.race_date).desc())
        .all()
    )
    if limit:
        rows = rows[:limit]
    total = len(rows)
    print(f"{total} dogs to backfill", flush=True)

    sem = asyncio.Semaphore(concurrency)
    done = failed = entries_updated = 0
    lock = asyncio.Lock()
    t0 = time.time()

    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30,
        limits=httpx.Limits(max_connections=concurrency),
    ) as client:

        async def one(dog_id: int, name: str):
            nonlocal done, failed, entries_updated
            async with sem:
                try:
                    profile = await scrape_dog_profile(name, client)
                except ScrapeError as e:
                    async with lock:
                        failed += 1
                        logger.warning("profile failed %s: %s", name, e)
                    return
                await asyncio.sleep(delay)
            async with lock:
                dog = db.query(Dog).get(dog_id)
                stats = apply_profile(db, dog, profile)
                if not dog.sex:
                    dog.sex = "?"  # mark visited even if header was sparse
                entries_updated += stats["entries_updated"]
                done += 1
                if done % 100 == 0:
                    db.commit()
                    rate = done / max(time.time() - t0, 1)
                    eta_h = (total - done) / max(rate, 0.01) / 3600
                    print(
                        f"{done}/{total} dogs ({failed} failed, "
                        f"{entries_updated} entries enriched, "
                        f"{rate:.1f}/s, ~{eta_h:.1f}h left)",
                        flush=True,
                    )

        CHUNK = 500
        for i in range(0, len(rows), CHUNK):
            await asyncio.gather(*(one(r.id, r.name) for r in rows[i:i + CHUNK]))
            db.commit()

    db.commit()
    print(
        f"DONE: {done}/{total} dogs, {failed} failed, "
        f"{entries_updated} entries enriched",
        flush=True,
    )

    # Final pass: adjusted_time for every entry whose race now has an
    # allowance (profiles only touched the scraped dog's own entries).
    n = db.execute(text("""
        UPDATE race_entries
        SET adjusted_time = ROUND(finish_time + (
            SELECT going_allowance FROM races WHERE races.id = race_entries.race_id
        ), 3)
        WHERE adjusted_time IS NULL AND finish_time IS NOT NULL
          AND (SELECT going_allowance FROM races
               WHERE races.id = race_entries.race_id) IS NOT NULL
    """)).rowcount
    db.commit()
    print(f"adjusted_time backfilled for {n} additional entries", flush=True)
    db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.15,
                    help="post-request pause per worker (politeness)")
    args = ap.parse_args()
    asyncio.run(main(args.concurrency, args.limit, args.delay))

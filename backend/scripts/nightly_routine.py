"""The daily prediction/scoring cycle — safe to run on a fresh container.

Steps (each idempotent, each survives the previous day's container being
reclaimed):

  1. Bootstrap: if the local DB is missing, rebuild it from the committed
     data_mirror dumps and re-register the retrain model experiment.
  2. Gap scrape: pull GRI results for every date between the last stored
     result and yesterday (self-heals any container gap straight from
     GRI), then today's cards.
  3. Enrich: dog-profile backfill for newly-seen dogs; weather archive
     top-up + today's forecast.
  4. Score yesterday's committed sheet (writes the scorecard markdown and
     appends the running record CSV).
  5. Generate today's prediction sheet.

Prints a compact summary; exits non-zero on a hard failure so the calling
agent investigates rather than reporting success.

Usage (from backend/):
    DATABASE_URL=sqlite:///./data/greyhound_local.db \
        python3 scripts/nightly_routine.py [--date 2026-08-02]
"""

import argparse
import asyncio
import os
import subprocess
import sys
from datetime import date as date_cls
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BACKEND)
PREDICTIONS_DIR = os.path.join(REPO, "docs", "predictions")
DB_PATH = os.path.join(BACKEND, "data", "greyhound_local.db")
MAX_GAP_DAYS = 45

DUBLIN = ZoneInfo("Europe/Dublin")

summary_lines: list[str] = []


def note(msg: str) -> None:
    print(f"[nightly] {msg}", flush=True)
    summary_lines.append(msg)


def run_script(rel_path: str, *args: str, timeout: int = 3600) -> None:
    subprocess.run(
        [sys.executable, rel_path, *args],
        cwd=BACKEND, check=True, timeout=timeout,
    )


def ensure_db() -> None:
    if not os.path.exists(DB_PATH):
        note("local DB missing — rebuilding from data_mirror dumps")
        run_script(os.path.join(REPO, "data_mirror", "load_mirror.py"))
    run_script(os.path.join("scripts", "register_retrain_model.py"))


def experiment_id() -> int:
    from app.database import SessionLocal
    from app.models.experiment import Experiment

    db = SessionLocal()
    try:
        exp = (db.query(Experiment)
               .filter(Experiment.name == "retrain-2026-08-01").first())
        if exp is None:
            raise SystemExit("retrain experiment not registered")
        return exp.id
    finally:
        db.close()


def scrape_gap(today: date_cls) -> None:
    from sqlalchemy import func

    from app.database import SessionLocal
    from app.models.race import Race
    from app.models.race_entry import RaceEntry
    from app.models.track import Track
    from scraping.db_pipeline import upsert_race_results
    from scraping.gri_scraper import ScrapeError, scrape_results

    db = SessionLocal()
    last_result = (
        db.query(func.max(Race.race_date))
        .join(RaceEntry).filter(RaceEntry.finish_position == 1)
        .scalar()
    )
    if isinstance(last_result, str):
        last_result = date_cls.fromisoformat(last_result)
    start = (last_result + timedelta(days=1)) if last_result else today
    if (today - start).days > MAX_GAP_DAYS:
        raise SystemExit(
            f"results gap {start}..{today} exceeds {MAX_GAP_DAYS} days — "
            "refresh the data_mirror dumps instead of scraping the gap"
        )
    dates = [start + timedelta(days=i) for i in range((today - start).days + 1)]
    tracks = db.query(Track).filter(Track.active.is_(True)).all()

    async def run() -> None:
        fetched = failed = 0
        for d in dates:
            for t in tracks:
                try:
                    races = await scrape_results(t.code, d)
                    if races:
                        upsert_race_results(db, races)
                        fetched += 1
                except ScrapeError:
                    failed += 1
                except Exception as e:
                    failed += 1
                    print(f"  ! {t.code} {d}: {e}", file=sys.stderr)
                await asyncio.sleep(2.0)
            db.commit()
        note(f"scraped {start}..{today}: {fetched} track-days with races, "
             f"{failed} fetch failures")

    asyncio.run(run())
    db.close()


def enrich(today: date_cls) -> None:
    try:
        run_script(os.path.join("scripts", "backfill_dog_profiles.py"),
                   "--concurrency", "2", timeout=1800)
        note("dog-profile enrichment: up to date")
    except Exception as e:
        note(f"dog-profile enrichment FAILED (non-fatal): {e}")

    from app.database import SessionLocal
    from ml.weather import backfill_archive, ensure_weather_for_date

    db = SessionLocal()
    try:
        backfill_archive(db)
        added = ensure_weather_for_date(db, today)
        note(f"weather: archive topped up, {added} forecast rows for {today}")
    except Exception as e:
        note(f"weather update FAILED (non-fatal): {e}")
    finally:
        db.close()


def score_yesterday(today: date_cls) -> None:
    yesterday = today - timedelta(days=1)
    csv_path = os.path.join(PREDICTIONS_DIR, f"{yesterday}.csv")
    card_path = os.path.join(PREDICTIONS_DIR, f"{yesterday}-scorecard.md")
    if not os.path.exists(csv_path):
        note(f"no sheet for {yesterday} — nothing to score")
        return
    if os.path.exists(card_path):
        note(f"{yesterday} already scored")
        return

    from scripts.score_prediction_sheet import score

    text, summary = score(csv_path)
    with open(card_path, "w") as f:
        f.write(
            f"# Scorecard — prediction sheet {yesterday}\n\n"
            f"Scored {today} by the nightly routine with the pre-registered\n"
            f"methodology in `backend/scripts/score_prediction_sheet.py`.\n\n"
            "```\n" + text + "\n```\n"
        )

    record = os.path.join(PREDICTIONS_DIR, "record.csv")
    header = ("date,races,hits,hit_rate,expected_hit_rate,log_loss,"
              "log_loss_uniform,brier,brier_uniform,races_missing_results\n")
    line = (f"{yesterday},{summary.get('races', 0)},{summary.get('hits', 0)},"
            f"{summary.get('hit_rate', '')},{summary.get('expected_hit_rate', '')},"
            f"{summary.get('log_loss', '')},{summary.get('log_loss_uniform', '')},"
            f"{summary.get('brier', '')},{summary.get('brier_uniform', '')},"
            f"{summary.get('races_missing_results', 0)}\n")
    new = not os.path.exists(record)
    with open(record, "a") as f:
        if new:
            f.write(header)
        f.write(line)
    if summary.get("races"):
        note(f"scored {yesterday}: {summary['hits']}/{summary['races']} "
             f"top picks ({summary['hit_rate']:.1%} vs expected "
             f"{summary['expected_hit_rate']:.1%}), log loss "
             f"{summary['log_loss']} vs uniform {summary['log_loss_uniform']}")
    else:
        note(f"scored {yesterday}: no strict-cohort results yet")


def generate_today(today: date_cls, exp_id: int) -> None:
    md_path = os.path.join(PREDICTIONS_DIR, f"{today}.md")
    if os.path.exists(md_path):
        note(f"sheet for {today} already exists")
        return

    from app.database import SessionLocal
    from app.models.race import Race

    db = SessionLocal()
    n_cards = (db.query(Race)
               .filter(Race.race_date == today, Race.status == "scheduled")
               .count())
    db.close()
    if not n_cards:
        note(f"no cards published yet for {today} — sheet not generated")
        return
    run_script(os.path.join("scripts", "generate_prediction_sheet.py"),
               "--experiment-id", str(exp_id), "--date", str(today),
               "--out-dir", PREDICTIONS_DIR, timeout=3600)
    note(f"generated sheet for {today}: {n_cards} scheduled races")


def main(today: date_cls) -> None:
    note(f"nightly routine for {today} "
         f"(now {datetime.now(DUBLIN):%H:%M} Dublin)")
    ensure_db()
    exp_id = experiment_id()
    scrape_gap(today)
    enrich(today)
    score_yesterday(today)
    generate_today(today, exp_id)
    print("\n=== NIGHTLY SUMMARY ===")
    for line in summary_lines:
        print(f"- {line}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=lambda s: date_cls.fromisoformat(s),
                    default=None)
    args = ap.parse_args()
    main(args.date or datetime.now(DUBLIN).date())

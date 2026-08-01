"""Daily scheduled prediction tracker.

Each ``ModelSchedule`` row drives one APScheduler cron job. When the job
fires we:

  1. Optionally scrape upcoming race cards for ``today .. today + N`` so any
     missing meetings are filled in before predicting.
  2. Iterate every ``status="scheduled"`` race in that date window and run
     ``predict_race`` against the schedule's experiment, persisting the
     predictions so they're locked in *before* the race runs.
  3. Write a ``ScheduledPredictionRun`` row capturing what happened so the
     UI can surface "last run" and any failures without polling APScheduler.

We deliberately do NOT place hypothetical bets here — the user's product
direction is "track predictions only" for Phase 1. Performance metrics
(accuracy, log-loss, calibration) are derived on read by joining
``predictions`` ↔ ``race_entries.finish_position`` once results have been
scraped by the existing daily results jobs.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.experiment import Experiment
from app.models.race import Race
from app.models.schedule import ModelSchedule, ScheduledPredictionRun
from app.models.track import Track
from app.services.prediction_service import predict_race, save_predictions

logger = logging.getLogger(__name__)


# None -> predict_race sizes stakes off the real BankrollConfig ledger
# balance (the whole point of the canonical staking module).
DEFAULT_PREDICTION_BANKROLL = None


def _scrape_upcoming_window(db: Session, start_date: date, days_ahead: int) -> dict[str, Any]:
    """Synchronously scrape upcoming race cards for the given window.

    Mirrors the brute-force scrape-date logic in app.api.scraping but runs
    inline (the caller is already on a background thread). Returns a dict
    of stats for logging.
    """
    from scraping.gri_scraper import scrape_card, DEFAULT_HEADERS
    from scraping.db_pipeline import upsert_race_results

    tracks = db.query(Track).filter(Track.active.is_(True)).all()
    track_codes = [t.code for t in tracks]
    if not track_codes:
        return {"races_new": 0, "races_updated": 0, "tracks_failed": []}

    stats = {"races_new": 0, "races_updated": 0, "tracks_failed": []}

    async def _run():
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30
        ) as client:
            for offset in range(days_ahead + 1):
                race_date_val = start_date + timedelta(days=offset)
                for tc in track_codes:
                    try:
                        races = await scrape_card(tc, race_date_val, client)
                    except Exception as e:
                        logger.warning("Card scrape %s %s failed: %s", tc, race_date_val, e)
                        stats["tracks_failed"].append(f"{tc}:{race_date_val}")
                        await asyncio.sleep(0.5)
                        continue

                    if not races:
                        await asyncio.sleep(0.5)
                        continue

                    # Each track's upsert gets its own short-lived session so
                    # a failure on one track can't poison the whole job.
                    db_local = SessionLocal()
                    try:
                        upsert = upsert_race_results(db_local, races)
                        stats["races_new"] += upsert["races_new"]
                        stats["races_updated"] += upsert["races_updated"]
                    except Exception as e:
                        logger.error("Upsert failed for %s %s: %s", tc, race_date_val, e)
                        db_local.rollback()
                    finally:
                        db_local.close()

                    await asyncio.sleep(0.5)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()

    return stats


def run_schedule_job(schedule_id: int, trigger: str = "scheduled") -> int:
    """Execute one scheduled prediction job.

    Creates a ``ScheduledPredictionRun`` row, runs the scrape→predict
    pipeline, and updates the row with results. Returns the run id so the
    caller (manual-trigger API endpoint) can return it.

    Safe to call from any thread; manages its own DB session.
    """
    db = SessionLocal()
    run_id: int | None = None
    try:
        schedule = db.query(ModelSchedule).filter(ModelSchedule.id == schedule_id).first()
        if not schedule:
            logger.error("Schedule %d not found, skipping run", schedule_id)
            return -1
        if not schedule.enabled:
            logger.info("Schedule %d disabled, skipping run", schedule_id)
            return -1

        experiment = (
            db.query(Experiment).filter(Experiment.id == schedule.experiment_id).first()
        )
        if not experiment or experiment.status != "completed":
            logger.warning(
                "Schedule %d points at experiment %d which is not completed; skipping",
                schedule_id, schedule.experiment_id,
            )
            return -1

        run_date = date.today()
        run = ScheduledPredictionRun(
            model_schedule_id=schedule.id,
            run_date=run_date,
            status="running",
            trigger=trigger,
            started_at=datetime.utcnow(),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id

        scrape_stats: dict[str, Any] = {}
        if schedule.scrape_upcoming:
            try:
                scrape_stats = _scrape_upcoming_window(
                    db, run_date, schedule.predict_days_ahead
                )
            except Exception as e:
                logger.error("Scheduled card scrape failed: %s", e)
                scrape_stats = {"error": str(e)}

        # Weather for the prediction window: fill today's (and the next
        # days') per-track rows from the forecast API so the weather
        # features are populated at serve time exactly like in training.
        try:
            from ml.weather import ensure_weather_for_date

            for offset in range(schedule.predict_days_ahead + 1):
                ensure_weather_for_date(db, run_date + timedelta(days=offset))
        except Exception as e:
            logger.warning("Weather fetch for prediction window failed: %s", e)

        end_date = run_date + timedelta(days=schedule.predict_days_ahead)
        scheduled_races = (
            db.query(Race)
            .filter(Race.status == "scheduled")
            .filter(Race.race_date >= run_date)
            .filter(Race.race_date <= end_date)
            .order_by(Race.race_date, Race.race_number)
            .all()
        )

        races_predicted = 0
        races_skipped = 0
        predictions_written = 0
        errors: list[str] = []

        for race in scheduled_races:
            try:
                preds = predict_race(
                    db,
                    schedule.experiment_id,
                    race.id,
                    bankroll=DEFAULT_PREDICTION_BANKROLL,
                )
                if preds:
                    written = save_predictions(db, preds)
                    races_predicted += 1
                    predictions_written += written
                else:
                    races_skipped += 1
            except Exception as e:
                # One race's failure (e.g. missing features) shouldn't stop
                # the whole batch. Log and continue.
                logger.warning(
                    "Predict failed exp=%d race=%d: %s",
                    schedule.experiment_id, race.id, e,
                )
                races_skipped += 1
                if len(errors) < 10:
                    errors.append(f"race={race.id}: {type(e).__name__}: {e}")

        run.races_predicted = races_predicted
        run.races_skipped = races_skipped
        run.predictions_written = predictions_written
        run.finished_at = datetime.utcnow()

        scrape_summary = ""
        if scrape_stats:
            if "error" in scrape_stats:
                scrape_summary = f"scrape_error={scrape_stats['error']}"
            else:
                scrape_summary = (
                    f"scrape_new={scrape_stats.get('races_new', 0)} "
                    f"scrape_updated={scrape_stats.get('races_updated', 0)}"
                )
                if scrape_stats.get("tracks_failed"):
                    scrape_summary += (
                        f" scrape_failed={','.join(scrape_stats['tracks_failed'][:5])}"
                    )

        if errors:
            run.status = "partial" if races_predicted > 0 else "failed"
            run.error_message = (scrape_summary + "; " if scrape_summary else "") + " | ".join(errors)
        else:
            run.status = "success"
            run.error_message = scrape_summary or None

        db.commit()
        logger.info(
            "Schedule %d run %d done: status=%s races_predicted=%d races_skipped=%d",
            schedule_id, run_id, run.status, races_predicted, races_skipped,
        )
        return run_id

    except Exception as e:
        logger.error(
            "Schedule %d run crashed: %s\n%s",
            schedule_id, e, traceback.format_exc(),
        )
        if run_id:
            try:
                run = db.query(ScheduledPredictionRun).filter(
                    ScheduledPredictionRun.id == run_id
                ).first()
                if run:
                    run.status = "failed"
                    run.error_message = f"{type(e).__name__}: {e}"
                    run.finished_at = datetime.utcnow()
                    db.commit()
            except Exception:
                db.rollback()
        return run_id or -1
    finally:
        db.close()


def compute_performance(db: Session, schedule_id: int, days: int = 30) -> dict[str, Any]:
    """Aggregate accuracy and calibration for a schedule over the last N days.

    Joins ``predictions`` for the schedule's experiment against the entries'
    finish positions on resulted races. We compute:

      - races_evaluated: races we predicted on that have since been resulted
      - top1_accuracy:    fraction of races where the model's top pick won
      - top3_hit_rate:    fraction of races where the winner was in our top 3
      - mean_log_loss:    -mean(log(p_winner)), winner's predicted win prob
      - calibration:      10 buckets of (mean_predicted, empirical_win_rate)

    All metrics are derived — no separate "settlement" job is required.
    """
    from collections import defaultdict
    import math

    from app.models.prediction import Prediction
    from app.models.race_entry import RaceEntry

    schedule = db.query(ModelSchedule).filter(ModelSchedule.id == schedule_id).first()
    if not schedule:
        raise ValueError(f"Schedule {schedule_id} not found")

    cutoff = date.today() - timedelta(days=days)

    rows = (
        db.query(
            Prediction.id.label("pred_id"),
            Prediction.race_entry_id,
            Prediction.win_probability,
            RaceEntry.race_id,
            RaceEntry.finish_position,
            Race.race_date,
            Race.status.label("race_status"),
        )
        .join(RaceEntry, Prediction.race_entry_id == RaceEntry.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(Prediction.experiment_id == schedule.experiment_id)
        .filter(Race.race_date >= cutoff)
        .filter(Race.status == "resulted")
        .all()
    )

    by_race: dict[int, list[Any]] = defaultdict(list)
    for r in rows:
        by_race[r.race_id].append(r)

    races_evaluated = 0
    top1_correct = 0
    top3_correct = 0
    log_losses: list[float] = []
    calibration_bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]

    for race_id, entries in by_race.items():
        # Need a winner to evaluate. Skip races where no entry has finish=1
        # (e.g. void/abandoned races or partial results).
        winner = next((e for e in entries if e.finish_position == 1), None)
        if not winner:
            continue
        races_evaluated += 1

        # Sort by predicted win probability descending; treat None as 0.
        ranked = sorted(
            entries, key=lambda e: e.win_probability or 0.0, reverse=True
        )
        if ranked[0].race_entry_id == winner.race_entry_id:
            top1_correct += 1
        if any(e.race_entry_id == winner.race_entry_id for e in ranked[:3]):
            top3_correct += 1

        # Log loss on the winner's probability (clipped to avoid -inf).
        p = max(min(winner.win_probability or 0.0, 1 - 1e-6), 1e-6)
        log_losses.append(-math.log(p))

        # Calibration buckets — every entry contributes one observation.
        for e in entries:
            if e.win_probability is None:
                continue
            bucket = min(int(e.win_probability * 10), 9)
            won = 1 if e.finish_position == 1 else 0
            calibration_bins[bucket].append((e.win_probability, won))

    calibration = []
    for i, bin_obs in enumerate(calibration_bins):
        if not bin_obs:
            calibration.append({
                "bucket": i,
                "mean_predicted": None,
                "empirical": None,
                "n": 0,
            })
            continue
        mean_pred = sum(p for p, _ in bin_obs) / len(bin_obs)
        empirical = sum(w for _, w in bin_obs) / len(bin_obs)
        calibration.append({
            "bucket": i,
            "mean_predicted": round(mean_pred, 4),
            "empirical": round(empirical, 4),
            "n": len(bin_obs),
        })

    return {
        "schedule_id": schedule_id,
        "experiment_id": schedule.experiment_id,
        "window_days": days,
        "races_evaluated": races_evaluated,
        "top1_accuracy": round(top1_correct / races_evaluated, 4) if races_evaluated else None,
        "top3_hit_rate": round(top3_correct / races_evaluated, 4) if races_evaluated else None,
        "mean_log_loss": round(sum(log_losses) / len(log_losses), 4) if log_losses else None,
        "calibration": calibration,
    }

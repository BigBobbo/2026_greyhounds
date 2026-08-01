import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import admin, tracks, dogs, races, features, training, predictions, bankroll, schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting up")
    stop_fn = None
    try:
        from app.tasks.scheduler import start_scheduler, stop_scheduler
        start_scheduler()
        stop_fn = stop_scheduler
    except Exception as e:
        logger.error("Scheduler failed to start: %s", e)
    yield
    logger.info("App shutting down")
    if stop_fn:
        try:
            stop_fn()
        except Exception as e:
            logger.error("Scheduler shutdown error: %s", e)


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tracks.router, prefix="/api")
app.include_router(dogs.router, prefix="/api")
app.include_router(races.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(bankroll.router, prefix="/api")
app.include_router(schedule.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

# Import scraping router separately — it uses Playwright which is heavy
try:
    from app.api import scraping
    app.include_router(scraping.router, prefix="/api")
    logger.info("Scraping router loaded")
except Exception as e:
    logger.error("Failed to load scraping router: %s", e)


@app.get("/api/health")
def health_check():
    """Health with substance: DB reachability, scrape freshness, scheduler.

    The old endpoint returned OK unconditionally, so a dead database or a
    scraper that hadn't succeeded in a month still reported healthy.
    Degradations are reported in-body with status "degraded" (HTTP 200 so
    platform healthchecks don't restart-loop the app over a stale scrape;
    a hard DB failure returns 503).
    """
    from datetime import datetime, timedelta

    from fastapi.responses import JSONResponse
    from sqlalchemy import text

    from app.database import SessionLocal

    problems: list[str] = []
    scrape_age_hours: float | None = None

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        try:
            from app.models.scrape_log import ScrapeLog

            last_ok = (
                db.query(ScrapeLog)
                .filter(ScrapeLog.status.in_(["success", "partial"]))
                .order_by(ScrapeLog.completed_at.desc())
                .first()
            )
            if last_ok and last_ok.completed_at:
                age = datetime.utcnow() - last_ok.completed_at
                scrape_age_hours = round(age.total_seconds() / 3600, 1)
                if age > timedelta(hours=36):
                    problems.append(
                        f"last successful scrape {scrape_age_hours}h ago"
                    )
            else:
                problems.append("no successful scrape recorded")
        except Exception as e:
            problems.append(f"scrape-log check failed: {e}")
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "app": settings.app_name,
                     "problems": [f"database unreachable: {e}"]},
        )
    finally:
        db.close()

    try:
        from app.tasks.scheduler import scheduler

        if not scheduler.running:
            problems.append("scheduler not running")
    except Exception as e:
        problems.append(f"scheduler check failed: {e}")

    return {
        "status": "degraded" if problems else "ok",
        "app": settings.app_name,
        "last_scrape_age_hours": scrape_age_hours,
        "problems": problems,
    }

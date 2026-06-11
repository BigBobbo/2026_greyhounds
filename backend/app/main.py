import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_api_key
from app.config import settings
from app.api import tracks, dogs, races, features, training, predictions, bankroll, schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting up")

    # Startup reaper + freshness check run BEFORE the scheduler starts so
    # jobs orphaned by the previous process (crash/redeploy) are marked
    # failed before anything new is kicked off. The freshness check only
    # LOGS a warning when results are behind — it deliberately does not
    # auto-trigger a scrape, because a crash-looping container would hammer
    # GRI on every boot. Operators are pointed at
    # POST /api/scraping/scrape-since-last-race-date instead.
    try:
        from app.database import SessionLocal
        from app.tasks.scheduler import reap_stale_jobs, warn_if_results_stale

        db = SessionLocal()
        try:
            reap_stale_jobs(db)
            warn_if_results_stale(db)
        finally:
            db.close()
    except Exception as e:
        logger.error("Startup reaper/freshness check failed: %s", e)

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


@lru_cache(maxsize=1)
def _app_version() -> str:
    """Single source of truth: the version in pyproject.toml."""
    try:
        import tomllib

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def _migration_head() -> str | None:
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        ini = Path(__file__).resolve().parent.parent / "alembic.ini"
        cfg = Config(str(ini))
        cfg.set_main_option(
            "script_location", str(Path(__file__).resolve().parent.parent / "alembic")
        )
        return ScriptDirectory.from_config(cfg).get_current_head()
    except Exception as e:
        logger.warning("Could not determine migration head: %s", e)
        return None


app = FastAPI(title=settings.app_name, version=_app_version(), lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_AUTH = [Depends(require_api_key)]

app.include_router(tracks.router, prefix="/api", dependencies=API_AUTH)
app.include_router(dogs.router, prefix="/api", dependencies=API_AUTH)
app.include_router(races.router, prefix="/api", dependencies=API_AUTH)
app.include_router(features.router, prefix="/api", dependencies=API_AUTH)
app.include_router(training.router, prefix="/api", dependencies=API_AUTH)
app.include_router(predictions.router, prefix="/api", dependencies=API_AUTH)
app.include_router(bankroll.router, prefix="/api", dependencies=API_AUTH)
app.include_router(schedule.router, prefix="/api", dependencies=API_AUTH)

# The scraping router pulls in the scraper stack; keep its import failure
# from taking down the whole API.
try:
    from app.api import scraping
    app.include_router(scraping.router, prefix="/api", dependencies=API_AUTH)
    logger.info("Scraping router loaded")
except Exception as e:
    logger.error("Failed to load scraping router: %s", e)


@app.get("/api/health")
def health_check():
    """Deep health check: verifies the DB answers and migrations match head.

    Returns 503 when the database is unreachable/locked or when the applied
    migration revision differs from the code's head. A database that has no
    alembic_version table at all (fresh local dev) is reported but not failed
    — production cannot reach that state because start.py exits on migration
    failure.
    """
    from sqlalchemy import text

    from app.database import engine

    body = {"status": "ok", "app": settings.app_name, "version": _app_version()}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            try:
                row = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
                current = row[0] if row else None
            except Exception:
                current = None
    except Exception as e:
        return JSONResponse(
            status_code=503, content={**body, "status": "error", "detail": f"database: {e}"}
        )

    head = _migration_head()
    body["migration"] = {"current": current, "head": head}
    if head is not None and current is not None and current != head:
        body["status"] = "error"
        body["detail"] = "applied migration does not match code head"
        return JSONResponse(status_code=503, content=body)
    return body

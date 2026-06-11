import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_api_key
from app.config import settings
from app.api import tracks, dogs, races, features, training, predictions, bankroll, schedule

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
    return {"status": "ok", "app": settings.app_name}

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import tracks, dogs, races, features, training, predictions, scraping
from app.tasks.scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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
app.include_router(scraping.router, prefix="/api")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}

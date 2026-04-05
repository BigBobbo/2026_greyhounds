FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml .
RUN pip install --no-cache-dir .

# Install Playwright Chromium + system dependencies
RUN playwright install chromium && playwright install-deps chromium

COPY backend/ .

# Create persistent data directory (mount Railway volume here at /app/data)
RUN mkdir -p data/models

# Run migrations and seed on startup, then start the server
CMD mkdir -p data/models && alembic upgrade head && python scripts/seed_tracks.py && python scripts/seed_features.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

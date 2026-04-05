FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

# Create artifacts directory
RUN mkdir -p artifacts/models

# Run migrations and seed tracks on startup, then start the server
CMD alembic upgrade head && python scripts/seed_tracks.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

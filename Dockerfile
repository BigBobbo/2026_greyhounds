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

# Run startup script (migrations, seeds, then uvicorn)
CMD ["python", "scripts/start.py"]

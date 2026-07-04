FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pinned dependencies only — the app itself runs from /app source,
# so installing the project package (an empty stub before sources are
# copied) is unnecessary and was masking import problems.
COPY backend/requirements.lock .
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY backend/ .

RUN mkdir -p data/models

# Deliberately running as root: Railway mounts the persistent volume at
# /app/data with root ownership, and a build-time chown cannot affect a
# runtime mount — a USER directive here would break DB/model writes on
# deploy. Revisit if Railway's volume UID handling (RAILWAY_RUN_UID) is
# adopted, or when moving the datastore off the volume (e.g. Postgres).
CMD ["python", "scripts/start.py"]

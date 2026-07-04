# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Greyhound Predictor — an Irish greyhound race prediction app with a Python/FastAPI ML backend and React/TypeScript frontend. It scrapes race data from GRI Ireland, computes configurable features, trains ML models (XGBoost, LightGBM, LambdaRank, sklearn), and serves predictions with Kelly criterion bankroll management. The success criterion is Kelly ROI on held-out data while beating the de-vigged SP baseline (see `backend/program.md` and the `sp_gate_*` experiment metrics).

## Common Commands

### Backend (from `backend/`)

```bash
pip install -r requirements.lock     # Pinned dependencies (lockfile is authoritative)
uvicorn app.main:app --reload        # Dev server
pytest                               # Run tests (uses a fresh migrated temp DB via tests/conftest.py)
ruff check .                         # Lint (config in pyproject.toml)
alembic upgrade head                 # Run DB migrations
python scripts/seed_tracks.py        # Seed 22 Irish tracks
python scripts/seed_features.py      # Seed default feature definitions
python scripts/backup.py             # Manual off-site backup (needs BACKUP_S3_*)
```

### Frontend (from `frontend/`)

```bash
npm install          # Install dependencies
npm run dev          # Dev server (Vite HMR, port 5173)
npm run build        # TypeScript check + Vite production build
npm run lint         # ESLint (must pass — enforced in CI)
npm test             # Vitest
```

### Docker / Production

The root Dockerfile is a multi-stage build (deps from `backend/requirements.lock`, app runs from `/app` source — the project package itself is not installed). Entrypoint is `python scripts/start.py`: create data dirs → alembic migrate (**fatal on failure**) → seed tracks → seed features → uvicorn. Deployed on Railway with a persistent volume at `/app/data/` (DB + model artifacts). Frontend deploys to Vercel. Railway auto-deploys from branch `claude/greyhound-prediction-app-4vNeO`.

## Architecture

### Data Flow

```
GRI Ireland scraper (httpx + BeautifulSoup)
  → db_pipeline.py → SQLite (WAL mode, busy_timeout, single writer)
  → feature_store.py materializes computed_features
  → dataset_builder.py assembles training matrices (ascending chronological,
    race-contiguous; global aggregates are time-aware prefix sums)
  → trainers/ (XGBoost, LightGBM, LambdaRank, sklearn) with Optuna tuning
  → prediction_service.py serves calibrated, per-race-normalized probabilities
  → Kelly staking parameterized by BankrollConfig
```

### Backend Structure (`backend/`)

- **`app/api/`** — FastAPI routers: races, dogs, features, tracks, training, predictions, bankroll, schedule, scraping. Every route except `/api/health` requires the `X-API-Key` header when `API_KEY` is set.
- **`app/models/`** — SQLAlchemy ORM (15 tables: tracks, dogs, races, race_entries, feature_definitions, computed_features, feature_versions, experiments, predictions, odds_snapshots, scrape_logs, bankroll_config, bet_records, model_schedule, scheduled_prediction_run)
- **`app/services/`** — training_service (threaded jobs, heartbeats, Optuna with a val objective-half/calibration-half split), prediction_service (calibrated probabilities, Kelly staking from BankrollConfig), feature_engine, feature_sandbox, backup_service, schedule_service
- **`ml/`** — feature_store, dataset_builder, evaluation (incl. beat-the-SP gate + serve-mirrored compounding Kelly backtest), race_features (time-aware batch aggregates), feature_availability (post-race feature classification), elo, trainers/
- **`scraping/`** — gri_scraper.py (header-keyed parsing; raises ScrapeFetchError/ParseStructureError instead of silently returning empty), db_pipeline.py (dogs resolve by GRI id; scratched-entry handling; days_since_last healing), backfill.py
- **`alembic/`** — the ONLY schema authority. Never call Base.metadata.create_all in app code; write a migration. tests/conftest.py migrates a fresh temp DB per test run, which doubles as a fresh-bootstrap regression test.

### Frontend Structure (`frontend/src/`)

- **`pages/`** — Dashboard, RaceList, RaceDetail, DogList, DogProfile, FeatureBuilder, TrainingLab, ExperimentDetail, Predictions, BankrollDashboard, ScrapingStatus, Schedule
- **`api/client.ts`** — Axios instance (`VITE_API_URL` + `VITE_API_KEY`); response interceptor toasts API errors (sonner)
- **`lib/kelly.ts`** — client-side Kelly mirror, parameterized by `/bankroll/config` (never hardcode staking constants)
- **`types/models.ts`** — shared TypeScript interfaces

### Key Design Decisions

- **No leakage, ever**: every feature must use only data strictly before the entry's race date. Batch aggregates are date-prefix sums; tests in `tests/test_no_future_leakage.py` enforce invariance to future races and to the entry's own result. Post-race-only features are listed in `ml/feature_availability.py` and excluded/refused automatically — `test_builtin_features_computable_pre_race` catches new offenders.
- **Datasets are ascending and race-contiguous** — walk-forward CV and LambdaRank groups depend on it and fail loudly otherwise.
- **Probabilities are per-race normalized everywhere** (serving, backtest, metrics) — see `ml/evaluation.normalize_probs_per_race`.
- **Selection vs calibration separation**: Optuna scores the earlier half of val; the final calibrator fits on the later half; autoresearch selects on val and touches test exactly once.
- **BankrollConfig is the single source of Kelly parameters** for backend and frontend.
- **Dog identity is GRI id first**, normalized name second (names get recycled).
- **Loud scraper failures**: markup drift and network errors raise; scrape jobs end "partial" with failed (track, date) pairs — never silent empty success.
- **Backups**: nightly VACUUM INTO snapshot + model artifacts to S3-compatible storage (`app/services/backup_service.py`); restore runbook in README.

## Environment Variables

See `backend/.env.example`: DATABASE_URL, CORS_ORIGINS, API_KEY, MODEL_ARTIFACTS_DIR (keep under ./data/), SCRAPE_DELAY, ENABLE_SCHEDULER, BACKUP_S3_*.
Frontend uses `VITE_API_URL` and `VITE_API_KEY`.

## Conventions for changes

- New revision for every schema change (`alembic revision`); keep the chain linear; CI runs the fresh-DB migration check.
- Run `pytest` and `ruff check .` (backend), `npm run lint && npm test && npm run build` (frontend) before pushing — CI enforces all of them.
- State-changing endpoints are POST (route-verb audit test in `tests/test_auth.py`).
- The `docs/` folder holds the audit (`improvement-audit-2026-06.html`) that tracks the improvement backlog this repo is being worked through against.

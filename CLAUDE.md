# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Greyhound Predictor — an Irish greyhound race prediction app with a Python/FastAPI ML backend and React/TypeScript frontend. It scrapes race data from GRI Ireland, computes configurable features, trains ML models (XGBoost, LightGBM, LambdaRank, sklearn), and serves predictions with Kelly criterion bankroll management.

## Common Commands

### Backend (from `backend/`)

```bash
pip install -e ".[dev]"              # Install dependencies
uvicorn app.main:app --reload        # Dev server
pytest                               # Run tests
alembic upgrade head                 # Run DB migrations
python scripts/seed_tracks.py        # Seed 24 Irish tracks
python scripts/seed_features.py      # Seed default feature definitions
```

### Frontend (from `frontend/`)

```bash
npm install          # Install dependencies
npm run dev          # Dev server (Vite HMR, port 5173)
npm run build        # TypeScript check + Vite production build
npm run lint         # ESLint
npm run preview      # Preview production build
```

### Docker / Production

The Dockerfile builds the backend (Python 3.12-slim). Startup sequence (see `backend/Procfile`): create data dirs → alembic migrate → seed tracks → seed features → uvicorn. Deployed on Railway with persistent volume at `/app/data/`. Frontend deploys to Vercel. Railway auto-deploys from branch `claude/greyhound-prediction-app-4vNeO`.

## Architecture

### Data Flow

```
GRI Ireland scraper (httpx + BeautifulSoup)
  → db_pipeline.py → SQLite (WAL mode)
  → feature_store.py materializes computed_features
  → dataset_builder.py assembles training matrices
  → trainers/ (XGBoost, LightGBM, LambdaRank, sklearn) with Optuna tuning
  → prediction_service.py serves calibrated probabilities
  → bankroll Kelly staking tracks P&L
```

### Backend Structure (`backend/`)

- **`app/api/`** — FastAPI routers: races, dogs, features, tracks, training, predictions, bankroll, scraping
- **`app/models/`** — SQLAlchemy ORM (13 tables: tracks, dogs, races, race_entries, feature_definitions, computed_features, feature_versions, experiments, predictions, odds_snapshots, scrape_logs, bankroll_config, bet_records)
- **`app/services/`** — Business logic: training_service (async threaded jobs with heartbeat), prediction_service (calibrated probabilities), feature_engine (visual/code feature execution), feature_sandbox (safe eval)
- **`ml/`** — ML pipeline: feature_store (materialization), dataset_builder, evaluation (metrics + SHAP), race_features (built-in isolation/context features), trainers/ (base class + 4 implementations)
- **`scraping/`** — gri_scraper.py (GRI Ireland), db_pipeline.py (DB writes), backfill.py (historical)
- **`alembic/`** — Database migrations

### Frontend Structure (`frontend/src/`)

- **`pages/`** — Dashboard, RaceList, RaceDetail, DogList, DogProfile, FeatureBuilder (visual + Monaco code editor), TrainingLab, ExperimentDetail, Predictions, BankrollDashboard, ScrapingStatus
- **`api/client.ts`** — Axios instance using `VITE_API_URL`
- **`types/models.ts`** — Shared TypeScript interfaces
- State: Zustand. Data fetching: TanStack Query. Styling: Tailwind CSS. Charts: Recharts.

### Key Design Decisions

- **Feature versioning**: Feature definitions are snapshotted at version creation for experiment reproducibility
- **Data completeness flags**: `computed_features.data_complete` tracks scraping gaps so models can handle missing data
- **Probability calibration**: Platt scaling on raw model outputs before serving predictions
- **Background training**: Threaded with heartbeat monitoring to detect stalled/crashed jobs
- **Feature types**: Visual (JSON config) and code-based (Python with sandboxed eval)

## Environment Variables

See `backend/.env.example`: DATABASE_URL, CORS_ORIGINS, BETFAIR_API credentials, MODEL_ARTIFACTS_DIR.
Frontend uses `VITE_API_URL` to point at the backend.

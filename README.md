# Greyhound Predictor

A betting research system for Irish greyhound racing. It scrapes results and
race cards from GRI Ireland, engineers ~100+ features (form, pace, ELO,
trainer/sire, race-relative), trains XGBoost / LightGBM / LambdaRank / sklearn
models with Optuna tuning and Platt calibration, serves win/forecast/trio
probabilities for upcoming cards, and sizes bets with fractional Kelly against
a tracked bankroll.

The objective (see `backend/program.md`) is **Kelly ROI > 5% on held-out data
while beating the SP-favourite baseline** — every completed experiment also
reports a "beat-the-SP" gate comparing its probabilities against de-vigged
starting prices.

## Architecture

```
GRI Ireland scraper (httpx + BeautifulSoup)
  → scraping/db_pipeline.py → SQLite (WAL)
  → ml/feature_store.py materializes computed_features
  → ml/dataset_builder.py assembles training matrices (time-aware aggregates)
  → ml/trainers/ (XGBoost, LightGBM, LambdaRank, sklearn) + Optuna
  → app/services/prediction_service.py serves calibrated probabilities
  → Kelly staking against bankroll_config
```

- **Backend**: Python 3.12 / FastAPI / SQLAlchemy / SQLite, deployed on
  Railway (single container; persistent volume at `/app/data/` holds the DB
  and trained model artifacts).
- **Frontend**: React + TypeScript + Vite + Tailwind, deployed on Vercel.
- **Scheduler**: in-process APScheduler (results scrapes at 23:00/08:00
  Europe/Dublin, nightly backup at 02:30, hourly stale-job reaper, plus
  user-defined daily prediction schedules).

## Local setup

Backend (from `backend/`):

```bash
pip install -r requirements.lock     # pinned deps (or: pip install -e ".[dev]")
alembic upgrade head                 # migrations are the only schema authority
python scripts/seed_tracks.py        # 22 Irish tracks
python scripts/seed_features.py      # default feature definitions
uvicorn app.main:app --reload        # dev server on :8000
pytest                               # test suite
ruff check .                         # lint
```

Frontend (from `frontend/`):

```bash
npm install
npm run dev          # Vite dev server on :5173 (proxies /api to :8000)
npm run lint && npm test && npm run build
```

## Configuration

Backend env vars (see `backend/.env.example`): `DATABASE_URL`,
`CORS_ORIGINS`, `API_KEY` (required in production — every `/api` route except
`/api/health` checks the `X-API-Key` header), `MODEL_ARTIFACTS_DIR` (keep it
under `./data/` so models live on the persistent volume), `SCRAPE_DELAY`,
`ENABLE_SCHEDULER`, and `BACKUP_S3_*` for off-site backups.

Frontend env vars (Vercel): `VITE_API_URL` (the Railway backend URL —
requests fail loudly in production without it) and `VITE_API_KEY`.

## Deployment

- Railway builds the root `Dockerfile` (multi-stage, installs from
  `backend/requirements.lock`) and runs `python scripts/start.py`:
  create data dirs → `alembic upgrade head` (**fatal on failure**) → seed
  tracks → seed features → uvicorn. `/api/health` checks the DB and the
  applied migration revision; Railway's healthcheck uses it.
- Vercel builds `frontend/` (SPA rewrites are configured in
  `frontend/vercel.json`).
- CI (`.github/workflows/ci.yml`) runs ruff + pytest + a fresh-database
  migration check + frontend lint/test/build on every PR.

## Backups & restore

A nightly job snapshots the SQLite DB (`VACUUM INTO`, integrity-checked) and
tars the model artifacts, uploading both to S3-compatible storage with
7-daily/4-weekly retention. Configure `BACKUP_S3_BUCKET`,
`BACKUP_S3_ENDPOINT_URL`, `BACKUP_S3_ACCESS_KEY`, `BACKUP_S3_SECRET_KEY`.

Manual run: `python scripts/backup.py`.

**Restore runbook** (volume loss / corruption):

1. Stop the API (or scale the Railway service to zero).
2. `python scripts/restore_backup.py --list` then
   `python scripts/restore_backup.py --latest --out ./data/greyhound.db`
   (downloads, integrity-checks).
3. Restore model artifacts from the matching `models-YYYYMMDD.tar.gz` into
   `./data/models/` if needed.
4. Start the API — `alembic upgrade head` no-ops if the snapshot is at head,
   otherwise applies the missing migrations.

## Documentation map

- `CLAUDE.md` — working guidance for LLM coding sessions.
- `docs/improvement-audit-2026-06.html` — full codebase audit with the task
  backlog this repo has been worked through against.
- `docs/PREDICTION_FEATURE_AUDIT.md` — pre-race vs post-race feature
  availability analysis and the train/serve-skew fix history.
- `docs/FEATURE_SUGGESTIONS.md` — research-backed feature engineering backlog.

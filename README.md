---
title: AutoML Studio
emoji: ✨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Inferyx

Inferyx is an end-to-end AutoML workspace built around a React single-page frontend, a FastAPI backend, and a Celery worker for background training. It supports dataset ingestion, profiling, training orchestration, results review, experiment tracking, drift monitoring, scenario analysis, synthetic data generation, and export workflows from one studio.

The live product is the React studio served by FastAPI. There are still some legacy Python frontend helpers in `frontend/`, but the primary UI path today is:

- React app source in `frontend/react-src/`
- built assets in `frontend/static/`
- FastAPI API and asset serving from `backend/`

The repo now treats generated assets, editor-only files, and scratch fixtures as local development artifacts unless they are part of a documented runtime path.

## What the Studio Does

- Ingests datasets from common tabular and document-style formats
- Profiles data quality, target hints, leakage, and lineage
- Runs AutoML training with progress telemetry and reasoning updates
- Reviews winning runs with leaderboard, validation drift, thresholds, and explainability
- Supports scenario simulation, future sweep, single prediction, counterfactuals, and ensemble building
- Tracks experiments, notes, workspace history, and drift schedules
- Exports model bundles, reports, and model cards

## Current Stack

- Frontend: React 19 + esbuild bundle script
- Backend: FastAPI
- Worker queue: Celery + Redis
- ML layer: scikit-learn, LightGBM, XGBoost, Optuna, custom training services
- Persistence: SQLAlchemy-backed local database plus run artifacts on disk

## Repository Layout

```text
CODIN/
├── backend/
│   ├── api/routes/                  # FastAPI route modules
│   ├── core/                        # loaders, export, synthetic, pipeline utilities
│   ├── services/                    # orchestration and training services
│   ├── tests/                       # backend tests
│   └── main.py                      # FastAPI entry point
├── frontend/
│   ├── react-src/                   # React source for the studio
│   ├── static/                      # built frontend bundle served by FastAPI
│   ├── app.py                       # legacy frontend entry
│   └── *.py                         # legacy frontend support helpers
├── scripts/
│   ├── build-frontend.mjs           # frontend bundle build
│   └── run-ruff.mjs                # repo-local Python lint launcher
├── run.sh                           # supported local Linux/macOS launcher
├── run_windows.bat                  # supported local Windows launcher
├── start_windows.bat                # compatibility wrapper for run_windows.bat
├── start.sh                         # supported container / deployed launcher
├── docker-compose.yml               # supported local container stack
├── FEATURE.md                       # product + architecture guide
├── PROJECT_LOGIC.md                 # feature-by-feature logic and system flow map
└── README.md
```

## Studio Pages

The React studio is organized into these main surfaces:

1. `Overview` - workspace pulse, uploads, connectors, import modes
2. `Data` - profiling, health, repair, lineage, leakage, merge preview
3. `Training` - progress rail, neural telemetry, reasoning, completed-run snapshot
4. `Results` - outcome brief, metrics, drift, leaderboard, thresholds, explainability, artifacts
5. `Tracking` - experiments, notes, history, diffs, workspaces
6. `Monitoring` - drift detection, schedules, retraining, prediction, scenario context
7. `Tools` - ensemble builder, synthetic data, natural-language helpers, batch tooling

## Supported Intake

The app currently advertises broad upload support through the frontend and shared loader path, including:

- `.csv`, `.tsv`
- `.xlsx`, `.xls`
- `.json`
- `.parquet`, `.feather`
- `.pdf`
- `.txt`
- `.html`, `.htm`
- `.xml`
- `.zip`
- `.sav` (SPSS)
- `.sas7bdat`
- `.dta`

Connector-based import is also available from the studio for SQL-style sources.

## Quality Gate

Repo-local checks live in `package.json` so they can run the same way on Linux, macOS, and Windows:

```bash
npm run lint:python
npm run lint:frontend
npm run format:check:frontend
npm run quality
```

Helpful local commands:

- `npm run format:frontend` reformats the React app, bundle scripts, and top-level JSON/Markdown files
- `npm run quality` runs Python linting, frontend linting, and a frontend build
- `CODIN_RUN_QUALITY=1 bash run.sh` runs the gate before launching the local Linux/macOS stack

Python linting is powered by `ruff` through `requirements.txt`. Frontend linting/formatting is local to this repo through ESLint and Prettier.
The Python lint gate is intentionally scoped to the active backend product path.

## Supported Runtime Paths

### Prerequisites

- Python 3.10+ recommended
- Node.js + npm
- Redis
- Bash on Linux/macOS, or Command Prompt / PowerShell on Windows

### Linux / macOS

```bash
bash run.sh
```

This script will:

- create and activate `venv/` if needed
- install Python requirements on first run
- install frontend dependencies on first run
- build the React frontend bundle
- start Redis if it is not already running
- start the Celery worker
- start FastAPI on port `8000`

After startup:

- Studio: `http://localhost:8000/overview`
- API root: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

Stop everything with `Ctrl+C`.

### Windows

Use the supported local launcher:

```bat
run_windows.bat
```

This script builds the frontend bundle, wires Python dependencies, and starts the backend services for a local workstation flow.
If you still invoke `start_windows.bat`, it now delegates to `run_windows.bat` so the Windows local path stays single-sourced.

### Docker local

Use the container stack when you want the deployed-style runtime locally:

```bash
docker compose up --build
```

This path builds the image from `Dockerfile`, starts Redis + Celery alongside FastAPI, and exposes the studio directly on port `7860`.

### Deployed / container runtime

`start.sh` is the supported launcher for container-style environments, including Spaces-style deployment. It starts:

- Redis
- FastAPI on the public `HOST` / `PORT`
- Celery

Important environment variables:

- `HOST`
- `PORT`
- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `AUTOML_ALLOWED_ORIGINS`
- `DATABASE_URL`
- `MAX_UPLOAD_MB`

The container path assumes the frontend bundle was created during image build. If you need a startup-time rebuild for debugging, set `CODIN_BUILD_FRONTEND_ON_START=1`.

For Hugging Face Spaces specifically:

- keep `sdk: docker` and `app_port: 7860` in the README front matter
- the image now uses a multi-stage build so Node is only used while building the React bundle
- Uvicorn serves the React app and API directly on the Spaces port, which avoids non-root Nginx edge cases
- `.dockerignore` excludes local databases, `node_modules`, `venv`, generated assets, and large data artifacts so Spaces builds stay smaller and faster

### Manual fallback

If you want to launch pieces yourself:

1. Create a virtual environment and install Python dependencies
2. Install frontend dependencies and build the bundle:

```bash
npm install
npm run build:frontend
```

3. Start Redis
4. Start Celery from `backend/`:

```bash
python -m celery -A core.worker.celery_app worker --loglevel=warning --concurrency=2
```

5. Start FastAPI from `backend/`:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600
```

## Training Flow

The active training path is service-driven and roughly follows this sequence:

1. Load dataset through shared loader utilities
2. Validate dataset and target configuration
3. Run sanitizer-driven cleaning when enabled
4. Infer task type and prep features
5. Build candidate model pool
6. Run sweep and optional Optuna tuning
7. Train the final artifact
8. Persist metrics, drift baseline, explainability, exports, and run metadata

The saved inference artifact is designed to keep train-time and serve-time preprocessing aligned.

## Notes About Legacy Frontend Files

You will still see files like:

- `frontend/app.py`
- `frontend/error_handler.py`
- `frontend/state_manager.py`
- `frontend/validators.py`

Those are legacy Python frontend/support pieces. They may still be useful for older flows or utility logic, but they are not the primary UI delivery path for the current studio.

## Dev Artifacts vs Product Artifacts

Keep these boundaries sharp:

- `frontend/react-src/` and `backend/` are product code
- `frontend/static/` is generated output and should be rebuilt, not edited by hand
- editor-only config such as `.vscode/` stays local and is ignored
- generated scratch fixtures should live under ignored paths such as `tests/data/generated/`
- one-off migration or experiment scaffolding should not become part of the documented runtime path unless it is intentionally supported

## Verification

Common checks during local work:

```bash
npm run build:frontend
./venv/bin/python -m ruff check backend
./venv/bin/pytest backend/tests/test_services.py -q
./venv/bin/pytest backend/tests/test_api_endpoints.py -q
```

## Contributing

When updating the product, keep the React frontend, top-level docs, launcher scripts, and quality scripts in sync. The most common drift in this repo comes from features landing in the studio while README and startup guidance still describe older behavior.

For the deeper feature-by-feature behavior map, sequence diagrams, and endpoint contract reference, see [PROJECT_LOGIC.md](/home/aj/Documents/CODIN/PROJECT_LOGIC.md:1).

## License

This project is licensed under the MIT License.

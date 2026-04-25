# CODIN Feature Guide

This file is the practical product map for the current repository state.

For the full feature logic, route-to-service flow map, sequence diagrams, and API contract reference, see [PROJECT_LOGIC.md](/home/aj/Documents/CODIN/PROJECT_LOGIC.md:1).

## Product Shape

CODIN is a React + FastAPI AutoML studio with a Celery worker and Redis-backed job queue.

The primary user experience is the React studio in `frontend/react-src/`, served through FastAPI after bundling to `frontend/static/`.

The app is organized around one continuous workflow:

1. ingest data
2. inspect and repair it
3. train models
4. review results
5. track experiments
6. monitor drift
7. use advanced tooling like scenarios, ensembles, and synthetic generation

## Main Studio Surfaces

### 1. Overview

Purpose:

- workspace status
- upload and import entry points
- connector setup
- high-level operational summary

Highlights:

- multi-format file intake
- connector import mode
- workspace heartbeat
- current dataset and run context

### 2. Data

Purpose:

- inspect dataset structure and quality before training

Highlights:

- profile summary
- health checks
- target hints
- leakage analysis
- repair preview/apply
- lineage graph and timeline
- merge preview

### 3. Training

Purpose:

- launch and monitor AutoML execution

Highlights:

- realtime progress rail
- neural pulse telemetry
- live reasoning feed
- checkpoint history
- completed-run winner snapshot

### 4. Results

Purpose:

- review the winning run quickly and deeply

Highlights:

- outcome brief
- performance metrics
- validation drift
- scenario snapshot
- calibration trace
- leaderboard
- threshold review
- explainability sections
- recommendations
- artifact and lineage export actions

### 5. Tracking

Purpose:

- compare runs and manage historical context

Highlights:

- experiments
- notes
- diffs
- workspaces
- dataset history

### 6. Monitoring

Purpose:

- operate trained models after training

Highlights:

- drift checks
- drift schedules
- retraining flow
- single prediction
- future sweep
- scenario context
- goal seeking

### 7. Tools

Purpose:

- advanced, reusable model and dataset utilities

Highlights:

- ensemble builder
- synthetic data generation
- natural-language intent parsing
- chat with run context
- quicktrain helpers
- batch tooling

## Backend Capabilities In Use

### Data and intake

- shared file loading across multiple formats
- dataset profiling
- leakage analysis
- repair and sanitation flows
- lineage tracking

### Training

- validation and cleaning
- task detection
- feature preparation
- candidate model selection
- sweep and optional tuning
- final artifact training
- holdout evaluation
- report/export generation

### Analysis

- thresholds
- calibration
- permutation importance
- SHAP summary when available
- trust surface when available
- scenario and prediction APIs
- drift timeline and schedule support

### Advanced tooling

- prefit ensemble strategies including voting, bagging, boosting, and stacking
- synthetic data generation with type-aware sampling
- synthetic quality judging
- counterfactual / goal-seeker support

## Current Truths Worth Keeping Straight

- The React studio is the main frontend. The Python frontend files in `frontend/` are legacy helpers, not the primary product shell.
- The Results page now prefers honest empty states over placeholder visuals when backend assets are missing.
- The Training page includes a completed-run snapshot so users can see the winner before opening Results.
- Ensemble strategies operate on existing trained runs and saved artifacts, not by retraining the base estimators from scratch.

## Scripts and Entry Points

- `run.sh` - supported local Linux/macOS launcher
- `run_windows.bat` - supported local Windows launcher
- `start_windows.bat` - compatibility wrapper that delegates to `run_windows.bat`
- `start.sh` - supported container / deployed launcher
- `docker-compose.yml` - local deployed-style runtime path
- `scripts/build-frontend.mjs` - React bundle builder
- `scripts/run-ruff.mjs` - repo-local Python lint launcher
- `npm run quality` - repo-local lint/build gate

## Common Drift Risks

If you change the product again, keep these in sync:

- `README.md`
- `FEATURE.md`
- `run.sh`
- `run_windows.bat`
- `start.sh`
- `docker-compose.yml`
- `package.json`
- `pyproject.toml`
- `PROJECT_LOGIC.md`
- any page names or route labels in the React studio

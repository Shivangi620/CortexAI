# CODIN Feature Guide

## What This Project Is

CODIN is an end-to-end AutoML studio built with:

- `FastAPI` for APIs and backend orchestration
- `Streamlit` for the multi-page workspace UI
- `SQLAlchemy + SQLite` for datasets, jobs, experiment history, notes, registry labels, and drift records
- `scikit-learn`, `LightGBM`, `Optuna`, and custom pipeline logic for training and evaluation

The product is designed as a guided ML workspace rather than a single training script. A user can ingest data, profile it, train models, inspect results, simulate predictions, detect drift, generate synthetic data, compare runs, and export reports from the same app.

## Core Product Flow

1. A dataset is uploaded or imported.
2. The backend profiles the dataset and stores metadata in `datasets`.
3. A training job is created and the pipeline runs validation, cleaning, feature engineering, model selection, training, and evaluation.
4. Results are saved to run artifacts and experiment history.
5. The user explores explainability, contracts, what-if tools, drift, augmentation, and reports.

## Main User-Facing Workspaces

### 1. Home / Upload And Configure

Primary purpose:

- ingest new data
- restore exported bundles
- import from external connectors
- merge datasets
- configure AutoML training

Key features:

- File upload for tabular, document, image, PDF, SQLite, and export bundle formats
- Connector-based import for SQL sources
- Merge Studio for previewing joins before materializing merged datasets
- Auto problem detection to infer likely target/task hints
- Training configuration for mode, metric, imbalance handling, cleaning, CV, and feature selection

Logic:

- Uploads are normalized through `core/file_loader.py`
- Dataset profile information is generated and saved in `profile_json`
- Dataset lineage is tracked with `parent_dataset_id`
- Training configuration is stored on the job and reused in later pages

### 2. Dataset DNA

Primary purpose:

- inspect dataset structure and quality before training

Key features:

- dataset profile overview
- numeric/categorical breakdown
- per-column stats table
- auto-detect details
- leakage and quality analysis
- repair preview and repair apply
- dataset lineage timeline and graph

Logic:

- Profiling computes row counts, column types, missingness, target hints, and column stats
- Leakage checks look for suspicious target correlation, ID-like fields, constants, duplicates, and temporal leakage hints
- Repair flows preview transformations first, then create a cleaned derived dataset

### 3. Training Lab

Primary purpose:

- run AutoML training with pipeline-backed execution

Key features:

- job launch
- progress/status monitoring
- reasoning stream from the pipeline
- deep analysis handoff to Results Console

Logic:

- Training uses component-based pipeline execution from `core/pipeline_engine.py`
- Each component updates status and appends reasoning so the UI can show not just what happened, but why

### 4. Results Console

Primary purpose:

- inspect the trained model and all downstream analysis

Key features:

- leaderboard and winner summary
- execution profile and pipeline metrics
- tested model breakdown
- SHAP global importance
- calibration report
- threshold tuner
- feature lineage
- recommendations
- trust heatmap
- drift timeline
- prediction sandbox
- feature contract checker
- counterfactual generation
- scenario sweep
- synthetic augmentation controls
- lightweight model chat

Logic:

- Results are normalized through `infra/result_contract.py`
- UI panels call dedicated APIs for each analysis instead of overloading one giant endpoint
- Mixed display tables are sanitized before rendering to keep Streamlit Arrow serialization stable

### 5. Experiment Tracker

Primary purpose:

- compare historical runs and manage model promotion workflow

Key features:

- experiment archive
- global leaderboard
- run history viewer
- reasoning stream viewer
- registry labels: champion, challenger, candidate, archived
- team notes
- side-by-side run comparison
- run diff engine
- battle arena charts

Logic:

- Each completed job can be persisted as an `ExperimentRun`
- Registry and note models allow lightweight model governance without needing an external MLOps platform

### 6. Drift Monitor

Primary purpose:

- detect dataset drift against the saved baseline and operationalize retraining

Key features:

- upload new batch for drift check
- per-feature drift metrics
- drift severity summary
- cadence scheduling
- drift history
- one-click retrain on drifted data

Logic:

- Baseline distributions are fit during training
- Later uploads are compared with PSI/KS-style checks
- Drift checks are stored so timeline views and cadence workflows can be built on top

### 7. Smart AI Hub

Primary purpose:

- provide higher-order utilities around the trained models and datasets

Key features:

- ensemble builder
- what-if simulation
- synthetic data generation
- natural language ML helpers

Logic:

- This page reuses completed jobs and current workspace datasets rather than creating isolated tooling
- It is meant to sit on top of the main AutoML lifecycle

## Training Pipeline Components

### Data Validation

Implemented in `backend/services/training/components.py`.

What it does:

- loads the dataset
- checks target existence
- trims to selected features if requested
- drops columns with extreme missingness
- attempts numeric coercion on object columns
- runs leakage detection
- saves the data contract
- fits drift baseline

Why:

- it prevents garbage-in training runs
- it creates the baseline metadata needed by later features such as drift monitoring and contract checks

### Feature Engineering

What it does:

- optional auto-cleaning
- task-type inference
- managed feature generation for smaller datasets
- invalid target cleanup
- label encoding for classification
- train/test split
- summary statistics for later reporting

Why:

- it balances automation with safety
- feature synthesis is only used when dataset size and width are still manageable

### Model Selection

What it does:

- builds a profile of the problem
- asks the selector/meta-learner for a candidate pool
- chooses light or full preprocessing based on mode
- enables dimensionality guidance for wide datasets

Why:

- it makes the model search adaptive instead of static

### Training

What it does:

- handles imbalance when configured
- executes sweep/tuning behavior
- trains the final pipeline
- evaluates on holdout data
- computes leaderboard, metrics, and explainability artifacts
- stores model metadata and run artifacts

Why:

- it separates fast exploration from deeper optimization
- it preserves enough metadata for later explainability and operational tools

## Explainability Features

### SHAP Global Importance

- ranks feature impact across the trained model
- used in the Results Console and reports

### Local Explanation

- explains a single prediction using per-feature contributions
- used to rank candidate features for counterfactual search

### Counterfactual Generator

Endpoint:

- `POST /api/counterfactual/{job_id}`

Logic:

- only enabled for classification jobs
- loads the saved model and training dataset context
- validates that all expected feature inputs are present
- scores the original row
- gets local feature contributions
- ranks the most influential features
- tries one-feature changes using numeric quantiles or common categorical alternatives
- returns the smallest single-field changes that flip the prediction

Current behavior:

- verified against job `3c29d593-31b3-4116-bf5e-a1b3d48d130b`
- returned a valid one-feature flip suggestion on `Age`

### Feature Lineage

- inspects the saved preprocessor and maps transformed feature names back to raw groups

### Calibration And Threshold Tuning

- computes classification calibration bins
- sweeps thresholds and compares precision/recall/F1

### Trust Heatmap

- checks how often features appear across recent runs for the same dataset
- compares historical importance stability
- marks features as stable, noisy, drift-prone, or leakage-risky

## Prediction And Simulation Features

### Live Prediction

Endpoint:

- `POST /api/predict/{job_id}`

Logic:

- validates exact feature contract
- builds a one-row inference frame in expected column order
- returns prediction and confidence when probabilities exist

### Scenario Sweep / What-If Simulation

Endpoint:

- `POST /api/future`

Logic:

- takes a base feature vector
- varies one selected feature over multiple values
- scores each generated row independently
- returns prediction and optional confidence for each point

Current behavior:

- verified against job `3c29d593-31b3-4116-bf5e-a1b3d48d130b`
- returned valid prediction points for a `Salary` sweep with no point-level errors

### Feature Contract Checker

- compares uploaded inference data against the training feature schema
- reports missing features, extra columns, dtype mismatches, and alignment status

## Data Management Features

### Dataset Catalog

- lists known datasets with profile summaries
- supports derived datasets and lineage-aware workflows

### Merge Studio

- previews join quality before creating merged datasets
- reports overlap, duplicate keys, row multiplier, and sample merged records

### Repair Preview / Apply

- shows proposed cleaning effect before materializing a new dataset version

### Synthetic Data Generator

- creates synthetic rows from the current dataset
- stores derived dataset with lineage back to parent

### Synthetic Data Judge

- compares synthetic output against the parent dataset
- checks numeric distribution drift and categorical mix overlap
- returns realism score, verdict, and notes

## Experiment And Governance Features

### Experiment Tracking

- stores run-level summary information in `experiment_runs`
- powers run archive and comparison UI

### Registry Labels

- lightweight promotion workflow for production candidates

### Team Notes

- lets users annotate runs without leaving the product

### Run Diff Engine

- compares config and output changes between two runs

## Reporting And Export

### Report Generator

- creates PDF summaries with dataset overview, metrics, and SHAP importance

### Export Bundle

- packages model, metadata, and training context for later restore

## Natural Language And AI Helpers

### Narrative Generator

- builds a human-readable summary of the experiment
- uses stored job story when available, otherwise composes a fallback narrative

### Natural Language ML Helpers

- parse or support ML-oriented natural language workflows from the Smart AI Hub

## Storage And State Model

### Database Models

- `DatasetModel`: uploaded/imported/derived datasets
- `JobModel`: training jobs and result payloads
- `ExperimentRun`: completed-run archive
- `DriftCheck`: drift history
- `DriftSchedule`: cadence configuration
- `ModelRegistryEntry` and `TeamNote`: governance helpers

### Run Artifacts

Per-run directories store:

- model pickle
- metrics JSON
- schema/data contract
- drift baseline
- model metadata
- exports and reports

## Important Implementation Notes

### Result Contract Normalization

- all result payloads are sanitized to be JSON-safe
- prevents NaN/Inf and shape drift from breaking the UI

### Session Safety

- the DB session now uses `expire_on_commit=False`
- this avoids detached-instance crashes when recently loaded attributes are used after commit

### Known Residual Noise

- LightGBM/scikit-learn still emits `X does not have valid feature names` warnings during some prediction paths
- these warnings did not block counterfactual or scenario sweep in runtime verification, but they are worth cleaning up later for quieter logs

## Verified Status From This Pass

- Counterfactual: working on the verified completed job
- Scenario sweep: working on the verified completed job
- Trust heatmap: no longer failing from the detached-session regression in the verified direct check
- Dataset list helper: no longer failing from the detached-session regression in the verified direct check

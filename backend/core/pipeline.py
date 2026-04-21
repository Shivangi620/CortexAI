"""
core/pipeline.py — Formal workflow tracking and execution pipeline for AutoML.

Replaces 'execute script' style monolithic functions.
Provides strict boundaries:
    Step 1: Validate
    Step 2: Profile & Clean
    Step 3: Feature Engineering
    Step 4: Train
    Step 5: Evaluate & Persist
"""

import json
from enum import Enum
import pandas as pd
from typing import Dict, Any

from infra.database import get_db, db_session, JobModel, DatasetModel
from services.training.trainer import train_automl


class PipelineStep(str, Enum):
    INITIALIZE = "Initializing"
    VALIDATE = "Validating Data"
    PROFILE = "Profiling & Cleaning"
    FEATURE_ENG = "Feature Engineering"
    TRAIN = "Model Training"
    EVALUATE = "Evaluation & Extraction"
    COMPLETE = "Completed"
    FAILED = "Failed"


class PipelineEngine:
    def __init__(self, job_id: str, dataset_id: str, file_path: str, target: str, goal: str, mode: str):
        self.job_id = job_id
        self.dataset_id = dataset_id
        self.file_path = file_path
        self.target = (target or "").strip()
        self.goal = goal
        self.mode = mode
        self.state: Dict[str, Any] = {}
        self.logs: list = []

    def log(self, message: str):
        print(f"[Pipeline][{self.job_id}] {message}")
        self.logs.append(message)

    def update_status(self, step: PipelineStep):
        self.log(f"Status update: {step.value}")

        try:
            with db_session() as db:
                job = db.query(JobModel).filter(JobModel.id == self.job_id).first()
                if job:
                    if step == PipelineStep.COMPLETE:
                        job.status = "completed"
                    elif step == PipelineStep.FAILED:
                        job.status = "failed"
                    else:
                        job.status = "running"
                    db.commit()
        except Exception:
            pass  # never break pipeline due to DB failure

    def run(
        self,
        eval_metric="Performance",
        handle_imbalance=False,
        auto_clean=True,
        cv_folds=0,
        selected_features=None
    ):
        try:
            # ── STEP 1: VALIDATE ─────────────────────────────
            self.update_status(PipelineStep.VALIDATE)

            df = pd.read_csv(self.file_path)
            if df is None or df.empty:
                raise ValueError("Dataset is empty or unreadable")

            if self.target not in df.columns:
                raise ValueError(f"Target column '{self.target}' not found in dataset")

            # Optional feature filtering
            if selected_features:
                keep_cols = list(selected_features) + [self.target]
                available = [c for c in keep_cols if c in df.columns]

                if self.target not in available:
                    raise ValueError("Target column removed after feature selection")

                df = df[available]

            # ── STEP 2: PROFILE ─────────────────────────────
            self.update_status(PipelineStep.PROFILE)

            with db_session() as db:
                dataset = db.query(DatasetModel).filter(DatasetModel.id == self.dataset_id).first()

                try:
                    profile = json.loads(dataset.profile_json) if dataset and dataset.profile_json else {}
                except Exception:
                    profile = {}

                health_metadata = profile.get("health", {})

            # ── STEP 3: TRAIN ───────────────────────────────
            self.update_status(PipelineStep.TRAIN)

            results = train_automl(
                df,
                self.target,
                self.goal,
                self.mode,
                self.job_id,
                eval_metric=eval_metric,
                handle_imbalance=handle_imbalance,
                auto_clean=auto_clean,
                health_metadata=health_metadata,
                cv_folds=cv_folds,
            )

            # ── STEP 4: EVALUATE ────────────────────────────
            self.update_status(PipelineStep.EVALUATE)

            results = results or {}
            self.logs.extend(results.get("reasoning", []))

            # ── STEP 5: COMPLETE ────────────────────────────
            self.update_status(PipelineStep.COMPLETE)

            return results, profile, self.logs

        except Exception as e:
            self.update_status(PipelineStep.FAILED)
            self.log(f"Error: {e}")
            raise e
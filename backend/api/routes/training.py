"""api/routes/training.py — Train, status, jobs, leaderboard, ensemble, WebSocket."""

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import asyncio
import pandas as pd

from infra.database import get_db, db_session, DatasetModel, JobModel
from infra.result_contract import normalize_results, sanitize_for_json
from core.file_loader import load_dataframe

router = APIRouter(prefix="/api", tags=["training"])


def _resolve_target_column(df, requested_target: str, suggested_target: str = "") -> str:
    requested = (requested_target or "").strip()
    suggested = (suggested_target or "").strip()
    columns = list(df.columns)

    def match(name: str) -> str:
        if not name:
            return ""
        if name in columns:
            return name
        normalized_name = name.casefold().replace(" ", "").replace("_", "")
        for column in columns:
            normalized_column = str(column).casefold().replace(" ", "").replace("_", "")
            if normalized_column == normalized_name:
                return column
        return ""

    return match(requested) or match(suggested) or ""


# ── Train ──────────────────────────────────────────────────────────────────────


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    goal: str
    mode: str
    preset_name: str = ""
    workspace_id: str = ""
    workspace_name: str = ""
    eval_metric: str = ""
    selected_features: List[str] = Field(default_factory=list)
    handle_imbalance: bool = False
    auto_clean: bool = True
    cv_folds: int = 0
    pca_mode: str = "auto"
    pca_components: int = 0
    export_model: bool = True
    export_code: bool = True
    export_report: bool = True


class TrainingForecastRequest(BaseModel):
    dataset_id: str
    target_column: str = ""
    goal: str = "Balanced"
    mode: str = "Balanced"
    preset_name: str = ""
    eval_metric: str = ""
    selected_features: List[str] = Field(default_factory=list)
    handle_imbalance: bool = False
    auto_clean: bool = True
    cv_folds: int = 0
    pca_mode: str = "auto"
    pca_components: int = 0


@router.post("/train/forecast")
def get_training_forecast(req: TrainingForecastRequest):
    from services.training.forecasting import estimate_training_forecast

    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        try:
            profile = json.loads(dataset.profile_json) if dataset.profile_json else {}
        except Exception:
            profile = {}

    target_column = (req.target_column or profile.get("suggested_target") or "").strip()
    return estimate_training_forecast(
        profile=profile,
        target_column=target_column,
        goal=req.goal,
        mode=req.mode,
        selected_features=req.selected_features,
        cv_folds=req.cv_folds,
        handle_imbalance=req.handle_imbalance,
        auto_clean=req.auto_clean,
        eval_metric=req.eval_metric,
        pca_mode=req.pca_mode,
        pca_components=req.pca_components,
    )


@router.post("/train")
def start_training(req: TrainRequest):
    req.target_column = (req.target_column or "").strip()

    from uuid import uuid4

    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        file_path = dataset.file_path
        try:
            profile = json.loads(dataset.profile_json) if dataset.profile_json else {}
        except Exception:
            profile = {}

    try:
        df_preview = load_dataframe(filepath=file_path)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read dataset for training: {e}",
        ) from e

    resolved_target = _resolve_target_column(
        df_preview,
        req.target_column,
        profile.get("suggested_target", ""),
    )
    if not resolved_target:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Target column '{req.target_column}' was not found in the selected dataset. "
                "Choose one of the available columns or use the suggested target."
            ),
        )
    req.target_column = resolved_target

    job_id = str(uuid4())

    eval_metric = req.eval_metric
    if not eval_metric:
        inferred_task = "classification"
        try:
            if (
                df_preview is not None
                and not df_preview.empty
                and req.target_column in df_preview.columns
            ):
                y = df_preview[req.target_column].dropna()
                if not y.empty and pd.api.types.is_numeric_dtype(y):
                    unique_count = y.nunique(dropna=True)
                    unique_ratio = unique_count / max(len(y), 1)
                    if pd.api.types.is_float_dtype(y) or not (
                        unique_count <= 20 and unique_ratio <= 0.2
                    ):
                        inferred_task = "regression"
            del df_preview
        except Exception:
            inferred_task = "classification"

        eval_metric = "RMSE" if inferred_task == "regression" else "Accuracy"

    full_params = {
        "target_column": req.target_column,
        "goal": req.goal,
        "mode": req.mode,
        "eval_metric": eval_metric,
        "selected_features": req.selected_features,
        "handle_imbalance": req.handle_imbalance,
        "auto_clean": req.auto_clean,
        "cv_folds": req.cv_folds or 5,
        "pca_mode": req.pca_mode,
        "pca_components": req.pca_components,
        "preset_name": req.preset_name,
        "workspace_id": req.workspace_id,
        "workspace_name": req.workspace_name,
        "export_model": req.export_model,
        "export_code": req.export_code,
        "export_report": req.export_report,
    }

    try:
        params_json = json.dumps(full_params)
    except Exception:
        params_json = json.dumps({})

    with db_session() as db:
        db.add(
            JobModel(
                id=job_id,
                dataset_id=req.dataset_id,
                status="training",
                reasoning_json="[]",
                params_json=params_json,
            )
        )
        db.commit()

    from core.worker import run_training_job

    run_training_job.delay(
        job_id,
        req.dataset_id,
        file_path,
        req.target_column,
        req.goal,
        req.mode,
        eval_metric=eval_metric,
        selected_features=req.selected_features,
        handle_imbalance=req.handle_imbalance,
        auto_clean=req.auto_clean,
        cv_folds=req.cv_folds or 5,
        pca_mode=req.pca_mode,
        pca_components=req.pca_components,
    )

    return {"job_id": job_id}


# ── Job Status ─────────────────────────────────────────────────────────────────


@router.get("/status/{job_id}")
def get_status(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        try:
            history = json.loads(job.history_json) if job.history_json else []
        except Exception:
            history = []

        try:
            results = (
                normalize_results(json.loads(job.results_json))
                if job.results_json
                else None
            )
        except Exception:
            results = None

        try:
            insights = json.loads(job.insights_json) if job.insights_json else {}
        except Exception:
            insights = {}

        try:
            reasoning = json.loads(job.reasoning_json) if job.reasoning_json else []
        except Exception:
            reasoning = []

        return {
            "status": job.status,
            "history": sanitize_for_json(history),
            "results": results,
            "insights": sanitize_for_json(insights),
            "reasoning": sanitize_for_json(reasoning),
            "story": job.story,
            "error": job.error,
        }


# ── Jobs List ──────────────────────────────────────────────────────────────────


@router.get("/jobs")
def list_jobs():
    with db_session() as db:
        jobs = db.query(JobModel).order_by(JobModel.created_at.desc()).limit(100).all()

        result = []
        for job in jobs:
            try:
                results_data = (
                    normalize_results(json.loads(job.results_json))
                    if job.results_json
                    else {}
                )
            except Exception:
                results_data = {}

            result.append(
                {
                    "id": job.id,
                    "dataset_id": job.dataset_id,
                    "status": job.status,
                    "created_at": (
                        job.created_at.isoformat() if job.created_at else None
                    ),
                    "best_model": results_data.get("best_model"),
                    "score": results_data.get("score"),
                    "metric_name": results_data.get("metric_name"),
                    "error": job.error,
                }
            )

        return result


# ── Global Leaderboard ─────────────────────────────────────────────────────────


@router.get("/leaderboard")
def global_leaderboard():
    with db_session() as db:
        jobs = db.query(JobModel).filter(JobModel.status == "completed").all()

        leaderboard = []
        for job in jobs:
            try:
                results = (
                    normalize_results(json.loads(job.results_json))
                    if job.results_json
                    else {}
                )
                for entry in results.get("leaderboard", []):
                    leaderboard.append(
                        {
                            "job_id": job.id,
                            "model": entry.get("model"),
                            "score": entry.get("score"),
                            "metric_name": results.get("metric_name"),
                            "created_at": (
                                job.created_at.isoformat() if job.created_at else None
                            ),
                            "is_classification": results.get("is_classification"),
                            "task": (
                                "Classification"
                                if results.get("is_classification")
                                else "Regression"
                            ),
                        }
                    )
            except Exception:
                continue

        leaderboard.sort(
            key=lambda x: x["score"] if isinstance(x["score"], (int, float)) else 0,
            reverse=True,
        )

        return leaderboard[:50]


# ── WebSocket live status ──────────────────────────────────────────────────────


@router.websocket("/ws/status/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            with db_session() as db:
                job = db.query(JobModel).filter(JobModel.id == job_id).first()
                if not job:
                    await websocket.send_json({"error": "Job not found"})
                    break

                try:
                    history = json.loads(job.history_json) if job.history_json else []
                except Exception:
                    history = []

                payload = {
                    "status": job.status,
                    "history": sanitize_for_json(history),
                }

            await websocket.send_json(payload)

            if payload["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass


# ── Ensemble Builder ───────────────────────────────────────────────────────────


class EnsembleRequest(BaseModel):
    job_ids: List[str]
    strategy: str = "voting"
    dataset_id: Optional[str] = None


@router.post("/ensemble")
def build_ensemble(req: EnsembleRequest):
    from services.ensemble_service import build_ensemble as _build

    return _build(
        req.job_ids,
        req.strategy,
        req.dataset_id,
    )

"""api/routes/training.py — Train, status, jobs, leaderboard, ensemble, WebSocket."""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import asyncio
import pandas as pd
import numpy as np

from infra.database import db_session, DatasetModel, JobModel
from infra.launch_origin import parse_launch_origin
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
    task_type: str = ""
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
    task_type: str = ""
    preset_name: str = ""
    eval_metric: str = ""
    selected_features: List[str] = Field(default_factory=list)
    handle_imbalance: bool = False
    auto_clean: bool = True
    cv_folds: int = 0
    pca_mode: str = "auto"
    pca_components: int = 0


class TrainingRegistryPreviewRequest(BaseModel):
    dataset_id: str
    target_column: str = ""
    goal: str = "Balanced"
    mode: str = "Balanced"
    task_type: str = ""
    eval_metric: str = ""
    selected_features: List[str] = Field(default_factory=list)
    handle_imbalance: bool = False


def _build_selector_profile(df: pd.DataFrame, target_column: str, selected_features: List[str]) -> dict:
    target_name = (target_column or "").strip()
    available_columns = [column for column in df.columns if str(column) != target_name]
    feature_names = list(selected_features or []) or available_columns
    filtered = [name for name in feature_names if name in df.columns and str(name) != target_name]
    feature_frame = df[filtered].copy() if filtered else pd.DataFrame(index=df.index)
    num_cols = list(feature_frame.select_dtypes(include="number").columns)
    cat_cols = [column for column in feature_frame.columns if column not in num_cols]
    target_series = df[target_name] if target_name and target_name in df.columns else pd.Series(dtype="object")
    numeric_max_corr = 0.0
    if len(num_cols) >= 2:
        numeric_frame = feature_frame[num_cols].apply(pd.to_numeric, errors="coerce")
        numeric_frame = numeric_frame.fillna(numeric_frame.median(numeric_only=True))
        try:
            corr = numeric_frame.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            finite = upper.to_numpy()
            finite = finite[np.isfinite(finite)]
            if finite.size:
                numeric_max_corr = float(finite.max())
        except Exception:
            numeric_max_corr = 0.0

    target_entropy = 0.0
    if not target_series.empty:
        clean_target = target_series.dropna()
        unique_count = int(clean_target.nunique())
        if unique_count > 1 and unique_count <= 20:
            probs = clean_target.astype(str).value_counts(normalize=True)
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            target_entropy = entropy / max(np.log(len(probs)), 1e-12)
        elif unique_count > 20:
            target_entropy = min(unique_count / max(len(clean_target), 1) * 20.0, 1.0)
    return {
        "rows": int(len(df)),
        "cols": int(len(feature_frame.columns)),
        "columns": list(feature_frame.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_entropy": round(float(target_entropy), 4),
        "numeric_max_corr": round(float(numeric_max_corr), 4),
    }


def _build_training_registry_payload(
    df_preview: pd.DataFrame,
    profile: dict,
    *,
    requested_target: str,
    task_type: str,
    goal: str,
    mode: str,
    eval_metric: str,
    selected_features: List[str],
    handle_imbalance: bool,
):
    from services.training.evaluator import detect_task_type, normalize_training_controls
    from services.training.model_selector import ModelSelector

    resolved_target = _resolve_target_column(
        df_preview,
        requested_target,
        profile.get("suggested_target", ""),
    )
    target_series = (
        df_preview[resolved_target]
        if resolved_target and resolved_target in df_preview.columns
        else pd.Series(dtype="object")
    )
    task_decision = detect_task_type(
        target_series,
        target_name=resolved_target,
        override=task_type,
    )
    controls = normalize_training_controls(
        task_type=task_decision["task_type"],
        goal=goal,
        mode=mode,
        eval_metric=eval_metric,
        handle_imbalance=handle_imbalance,
    )
    selector_profile = _build_selector_profile(
        df_preview,
        resolved_target,
        selected_features,
    )
    pool, recommendation = ModelSelector.select_pool(
        rows=selector_profile["rows"],
        is_clf=controls["task_type"] == "classification",
        goal=controls["goal"],
        profile=selector_profile,
        mode=controls["mode"],
    )
    goal_profile = recommendation.get("goal_profile") or {}
    selected_models = list(pool.keys())
    response = {
        "task_type": controls["task_type"],
        "requested_goal": controls["goal"],
        "selection_goal": goal_profile.get("goal", controls["goal"]),
        "mode": controls["mode"],
        "target_column": resolved_target,
        "selected_models": selected_models,
        "dataset_traits": goal_profile.get("dataset_traits") or {},
        "meta_advisory": {
            "reason": recommendation.get("reason", ""),
            "source": recommendation.get("source", ""),
            "confidence": recommendation.get("confidence", 0),
            "memory_applied": (recommendation.get("memory_signal") or {}).get("applied", False),
            "reordered_models": (recommendation.get("memory_signal") or {}).get("reordered_models", []),
        },
        "model_groups": {
            "baseline": [
                name for name in selected_models
                if name in {"Logistic Regression", "Linear Regression", "Ridge", "ElasticNet"}
            ],
            "boosting": [
                name for name in selected_models
                if name in {"Hist Gradient Boosting", "XGBoost", "LightGBM"}
            ],
            "optional": [
                name for name in selected_models
                if name in {"KNN", "SVM", "MLP", "Extra Trees"}
            ],
        },
        "rules": _registry_preview_notes(
            controls["task_type"],
            goal_profile.get("dataset_traits") or {},
            controls["mode"],
            selected_models,
        ),
    }
    return controls, sanitize_for_json(response)


def _registry_preview_notes(task_type: str, traits: dict, mode: str, selected_models: List[str]) -> List[str]:
    notes: List[str] = []
    if mode == "Full":
        notes.append("Full mode keeps the Performance model registry and spends more time on search and Optuna tuning.")
    memory_signal = traits.get("memory_signal") or {}
    if memory_signal.get("applied"):
        reordered = memory_signal.get("reordered_models") or []
        if reordered:
            notes.append(
                "Historical winner memory adjusted the order inside the safe model hierarchy: "
                + ", ".join(reordered[:4])
                + "."
            )
    if traits.get("low_complexity"):
        notes.append(
            f"Low dataset complexity detected (score {traits.get('complexity_score', 0):.3f}), so advanced boosters are skipped in favor of faster strong models."
        )
    if traits.get("small_dataset"):
        notes.append("Small dataset detected, so lightweight specialists like KNN and SVM can be considered.")
    if not traits.get("knn_allowed", True):
        notes.append("KNN is skipped because the feature count is above 50.")
    if traits.get("high_dimensional"):
        notes.append("High-dimensional feature space detected, so boosting and regularized linear models are prioritized.")
    if traits.get("very_large_dataset"):
        notes.append("Very large dataset detected, so SVM is skipped to avoid slow scaling.")
    if not traits.get("mlp_allowed", True):
        notes.append("MLP is skipped until the dataset reaches at least 2,000 rows.")
    if task_type == "regression":
        notes.append("Regression registry prefers ElasticNet over Lasso for a more flexible regularized baseline.")
    booster_count = sum(
        name in selected_models for name in ["Hist Gradient Boosting", "XGBoost", "LightGBM"]
    )
    if booster_count >= 2:
        notes.append("Boosting is already strongly represented, so Extra Trees stays optional instead of widening the sweep.")
    return notes


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
        task_type=req.task_type,
        pca_mode=req.pca_mode,
        pca_components=req.pca_components,
    )


@router.post("/train/model-registry")
def get_training_model_registry(req: TrainingRegistryPreviewRequest):
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
            detail=f"Could not read dataset for registry preview: {e}",
        ) from e

    _, response = _build_training_registry_payload(
        df_preview,
        profile,
        requested_target=req.target_column,
        task_type=req.task_type,
        goal=req.goal,
        mode=req.mode,
        eval_metric=req.eval_metric,
        selected_features=req.selected_features,
        handle_imbalance=req.handle_imbalance,
    )
    return response


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

    from services.training.evaluator import detect_task_type, normalize_training_controls

    task_decision = detect_task_type(
        df_preview[req.target_column] if req.target_column in df_preview.columns else pd.Series(dtype="object"),
        target_name=req.target_column,
        override=req.task_type,
    )
    normalized_controls = normalize_training_controls(
        task_type=task_decision["task_type"],
        goal=req.goal,
        mode=req.mode,
        eval_metric=req.eval_metric,
        handle_imbalance=req.handle_imbalance,
    )
    resolved_task_type = normalized_controls["task_type"]
    req.goal = normalized_controls["goal"]
    req.mode = normalized_controls["mode"]
    req.handle_imbalance = normalized_controls["handle_imbalance"]
    eval_metric = normalized_controls["eval_metric"]
    _, model_registry_preview = _build_training_registry_payload(
        df_preview,
        profile,
        requested_target=req.target_column,
        task_type=resolved_task_type,
        goal=req.goal,
        mode=req.mode,
        eval_metric=eval_metric,
        selected_features=req.selected_features,
        handle_imbalance=req.handle_imbalance,
    )

    full_params = {
        "target_column": req.target_column,
        "task_type": resolved_task_type,
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
        "normalization_warnings": normalized_controls["warnings"],
        "model_registry_preview": model_registry_preview,
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
        task_type=resolved_task_type,
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

        try:
            params = json.loads(job.params_json) if job.params_json else {}
        except Exception:
            params = {}

        return {
            "id": job.id,
            "status": job.status,
            "history": sanitize_for_json(history),
            "results": results,
            "insights": sanitize_for_json(insights),
            "reasoning": sanitize_for_json(reasoning),
            "config": sanitize_for_json(params),
            "model_registry_preview": sanitize_for_json(params.get("model_registry_preview")),
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
                params = json.loads(job.params_json) if job.params_json else {}
            except Exception:
                params = {}
            launch_origin = parse_launch_origin(params)
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
                    "is_classification": results_data.get("is_classification"),
                    "target": results_data.get("target"),
                    "feature_names": results_data.get("feature_names") or [],
                    "error": job.error,
                    "launch_source": launch_origin["launch_source"],
                    "launch_label": launch_origin["launch_label"],
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
                params = json.loads(job.params_json) if job.params_json else {}
            except Exception:
                params = {}
            launch_origin = parse_launch_origin(params)
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
                            "launch_source": launch_origin["launch_source"],
                            "launch_label": launch_origin["launch_label"],
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

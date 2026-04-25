from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.file_loader import load_dataframe
from infra.database import DatasetModel, ExperimentRun, JobModel, WorkspaceModel, NotificationModel, db_session
from services.data_sanitizer import build_dataset_version_report


def list_datasets(limit: int = 100, include_archived: bool = False) -> List[Dict[str, Any]]:
    with db_session() as db:
        rows = (
            db.query(DatasetModel)
            .order_by(DatasetModel.created_at.desc())
            .limit(limit)
            .all()
        )
        items: List[Dict[str, Any]] = []
        for row in rows:
            try:
                profile = json.loads(row.profile_json) if row.profile_json else {}
            except Exception:
                profile = {}
            archived = bool(profile.get("archived")) or str(row.source_type or "").startswith("archived:")
            if archived and not include_archived:
                continue
            items.append(
                {
                    "id": row.id,
                    "source_type": row.source_type or "unknown",
                    "display_name": row.display_name,
                    "archived": archived,
                    "parent_dataset_id": row.parent_dataset_id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "rows": profile.get("rows"),
                    "cols": profile.get("cols"),
                    "missing_pct": profile.get("missing_pct"),
                    "columns": profile.get("columns") or [],
                    "suggested_target": profile.get("suggested_target"),
                }
            )
        return items


def _dataset_snapshot(row: DatasetModel | None) -> Dict[str, Any] | None:
    if not row:
        return None
    try:
        profile = json.loads(row.profile_json) if row.profile_json else {}
    except Exception:
        profile = {}

    preview_records: List[Dict[str, Any]] = []
    try:
        if str(row.file_path or "").lower().endswith(".csv"):
            df = pd.read_csv(row.file_path, nrows=8)
        else:
            df = load_dataframe(filepath=row.file_path)
        if df is not None and not df.empty:
            preview_records = json.loads(
                df.head(8).to_json(orient="records", date_format="iso")
            )
    except Exception:
        preview_records = []

    return {
        "id": row.id,
        "source_type": row.source_type or "unknown",
        "parent_dataset_id": row.parent_dataset_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "profile": profile,
        "preview_records": preview_records,
        "ingest_summary": {
            "source_type": row.source_type or "unknown",
            "rows": int(profile.get("rows") or 0),
            "columns": int(
                profile.get("cols") or len(profile.get("columns") or [])
            ),
            "column_names": profile.get("columns") or [],
        },
        "auto_detect": {
            "suggested_target": profile.get("suggested_target"),
            "task_type": profile.get("task_type"),
            "confidence": profile.get("confidence", 0),
            "warnings": [],
        },
    }


def _job_snapshot(row: JobModel | None) -> Dict[str, Any] | None:
    if not row:
        return None
    return {
        "id": row.id,
        "dataset_id": row.dataset_id,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "error": row.error,
    }


def get_workspace_snapshot(
    dataset_id: str | None = None,
    job_id: str | None = None,
    
) -> Dict[str, Any]:
    with db_session() as db:
        dataset_row = None
        job_row = None
        workspace_row = None

        if job_id:
            job_row = db.query(JobModel).filter(JobModel.id == job_id).first()
            if job_row and not dataset_id:
                dataset_id = job_row.dataset_id

        if dataset_id:
            dataset_row = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        else:
            dataset_row = (
                db.query(DatasetModel)
                .order_by(DatasetModel.created_at.desc())
                .first()
            )

        if not job_row and dataset_row:
            job_row = (
                db.query(JobModel)
                .filter(JobModel.dataset_id == dataset_row.id)
                .order_by(JobModel.created_at.desc())
                .first()
            )

        if not job_row:
            job_row = (
                db.query(JobModel).order_by(JobModel.created_at.desc()).first()
            )

        if job_row and job_row.params_json:
            try:
                params = json.loads(job_row.params_json)
            except Exception:
                params = {}
            workspace_id = params.get("workspace_id")
            workspace_name = params.get("workspace_name")
            if workspace_id:
                workspace_row = db.query(WorkspaceModel).filter(WorkspaceModel.id == workspace_id).first()
            elif workspace_name:
                workspace_row = db.query(WorkspaceModel).filter(WorkspaceModel.name == workspace_name).first()

        return {
            "workspace": {
                "id": workspace_row.id,
                "name": workspace_row.name,
                "dataset_id": workspace_row.dataset_id,
                "last_job_id": workspace_row.last_job_id,
                "last_run_id": workspace_row.last_run_id,
            } if workspace_row else None,
            "dataset": _dataset_snapshot(dataset_row),
            "job": _job_snapshot(job_row),
        }


def compare_dataset_versions(dataset_id: str, target: str | None = None) -> Dict[str, Any]:
    with db_session() as db:
        current = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not current:
            return {"error": "Dataset not found"}
        previous = None
        if current.parent_dataset_id:
            previous = db.query(DatasetModel).filter(DatasetModel.id == current.parent_dataset_id).first()
        if not previous:
            previous = (
                db.query(DatasetModel)
                .filter(DatasetModel.id != dataset_id)
                .order_by(DatasetModel.created_at.desc())
                .first()
            )
        if not previous:
            return {"error": "No previous dataset version found"}

    current_df = load_dataframe(filepath=current.file_path)
    previous_df = load_dataframe(filepath=previous.file_path)
    if current_df is None or previous_df is None:
        return {"error": "Could not load dataset versions"}

    report = build_dataset_version_report(current_df, previous_df, target=target)
    report.update(
        {
            "dataset_id": dataset_id,
            "current_dataset_id": current.id,
            "previous_dataset_id": previous.id,
            "current_name": current.display_name,
            "previous_name": previous.display_name,
        }
    )
    return report


def list_notifications(limit: int = 50) -> Dict[str, Any]:
    with db_session() as db:
        rows = db.query(NotificationModel).order_by(NotificationModel.created_at.desc()).limit(limit).all()
    return {
        "notifications": [
            {
                "id": row.id,
                "entity_type": row.entity_type,
                "entity_id": row.entity_id,
                "title": row.title,
                "message": row.message,
                "level": row.level,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }


def merge_preview(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    join_type: str = "inner",
) -> Dict[str, Any]:
    left_df = left_df.copy()
    right_df = right_df.copy()
    left_key_series = left_df[left_key]
    right_key_series = right_df[right_key]
    coerced_keys = False

    if str(left_key_series.dtype) != str(right_key_series.dtype):
        left_df[left_key] = left_key_series.astype("string")
        right_df[right_key] = right_key_series.astype("string")
        left_key_series = left_df[left_key]
        right_key_series = right_df[right_key]
        coerced_keys = True

    left_non_null = left_key_series.dropna()
    right_non_null = right_key_series.dropna()
    left_unique = set(left_non_null.astype(str))
    right_unique = set(right_non_null.astype(str))
    overlap = left_unique.intersection(right_unique)

    left_match_pct = round(len(overlap) / max(len(left_unique), 1) * 100, 1)
    right_match_pct = round(len(overlap) / max(len(right_unique), 1) * 100, 1)

    merged = pd.merge(
        left_df,
        right_df,
        left_on=left_key,
        right_on=right_key,
        how=join_type if join_type in {"inner", "left", "right", "outer"} else "inner",
        suffixes=("_left", "_right"),
        indicator=True,
    )

    preview_records = json.loads(
        merged.head(8).drop(columns=["_merge"], errors="ignore").to_json(
            orient="records", date_format="iso"
        )
    )

    merge_counts = merged["_merge"].value_counts(dropna=False).to_dict()
    estimated_row_multiplier = round(len(merged) / max(len(left_df), 1), 2)

    return {
        "left_rows": int(len(left_df)),
        "right_rows": int(len(right_df)),
        "estimated_rows": int(len(merged)),
        "estimated_row_multiplier": estimated_row_multiplier,
        "left_unique_keys": int(left_non_null.astype(str).nunique()),
        "right_unique_keys": int(right_non_null.astype(str).nunique()),
        "overlapping_keys": int(len(overlap)),
        "left_duplicate_keys": int(left_non_null.astype(str).duplicated().sum()),
        "right_duplicate_keys": int(right_non_null.astype(str).duplicated().sum()),
        "left_match_pct": left_match_pct,
        "right_match_pct": right_match_pct,
        "join_key_coerced_to_string": coerced_keys,
        "left_key_dtype": str(left_df[left_key].dtype),
        "right_key_dtype": str(right_df[right_key].dtype),
        "join_breakdown": merge_counts,
        "preview_records": preview_records,
    }


def build_lineage_graph(dataset_id: str) -> Dict[str, Any]:
    with db_session() as db:
        current = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not current:
            return {"error": "Dataset not found"}

        nodes: List[Dict[str, Any]] = []
        seen = set()
        cursor = current

        while cursor and cursor.id not in seen:
            seen.add(cursor.id)
            try:
                profile = json.loads(cursor.profile_json) if cursor.profile_json else {}
            except Exception:
                profile = {}

            # Add Training Jobs for this dataset
            jobs = db.query(JobModel).filter(JobModel.dataset_id == cursor.id).all()
            for job in jobs:
                nodes.append({
                    "id": f"job-{job.id}",
                    "type": "Training Mission",
                    "label": f"AutoML: {job.id[:8]}",
                    "detail": f"Status: {job.status} • Result: {job.error or 'Success'}",
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "is_job": True
                })

            # Add Dataset node
            nodes.append(
                {
                    "id": cursor.id,
                    "type": "Data Ingestion",
                    "label": f"{(cursor.source_type or 'dataset').replace('_', ' ').title()}",
                    "detail": f"{profile.get('rows', '—')} rows • {profile.get('cols', '—')} columns",
                    "created_at": cursor.created_at.isoformat() if cursor.created_at else None,
                    "is_dataset": True
                }
            )

            if cursor.parent_dataset_id:
                cursor = (
                    db.query(DatasetModel)
                    .filter(DatasetModel.id == cursor.parent_dataset_id)
                    .first()
                )
            else:
                cursor = None

    # Sort by creation date to show lineage flow
    nodes.sort(key=lambda x: x.get("created_at") or "", reverse=False)
    
    return {"dataset_id": dataset_id, "nodes": nodes}


def synthetic_data_judge(dataset_id: str) -> Dict[str, Any]:
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}
        if not dataset.parent_dataset_id:
            return {"error": "Synthetic judge needs a derived dataset with a parent dataset."}

        parent = (
            db.query(DatasetModel)
            .filter(DatasetModel.id == dataset.parent_dataset_id)
            .first()
        )
        if not parent:
            return {"error": "Parent dataset not found"}

    current_df = load_dataframe(filepath=dataset.file_path)
    parent_df = load_dataframe(filepath=parent.file_path)
    if current_df is None or current_df.empty or parent_df is None or parent_df.empty:
        return {"error": "Could not load datasets for judging"}

    extra_rows = max(len(current_df) - len(parent_df), 0)
    synthetic_df = current_df.tail(extra_rows).copy() if extra_rows else current_df.copy()

    numeric_cols = [
        col for col in synthetic_df.select_dtypes(include=[np.number]).columns if col in parent_df.columns
    ]
    cat_cols = [
        col for col in synthetic_df.select_dtypes(include=["object", "category", "bool"]).columns if col in parent_df.columns
    ]

    notes: List[str] = []
    realism_score = 100.0

    duplicate_ratio = float(synthetic_df.duplicated().mean()) if not synthetic_df.empty else 0.0
    if duplicate_ratio > 0.08:
        realism_score -= min(25.0, duplicate_ratio * 80.0)
        notes.append(f"Duplicate synthetic rows are too common ({round(duplicate_ratio * 100, 1)}%).")

    missing_deltas: List[float] = []
    for col in synthetic_df.columns:
        if col not in parent_df.columns:
            continue
        base_missing = float(parent_df[col].isna().mean())
        synth_missing = float(synthetic_df[col].isna().mean())
        delta = abs(synth_missing - base_missing)
        missing_deltas.append(delta)
        if delta > 0.2:
            realism_score -= min(10.0, delta * 20.0)
            notes.append(f"{col}: missing-value behavior drifted from the original dataset.")

    for col in numeric_cols[:40]:
        base = pd.to_numeric(parent_df[col], errors="coerce").dropna()
        synth = pd.to_numeric(synthetic_df[col], errors="coerce").dropna()
        if base.empty or synth.empty:
            continue
        mean_shift = abs(float(synth.mean()) - float(base.mean())) / max(float(base.std()) or 1.0, 1.0)
        if mean_shift > 1.0:
            realism_score -= min(12.0, mean_shift * 8.0)
            notes.append(f"{col}: numeric mean shifted more than expected.")

    for col in cat_cols[:30]:
        base_freq = parent_df[col].astype(str).value_counts(normalize=True, dropna=True)
        synth_freq = synthetic_df[col].astype(str).value_counts(normalize=True, dropna=True)
        overlap = set(base_freq.index).intersection(set(synth_freq.index))
        coverage = sum(float(min(base_freq.get(k, 0), synth_freq.get(k, 0))) for k in overlap)
        if coverage < 0.7:
            realism_score -= (0.7 - coverage) * 20
            notes.append(f"{col}: category mix drifted from the original distribution.")

    realism_score = round(max(0.0, min(100.0, realism_score)), 1)
    verdict = "Strong" if realism_score >= 85 else "Usable" if realism_score >= 70 else "Review before retraining"
    return {
        "dataset_id": dataset_id,
        "parent_dataset_id": dataset.parent_dataset_id,
        "realism_score": realism_score,
        "verdict": verdict,
        "rows_evaluated": int(len(synthetic_df)),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "avg_missing_delta": round(float(np.mean(missing_deltas)) if missing_deltas else 0.0, 4),
        "notes": notes[:10],
    }


def _json_load(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def experiment_diff(run_a_id: str, run_b_id: str) -> Dict[str, Any]:
    with db_session() as db:
        run_a = db.query(ExperimentRun).filter(ExperimentRun.id == run_a_id).first()
        run_b = db.query(ExperimentRun).filter(ExperimentRun.id == run_b_id).first()
        if not run_a and run_a_id:
            run_a = db.query(ExperimentRun).filter(ExperimentRun.job_id == run_a_id).first()
        if not run_b and run_b_id:
            run_b = db.query(ExperimentRun).filter(ExperimentRun.job_id == run_b_id).first()
        if not run_a or not run_b:
            return {"error": "One or both experiment runs were not found"}

    a_params = _json_load(run_a.hyperparams_json, {})
    b_params = _json_load(run_b.hyperparams_json, {})
    a_metrics = _json_load(run_a.metrics_json, {})
    b_metrics = _json_load(run_b.metrics_json, {})

    def scalar_diff(left: Dict[str, Any], right: Dict[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key in keys:
            if left.get(key) != right.get(key):
                rows.append({"field": key, "before": left.get(key), "after": right.get(key)})
        return rows

    config_changes = scalar_diff(
        {
            "model_name": run_a.model_name,
            "mode": run_a.mode,
            "goal": run_a.goal,
            "metric_name": run_a.metric_name,
            "dataset_id": run_a.dataset_id,
            **a_params,
        },
        {
            "model_name": run_b.model_name,
            "mode": run_b.mode,
            "goal": run_b.goal,
            "metric_name": run_b.metric_name,
            "dataset_id": run_b.dataset_id,
            **b_params,
        },
        [
            "model_name",
            "mode",
            "goal",
            "metric_name",
            "dataset_id",
            "cv_folds",
            "selected_features",
            "handle_imbalance",
            "auto_clean",
        ],
    )

    output_changes = scalar_diff(
        {
            "score": run_a.score,
            "feature_count": run_a.feature_count,
            "row_count": run_a.row_count,
            "best_model": a_metrics.get("best_model"),
            "preprocessor": (a_metrics.get("model_metadata") or {}).get("preprocessor"),
            "pca_applied": (a_metrics.get("eda_summary") or {}).get("pca_applied"),
        },
        {
            "score": run_b.score,
            "feature_count": run_b.feature_count,
            "row_count": run_b.row_count,
            "best_model": b_metrics.get("best_model"),
            "preprocessor": (b_metrics.get("model_metadata") or {}).get("preprocessor"),
            "pca_applied": (b_metrics.get("eda_summary") or {}).get("pca_applied"),
        },
        ["score", "feature_count", "row_count", "best_model", "preprocessor", "pca_applied"],
    )

    explanations: List[str] = []
    if run_a.dataset_id != run_b.dataset_id:
        explanations.append("The compared runs used different datasets.")
    if run_a.mode != run_b.mode:
        explanations.append(f"Execution mode changed from {run_a.mode} to {run_b.mode}.")
    if run_a.model_name != run_b.model_name:
        explanations.append(f"The winning model changed from {run_a.model_name} to {run_b.model_name}.")
    try:
        score_delta = round(float(run_b.score or 0) - float(run_a.score or 0), 2)
        explanations.append(f"Primary score moved by {score_delta} points.")
    except Exception:
        pass

    return {
        "run_a": {"id": run_a.id, "model_name": run_a.model_name, "score": run_a.score},
        "run_b": {"id": run_b.id, "model_name": run_b.model_name, "score": run_b.score},
        "config_changes": config_changes,
        "output_changes": output_changes,
        "explanations": explanations,
    }


def build_trust_heatmap(dataset_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    feature_names = results.get("feature_names") or []
    shap_summary = results.get("shap_summary") or {}

    with db_session() as db:
        runs = (
            db.query(ExperimentRun)
            .filter(ExperimentRun.dataset_id == dataset_id)
            .order_by(ExperimentRun.created_at.desc())
            .limit(25)
            .all()
        )
        run_metrics = [_json_load(run.metrics_json, {}) for run in runs]
    rows: List[Dict[str, Any]] = []

    for feature in feature_names[:60]:
        historical_presence = 0
        historical_importance: List[float] = []
        for metrics in run_metrics:
            feats = metrics.get("feature_names") or []
            if feature in feats:
                historical_presence += 1
            shap_val = (metrics.get("shap_summary") or {}).get(feature)
            if shap_val is not None:
                try:
                    historical_importance.append(abs(float(shap_val)))
                except Exception:
                    pass

        current_importance = 0.0
        if feature in shap_summary:
            try:
                current_importance = abs(float(shap_summary[feature]))
            except Exception:
                current_importance = 0.0

        leakage_risk = any(token in feature.lower() for token in ("target", "label", "id", "uuid"))
        presence_ratio = historical_presence / max(len(run_metrics), 1) if run_metrics else 1.0
        variability = float(np.std(historical_importance)) if historical_importance else 0.0

        status = "stable"
        if leakage_risk:
            status = "leakage-risky"
        elif presence_ratio < 0.35:
            status = "drift-prone"
        elif variability > max(current_importance, 1e-6):
            status = "noisy"

        rows.append(
            {
                "feature": feature,
                "status": status,
                "importance": round(current_importance, 6),
                "historical_presence_pct": round(presence_ratio * 100, 1),
                "importance_variability": round(variability, 6),
                "leakage_risk": leakage_risk,
            }
        )

    return {"rows": rows, "run_count": len(run_metrics)}


def narrate_experiment(profile: Dict[str, Any], results: Dict[str, Any], story: str | None = None) -> Dict[str, Any]:
    if story:
        concise = story.strip().replace("\n", " ")
        return {"narrative": concise}

    model = results.get("best_model", "Unknown model")
    score = results.get("score", "—")
    metric = results.get("metric_name", "Score")
    rows = profile.get("rows", "—")
    cols = profile.get("cols", "—")
    execution = results.get("execution_profile") or {}
    eda = results.get("eda_summary") or {}

    pieces = [
        f"{model} won on a dataset with {rows} rows and {cols} columns",
        f"finishing with {metric} {score}",
    ]
    if eda.get("pca_applied"):
        pieces.append(
            f"after dimensionality reduction to {eda.get('pca_components_used')} PCA components"
        )
    if execution.get("run_optuna"):
        pieces.append(
            f"and survived a deeper optimization pass across {execution.get('top_k')} shortlisted candidates"
        )
    return {"narrative": ". ".join(pieces) + "."}

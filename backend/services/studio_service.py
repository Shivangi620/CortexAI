from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from core.file_loader import load_dataframe
from infra.database import DatasetModel, ExperimentRun, get_db


def list_datasets(limit: int = 100) -> List[Dict[str, Any]]:
    with get_db() as db:
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
            items.append(
                {
                    "id": row.id,
                    "source_type": row.source_type or "unknown",
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


def merge_preview(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    join_type: str = "inner",
) -> Dict[str, Any]:
    left_key_series = left_df[left_key]
    right_key_series = right_df[right_key]

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
        "join_breakdown": merge_counts,
        "preview_records": preview_records,
    }


def build_lineage_graph(dataset_id: str) -> Dict[str, Any]:
    with get_db() as db:
        current = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not current:
            return {"error": "Dataset not found"}

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen = set()
        cursor = current

        while cursor and cursor.id not in seen:
            seen.add(cursor.id)
            try:
                profile = json.loads(cursor.profile_json) if cursor.profile_json else {}
            except Exception:
                profile = {}

            nodes.append(
                {
                    "id": cursor.id,
                    "label": f"{(cursor.source_type or 'dataset').replace('_', ' ').title()}",
                    "source_type": cursor.source_type or "unknown",
                    "rows": profile.get("rows"),
                    "cols": profile.get("cols"),
                    "missing_pct": profile.get("missing_pct"),
                    "created_at": cursor.created_at.isoformat() if cursor.created_at else None,
                }
            )
            if cursor.parent_dataset_id:
                edges.append(
                    {
                        "source": cursor.parent_dataset_id,
                        "target": cursor.id,
                        "label": cursor.source_type or "derived",
                    }
                )
                cursor = (
                    db.query(DatasetModel)
                    .filter(DatasetModel.id == cursor.parent_dataset_id)
                    .first()
                )
            else:
                cursor = None

    nodes.reverse()
    edges.reverse()
    return {"dataset_id": dataset_id, "nodes": nodes, "edges": edges}


def synthetic_data_judge(dataset_id: str) -> Dict[str, Any]:
    with get_db() as db:
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
        "notes": notes[:10],
    }


def _json_load(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def experiment_diff(run_a_id: str, run_b_id: str) -> Dict[str, Any]:
    with get_db() as db:
        run_a = db.query(ExperimentRun).filter(ExperimentRun.id == run_a_id).first()
        run_b = db.query(ExperimentRun).filter(ExperimentRun.id == run_b_id).first()
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

    with get_db() as db:
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

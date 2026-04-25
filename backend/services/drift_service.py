"""
services/drift_service.py
Feature 6: Model drift dashboard — per-feature PSI + KS + severity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def _build_retrain_profile(current_df: pd.DataFrame, feature_names: List[str], target_name: str = None) -> Dict[str, Any]:
    requested_features = [
        str(feature)
        for feature in (feature_names or [])
        if str(feature).strip() and str(feature) != str(target_name)
    ]
    if requested_features:
        feature_frame = current_df[[col for col in requested_features if col in current_df.columns]].copy()
    else:
        feature_frame = current_df.drop(columns=[target_name], errors="ignore").copy()

    num_cols = list(feature_frame.select_dtypes(include="number").columns)
    cat_cols = [column for column in feature_frame.columns if column not in num_cols]
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

    return {
        "rows": int(len(feature_frame)),
        "cols": int(len(feature_frame.columns)),
        "columns": list(feature_frame.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_entropy": 0.0,
        "numeric_max_corr": round(float(numeric_max_corr), 4),
    }


def _build_retrain_recommendation(
    current_df: pd.DataFrame,
    feature_names: List[str],
    *,
    target_name: str = None,
    task_type: str = "",
    current_model: str = "",
    metric_name: str = "",
    current_score: float | int | None = None,
    current_validation_gap: float | int | None = None,
    goal: str = "",
    mode: str = "",
    alert_level: str = "stable",
    concept_drift_detected: bool = False,
) -> Dict[str, Any]:
    from core.meta_learning import get_cross_dataset_insights
    from services.training.model_selector import ModelSelector

    profile = _build_retrain_profile(current_df, feature_names, target_name=target_name)
    is_classification = str(task_type or "").strip().lower() == "classification"

    if concept_drift_detected or alert_level == "critical":
        recommended_goal = "Performance"
        recommended_mode = "Full"
    elif alert_level == "warning":
        recommended_goal = "Balanced"
        recommended_mode = "Balanced"
    else:
        recommended_goal = (goal or "Balanced").strip() or "Balanced"
        recommended_mode = (mode or "Balanced").strip() or "Balanced"

    pool, recommendation = ModelSelector.select_pool(
        rows=profile["rows"],
        is_clf=is_classification,
        goal=recommended_goal,
        profile=profile,
        mode=recommended_mode,
    )
    selected_models = list(pool.keys())
    historical = get_cross_dataset_insights(profile)
    historical_winner = ((historical or {}).get("most_common_winner") or {}).get("model", "")

    if current_model and current_model in selected_models:
        posture = f"Keep '{current_model}' in the retrain challenger set and compare it against the strongest reopen candidates."
    elif current_model:
        posture = f"Re-open the model search beyond '{current_model}' because the current winner no longer looks like the safest default under drift."
    else:
        posture = "Open a fresh retrain lane with the strongest candidates first."

    candidate_models = list(selected_models[:5])
    if "MLP" in selected_models and "MLP" not in candidate_models:
        if len(candidate_models) >= 5:
            candidate_models[-1] = "MLP"
        else:
            candidate_models.append("MLP")

    return {
        "recommended_goal": recommended_goal,
        "recommended_mode": recommended_mode,
        "metric_name": metric_name,
        "current_model": current_model or "",
        "current_score": current_score,
        "current_validation_gap": current_validation_gap,
        "candidate_models": candidate_models,
        "historical_winner": historical_winner,
        "historical_runs": int((historical or {}).get("historical_runs") or 0),
        "memory_confidence": recommendation.get("confidence", 0),
        "memory_applied": (recommendation.get("memory_signal") or {}).get("applied", False),
        "memory_reordered_models": (recommendation.get("memory_signal") or {}).get("reordered_models", []),
        "message": posture,
    }


def get_drift_dashboard(
    current_df: pd.DataFrame,
    baseline_stats: Dict[str, Any],
    feature_names: List[str],
    target_name: str = None,
    timestamp: str = None,
    baseline_version: str = "v1.0",
    warning_threshold: float | None = None,
    critical_threshold: float | None = None,
    task_type: str = "",
    current_model: str = "",
    metric_name: str = "",
    current_score: float | int | None = None,
    current_validation_gap: float | int | None = None,
    goal: str = "",
    mode: str = "",
) -> Dict[str, Any]:
    from core.drift_detector import DriftDetector
    from datetime import datetime, UTC
    from infra.config import settings

    feature_drift: List[Dict[str, Any]] = []
    drifted_features: List[str] = []
    critical_features: List[str] = []

    timestamp = timestamp or datetime.now(UTC).isoformat()

    baseline_lookup = baseline_stats or {}
    requested_features = [
        str(feature)
        for feature in (feature_names or [])
        if str(feature).strip() and str(feature) != str(target_name)
    ]
    try:
        if requested_features:
            cols_to_check = [
                col
                for col in requested_features
                if col in current_df.columns and col in baseline_lookup
            ]
        else:
            cols_to_check = [
                c
                for c in current_df.columns
                if c in baseline_lookup and c != target_name
            ]
    except Exception:
        cols_to_check = []

    concept_drift_detected = False
    warning_threshold = (
        float(warning_threshold)
        if warning_threshold is not None
        else float(settings.psi_warning_threshold)
    )
    critical_threshold = (
        float(critical_threshold)
        if critical_threshold is not None
        else float(settings.psi_critical_threshold)
    )

    for col in cols_to_check:
        col_stats = baseline_lookup.get(col, {}) if isinstance(baseline_lookup, dict) else {}
        current_series = current_df[col].dropna()

        if current_series.empty:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(current_series)

        entry: Dict[str, Any] = {
            "feature": col,
            "is_numeric": is_numeric,
            "current_mean": None,
            "baseline_mean": None,
            "psi": None,
            "ks_p_value": None,
            "drift_detected": False,
            "severity": "✅ Stable",
        }

        if is_numeric:
            # Reconstruct or load baseline values
            if isinstance(col_stats, list) and len(col_stats) > 0:
                baseline_vals = np.array(col_stats)
                b_mean = float(baseline_vals.mean())
                b_std = float(baseline_vals.std())
                b_count = len(col_stats)
            elif isinstance(col_stats, dict):
                b_mean = col_stats.get("mean", current_series.mean())
                b_std = col_stats.get("std", current_series.std()) or 1
                b_count = int(col_stats.get("count", 100) or 100)
                np.random.seed(42)
                baseline_vals = np.random.normal(b_mean, b_std, b_count)
            else:
                baseline_vals = current_series.values
                b_mean = float(current_series.mean())
                b_std = float(current_series.std())
                b_count = len(current_series)

            entry["current_mean"] = round(float(current_series.mean()), 4)
            entry["baseline_mean"] = round(float(b_mean), 4)
            entry["current_std"] = round(float(current_series.std()), 4)
            entry["baseline_std"] = round(float(b_std), 4)

            try:
                drifted, p_val = DriftDetector.compute_ks_drift(
                    baseline_vals, current_series.values
                )
                entry["ks_p_value"] = round(float(p_val), 4)
            except Exception:
                p_val = 1.0

            try:
                psi = DriftDetector.compute_psi(
                    baseline_vals, current_series.values
                )
                entry["psi"] = round(float(psi), 4)
            except Exception:
                psi = 0.0

            is_drifted = False

            if psi >= critical_threshold or p_val < settings.ks_p_threshold:
                entry["drift_detected"] = True
                is_drifted = True

                if psi >= critical_threshold:
                    entry["severity"] = "🔴 Critical Drift"
                    critical_features.append(col)
                else:
                    entry["severity"] = "🟡 Moderate Drift"

                drifted_features.append(col)

            elif psi >= warning_threshold:
                entry["severity"] = "🟡 Moderate Drift"
                entry["drift_detected"] = True
                is_drifted = True
                drifted_features.append(col)

            if is_drifted and target_name and col == target_name:
                concept_drift_detected = True

        feature_drift.append(entry)

    try:
        feature_drift.sort(
            key=lambda x: (
                not x.get("drift_detected", False),
                x.get("psi") or 0
            ),
            reverse=False
        )
    except Exception:
        pass

    total = len(feature_drift)
    drifted_count = len(set(drifted_features))

    try:
        drift_score = round(drifted_count / max(total, 1) * 100, 1)
    except Exception:
        drift_score = 0.0

    overall_status = (
        "🔴 Critical Drift Detected" if critical_features else
        "🟡 Moderate Drift Detected" if drifted_features else
        "✅ No Significant Drift"
    )

    alert_message = ""
    if concept_drift_detected:
        alert_message = f"🚨 Concept Drift Detected on target '{target_name}'! Model retraining highly recommended."
    elif critical_features:
        alert_message = f"Critical data drift in: {', '.join(critical_features[:5])}"
    elif drifted_features:
        alert_message = f"Moderate data drift in: {', '.join(drifted_features[:5])}"

    alert_level = (
        "critical"
        if critical_features
        else "warning"
        if drifted_features
        else "stable"
    )
    recommended_action = (
        "Retrain soon and review the affected features before promoting new predictions."
        if critical_features
        else "Monitor closely and schedule a fresh drift check after the next data refresh."
        if drifted_features
        else "No urgent action is needed. Continue with the saved review cadence."
    )
    retrain_recommendation = _build_retrain_recommendation(
        current_df,
        feature_names,
        target_name=target_name,
        task_type=task_type,
        current_model=current_model,
        metric_name=metric_name,
        current_score=current_score,
        current_validation_gap=current_validation_gap,
        goal=goal,
        mode=mode,
        alert_level=alert_level,
        concept_drift_detected=concept_drift_detected,
    )

    return {
        "timestamp": timestamp,
        "baseline_version": baseline_version,
        "overall_status": overall_status,
        "drift_score_pct": drift_score,
        "concept_drift_detected": concept_drift_detected,
        "drifted_features": list(set(drifted_features)),
        "critical_features": critical_features,
        "alert_message": alert_message,
        "alert_level": alert_level,
        "recommended_action": recommended_action,
        "thresholds": {
            "warning_psi": round(float(warning_threshold), 4),
            "critical_psi": round(float(critical_threshold), 4),
        },
        "alert_summary": {
            "level": alert_level,
            "headline": overall_status,
            "message": alert_message or "No significant drift was detected.",
            "recommended_action": recommended_action,
        },
        "retrain_recommendation": retrain_recommendation,
        "feature_drift": feature_drift,
        "total_features_checked": total,
        "drifted_count": drifted_count,
    }

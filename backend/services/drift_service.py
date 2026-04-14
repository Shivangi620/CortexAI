"""
services/drift_service.py
Feature 6: Model drift dashboard — per-feature PSI + KS + severity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def get_drift_dashboard(
    current_df: pd.DataFrame,
    baseline_stats: Dict[str, Any],
    feature_names: List[str],
    target_name: str = None,
    timestamp: str = None,
    baseline_version: str = "v1.0"
) -> Dict[str, Any]:
    from core.drift_detector import DriftDetector
    from datetime import datetime

    feature_drift: List[Dict[str, Any]] = []
    drifted_features: List[str] = []
    critical_features: List[str] = []

    timestamp = timestamp or datetime.utcnow().isoformat()

    try:
        cols_to_check = [c for c in current_df.columns if c in (baseline_stats or {})]
    except Exception:
        cols_to_check = []

    concept_drift_detected = False

    for col in cols_to_check:
        col_stats = baseline_stats.get(col, {}) if isinstance(baseline_stats, dict) else {}
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
            try:
                entry["current_mean"] = round(float(current_series.mean()), 4)
                entry["baseline_mean"] = round(float(col_stats.get("mean", current_series.mean())), 4)
                entry["current_std"] = round(float(current_series.std()), 4)
                entry["baseline_std"] = round(float(col_stats.get("std", current_series.std())), 4)
            except Exception:
                pass

            b_mean = col_stats.get("mean", current_series.mean())
            b_std = col_stats.get("std", current_series.std()) or 1
            b_count = int(col_stats.get("count", 100) or 100)

            try:
                np.random.seed(42)
                baseline_approx = pd.Series(np.random.normal(b_mean, b_std, b_count))
            except Exception:
                baseline_approx = current_series

            try:
                drifted, p_val = DriftDetector.compute_ks_drift(
                    baseline_approx.values, current_series.values
                )
                entry["ks_p_value"] = round(float(p_val), 4)
            except Exception:
                drifted, p_val = False, 1.0

            try:
                psi = DriftDetector.compute_psi(
                    baseline_approx.values, current_series.values
                )
                entry["psi"] = round(float(psi), 4)
            except Exception:
                psi = 0.0

            from infra.config import settings

            is_drifted = False

            if psi >= settings.psi_critical_threshold or p_val < settings.ks_p_threshold:
                entry["drift_detected"] = True
                is_drifted = True

                if psi >= settings.psi_critical_threshold:
                    entry["severity"] = "🔴 Critical Drift"
                    critical_features.append(col)
                else:
                    entry["severity"] = "🟡 Moderate Drift"

                drifted_features.append(col)

            elif psi >= settings.psi_warning_threshold:
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

    return {
        "timestamp": timestamp,
        "baseline_version": baseline_version,
        "overall_status": overall_status,
        "drift_score_pct": drift_score,
        "concept_drift_detected": concept_drift_detected,
        "drifted_features": list(set(drifted_features)),
        "critical_features": critical_features,
        "alert_message": alert_message,
        "feature_drift": feature_drift,
        "total_features_checked": total,
        "drifted_count": drifted_count,
    }
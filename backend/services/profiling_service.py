"""
services/profiling_service.py
Auto problem type detection (Feature 2) and dataset profiling helpers.
"""
from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any, List


def detect_problem_type(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    rows = len(df)
    cols = df.columns.tolist()

    # ── 1. Suggest target column ───────────────────────────────────────────────
    if target_column and target_column in df.columns:
        suggested = target_column
    else:
        suggested = _suggest_target(df)

    # ── 2. Detect task type ────────────────────────────────────────────────────
    if suggested and suggested in df.columns:
        task_type, confidence, task_reason = _detect_task(df[suggested])
    else:
        task_type, confidence, task_reason = "classification", 50, "Fallback default"

    # ── 3. Dataset warnings ────────────────────────────────────────────────────
    if rows < 100:
        warnings.append(f"⚠️ Very small dataset ({rows} rows). Model will likely overfit.")
    elif rows < 300:
        warnings.append(f"🟡 Small dataset ({rows} rows). Consider collecting more data.")

    try:
        missing_pct = float(profile.get("missing_pct", 0) or 0)
    except Exception:
        missing_pct = 0.0

    if missing_pct > 40:
        warnings.append(f"⚠️ High missing values ({missing_pct:.1f}%). Imputation may introduce bias.")
    elif missing_pct > 20:
        warnings.append(f"🟡 Moderate missing values ({missing_pct:.1f}%).")

    imbalance = profile.get("imbalance", "")
    imbalance_str = str(imbalance)

    if "High" in imbalance_str:
        warnings.append("⚠️ Severe class imbalance detected. Consider SMOTE or class weights.")
    elif "Medium" in imbalance_str:
        warnings.append("🟡 Moderate class imbalance detected.")

    if len(cols) > rows:
        warnings.append(f"⚠️ More features ({len(cols)}) than rows ({rows}). High overfitting risk.")

    # ── 4. Column candidate scores ─────────────────────────────────────────────
    try:
        column_scores = _rank_target_candidates(df)
    except Exception:
        column_scores = []

    return {
        "suggested_target": suggested,
        "task_type": task_type,
        "confidence": confidence,
        "task_reason": task_reason,
        "warnings": warnings,
        "column_scores": column_scores,
        "row_count": rows,
        "col_count": len(cols),
    }


def _suggest_target(df: pd.DataFrame) -> str:
    target_hints = [
        "target", "label", "class", "output", "y", "result",
        "outcome", "price", "salary", "churn", "survived",
        "diagnosis", "fraud", "default", "status", "grade", "score"
    ]

    try:
        for col in reversed(df.columns.tolist()):
            col_lower = str(col).lower().strip()
            if any(hint in col_lower for hint in target_hints):
                return col
    except Exception:
        pass

    try:
        return df.columns[-1]
    except Exception:
        return ""


def _detect_task(series: pd.Series):
    try:
        if not pd.api.types.is_numeric_dtype(series):
            return "classification", 95, "Non-numeric column → classification"

        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(len(series), 1)

        if unique_count <= 2:
            return "classification", 99, f"Binary column ({unique_count} unique values)"
        if unique_count <= 10 and unique_ratio < 0.05:
            return "classification", 92, f"Low cardinality ({unique_count} classes)"
        if unique_count <= 20 and unique_ratio < 0.1:
            return "classification", 80, f"Likely multi-class ({unique_count} unique values)"
        if pd.api.types.is_float_dtype(series):
            return "regression", 90, "Continuous float column → regression"

        return "regression", 70, f"High cardinality int column ({unique_count} unique)"
    except Exception:
        return "classification", 50, "Fallback due to error"


def _rank_target_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    scores = []
    target_hints = [
        "target", "label", "class", "output", "y", "result",
        "outcome", "price", "salary", "churn", "survived", "diagnosis"
    ]

    for col in df.columns:
        try:
            s = df[col].dropna()
            unique = s.nunique()
            score = 0

            if any(h in str(col).lower() for h in target_hints):
                score += 40

            if col == df.columns[-1]:
                score += 20

            if unique <= 20:
                score += 15

            if unique > 1:
                score += 10

            missing = df[col].isnull().mean()

            if missing < 0.05:
                score += 15

            task, _, _ = _detect_task(df[col])

            scores.append({
                "column": col,
                "score": score,
                "unique": int(unique),
                "task_type": task,
                "missing_pct": round(float(missing * 100), 1),
            })

        except Exception:
            continue

    try:
        scores.sort(key=lambda x: x["score"], reverse=True)
    except Exception:
        pass

    return scores[:10]
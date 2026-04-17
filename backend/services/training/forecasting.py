from __future__ import annotations

from typing import Any, Dict, List


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def estimate_training_forecast(
    profile: Dict[str, Any] | None,
    target_column: str,
    goal: str,
    mode: str,
    selected_features: List[str] | None = None,
    cv_folds: int = 0,
    handle_imbalance: bool = False,
    auto_clean: bool = True,
    eval_metric: str = "",
) -> Dict[str, Any]:
    profile = profile or {}
    selected_features = list(selected_features or [])
    rows = _safe_int(profile.get("rows"), 0)
    cols = _safe_int(profile.get("cols"), len(profile.get("columns") or []))
    all_columns = list(profile.get("columns") or [])

    if cols <= 0 and all_columns:
        cols = len(all_columns)

    usable_feature_count = len(selected_features) if selected_features else max(cols - 1, 0)
    task_type = str(profile.get("task_type") or "classification")

    mode_profile = {
        "Fast": {"sweep_size": 0.2 if rows < 5000 else 0.08, "top_k": 1, "trials": 0, "optuna": False},
        "Balanced": {"sweep_size": 0.35 if rows < 5000 else 0.12, "top_k": 2, "trials": 12, "optuna": True},
        "Full": {"sweep_size": 0.5 if rows < 5000 else 0.2, "top_k": 3, "trials": 32, "optuna": True},
    }.get(mode, {"sweep_size": 0.12, "top_k": 2, "trials": 12, "optuna": True})

    goal_pool_size = {
        "Speed": 3,
        "Balanced": 4 if task_type == "classification" else 5,
        "Performance": 6,
    }.get(goal, 4)

    size_score = (
        (rows / 2500.0)
        + (usable_feature_count / 20.0)
        + (max(_safe_float(profile.get("missing_pct"), 0.0), 0.0) / 20.0)
    )
    multiplier = 1.0
    if auto_clean:
        multiplier += 0.15
    if handle_imbalance and task_type == "classification":
        multiplier += 0.18
    if cv_folds >= 2:
        multiplier += min(cv_folds, 8) * 0.22
    if mode_profile["optuna"]:
        multiplier += mode_profile["trials"] * 0.035
    if mode == "Full":
        multiplier += 0.4

    estimated_seconds = max(8.0, size_score * 7.0 * multiplier)
    min_seconds = int(round(max(5.0, estimated_seconds * 0.65)))
    max_seconds = int(round(max(min_seconds + 5.0, estimated_seconds * 1.7)))

    compute_intensity = (
        "Low"
        if max_seconds < 45
        else "Medium"
        if max_seconds < 180
        else "High"
        if max_seconds < 480
        else "Very High"
    )
    memory_risk = (
        "Low"
        if usable_feature_count < 40 and rows < 15000
        else "Medium"
        if usable_feature_count < 120 and rows < 100000
        else "High"
    )
    model_budget = min(goal_pool_size, max(mode_profile["top_k"], goal_pool_size - 1))
    sweep_rows = min(max(int(rows * mode_profile["sweep_size"]), 64), 5000 if mode == "Fast" else 8000)

    recommendations = []
    if max_seconds > 300:
        recommendations.append("Consider Fast or Balanced mode first to validate the dataset quickly.")
    if cv_folds >= 5:
        recommendations.append("High CV folds will noticeably increase runtime; use 3 for a faster first pass.")
    if selected_features:
        recommendations.append("Feature selection is reducing the search width, which should help runtime and stability.")
    if handle_imbalance and task_type == "classification":
        recommendations.append("Imbalance handling adds overhead but should improve minority-class behavior.")
    if not recommendations:
        recommendations.append("This configuration looks practical for an exploratory run.")

    return {
        "task_type": task_type,
        "target_column": target_column,
        "goal": goal,
        "mode": mode,
        "eval_metric": eval_metric,
        "estimated_duration_seconds": {"min": min_seconds, "max": max_seconds},
        "estimated_duration_label": f"{min_seconds}s to {max_seconds}s",
        "compute_intensity": compute_intensity,
        "memory_risk": memory_risk,
        "estimated_model_count": model_budget,
        "estimated_sweep_rows": sweep_rows,
        "estimated_feature_count": usable_feature_count,
        "cv_folds": int(cv_folds or 0),
        "optuna_trials": mode_profile["trials"],
        "uses_bayesian_optimization": mode_profile["optuna"],
        "notes": recommendations,
    }

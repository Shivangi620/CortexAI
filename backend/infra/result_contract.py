"""Result contract enforcement for AutoML job outputs."""
import math
from numbers import Integral, Real
from typing import Any, Dict

REQUIRED_RESULT_CONTRACT = {
    "best_model": "",
    "score": 0.0,
    "metric_name": "",
    "leaderboard": [],
    "model_path": "",
    "shap_summary": {},
    "reasoning": [],
}


def sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-JSON-safe values like NaN/Inf."""
    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(v) for v in value]

    return str(value)


def _normalize_value(key: str, value: Any) -> Any:
    if key in {"best_model", "metric_name", "model_path"}:
        try:
            return str(value) if value is not None else ""
        except Exception:
            return ""

    if key == "score":
        try:
            score = float(value)
            return score if math.isfinite(score) else 0.0
        except Exception:
            return 0.0

    if key == "leaderboard":
        try:
            return sanitize_for_json(list(value) if isinstance(value, (list, tuple)) else [])
        except Exception:
            return []

    if key == "shap_summary":
        try:
            return sanitize_for_json(dict(value) if isinstance(value, dict) else {})
        except Exception:
            return {}

    if key == "reasoning":
        if isinstance(value, list):
            return sanitize_for_json(value)
        if isinstance(value, str):
            return [value]
        if isinstance(value, (tuple, set)):
            return sanitize_for_json(list(value))
        return []

    return sanitize_for_json(value)


def normalize_results(results: Any) -> Dict[str, Any]:
    """Normalize a results payload to the shared result contract."""
    if not isinstance(results, dict):
        results = {}

    normalized = sanitize_for_json(dict(results))

    for key, default in REQUIRED_RESULT_CONTRACT.items():
        try:
            if key not in normalized or normalized[key] is None:
                normalized[key] = default
            else:
                normalized[key] = _normalize_value(key, normalized[key])
        except Exception:
            normalized[key] = default

    return normalized

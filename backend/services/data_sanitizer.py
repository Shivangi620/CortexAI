from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


NULL_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "invalid",
    "?",
    "??",
    "-",
}


@dataclass
class SanitizeResult:
    df: pd.DataFrame
    logs: List[str]
    report: Dict[str, Any]


def _normalize_text_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() in NULL_LIKE_VALUES:
        return np.nan
    return stripped


def _maybe_numeric(series: pd.Series) -> tuple[pd.Series, bool]:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series, False
    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return series, False
    coerced = pd.to_numeric(series, errors="coerce")
    success_ratio = float(coerced.notna().mean())
    if success_ratio >= 0.85:
        return coerced, True
    return series, False


def _maybe_datetime(series: pd.Series) -> tuple[pd.Series, bool]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, False
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series, False
    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return series, False
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    success_ratio = float(parsed.notna().mean())
    if success_ratio >= 0.8:
        return parsed, True
    return series, False


def _normalize_categories(series: pd.Series) -> tuple[pd.Series, bool]:
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    ):
        return series, False
    normalized = series.map(_normalize_text_cell)
    non_null = normalized.dropna()
    if non_null.empty:
        return normalized, True
    lowered = non_null.astype(str).str.lower()
    if lowered.nunique() <= max(50, int(len(lowered) * 0.5)):
        remapped = normalized.map(
            lambda x: x.lower() if isinstance(x, str) else x
        )
        return remapped.astype("category"), True
    return normalized, True


def sanitize_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
    dataset_name: str | None = None,
) -> SanitizeResult:
    clean_df = df.copy()
    logs: List[str] = []
    report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "rows_before": int(len(clean_df)),
        "columns_before": int(len(clean_df.columns)),
        "duplicate_rows_removed": 0,
        "numeric_coercions": [],
        "datetime_columns": [],
        "categorical_columns": [],
        "empty_string_cells_cleaned": 0,
        "target_valid": True,
        "target_issue": None,
        "dropped_target_rows": 0,
    }

    original_columns = list(clean_df.columns)
    clean_df.columns = [str(col).strip().replace("\n", " ") for col in clean_df.columns]
    if clean_df.columns.tolist() != original_columns:
        logs.append("Sanitizer: Normalized column names.")

    for col in clean_df.columns:
        if clean_df[col].dtype == object or pd.api.types.is_string_dtype(clean_df[col]):
            raw = clean_df[col]
            empty_like_count = int(
                raw.astype(str).str.strip().str.lower().isin(NULL_LIKE_VALUES).sum()
            )
            report["empty_string_cells_cleaned"] += empty_like_count
            clean_df[col] = raw.map(_normalize_text_cell)

            coerced, changed = _maybe_numeric(clean_df[col])
            if changed:
                clean_df[col] = coerced
                report["numeric_coercions"].append(col)
                logs.append(f"Sanitizer: Coerced '{col}' to numeric.")
                continue

            parsed, changed = _maybe_datetime(clean_df[col])
            if changed:
                clean_df[col] = parsed
                report["datetime_columns"].append(col)
                logs.append(f"Sanitizer: Detected datetime column '{col}'.")
                continue

            normalized, changed = _normalize_categories(clean_df[col])
            clean_df[col] = normalized
            if changed:
                report["categorical_columns"].append(col)

    before_dupes = len(clean_df)
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    dupes_removed = before_dupes - len(clean_df)
    if dupes_removed:
        report["duplicate_rows_removed"] = int(dupes_removed)
        logs.append(f"Sanitizer: Removed {dupes_removed} duplicate rows.")

    if target:
        if target not in clean_df.columns:
            report["target_valid"] = False
            report["target_issue"] = f"Target column '{target}' not found."
        else:
            target_series = clean_df[target]
            invalid_target = target_series.isna()
            if (
                target_series.dtype == object
                or pd.api.types.is_string_dtype(target_series)
                or pd.api.types.is_categorical_dtype(target_series)
            ):
                invalid_target = invalid_target | target_series.astype(str).str.strip().str.lower().isin(NULL_LIKE_VALUES)
            removed = int(invalid_target.sum())
            if removed:
                clean_df = clean_df.loc[~invalid_target].reset_index(drop=True)
                report["dropped_target_rows"] = removed
                logs.append(f"Sanitizer: Removed {removed} rows with invalid target values.")
            remaining = clean_df[target].dropna()
            if remaining.empty or remaining.nunique(dropna=True) <= 1:
                report["target_valid"] = False
                report["target_issue"] = "Target has insufficient non-null variation after sanitization."

    report["rows_after"] = int(len(clean_df))
    report["columns_after"] = int(len(clean_df.columns))
    return SanitizeResult(df=clean_df, logs=logs, report=report)


def build_dataset_version_report(current_df: pd.DataFrame, previous_df: pd.DataFrame, target: str | None = None) -> Dict[str, Any]:
    current_columns = set(current_df.columns)
    previous_columns = set(previous_df.columns)
    added_columns = sorted(current_columns - previous_columns)
    removed_columns = sorted(previous_columns - current_columns)

    current_missing = current_df.isna().mean().replace([np.inf, -np.inf], np.nan).fillna(0)
    previous_missing = previous_df.isna().mean().replace([np.inf, -np.inf], np.nan).fillna(0)

    missing_delta = []
    for col in sorted(current_columns.intersection(previous_columns)):
        delta = float(current_missing.get(col, 0) - previous_missing.get(col, 0))
        if abs(delta) >= 0.05:
            missing_delta.append(
                {"column": col, "missing_pct_delta": round(delta * 100, 2)}
            )

    target_shift = {}
    if target and target in current_df.columns and target in previous_df.columns:
        current_target = current_df[target].astype(str).value_counts(normalize=True, dropna=True)
        previous_target = previous_df[target].astype(str).value_counts(normalize=True, dropna=True)
        labels = sorted(set(current_target.index).union(set(previous_target.index)))
        target_shift = {
            "distribution_delta": [
                {
                    "label": label,
                    "current_pct": round(float(current_target.get(label, 0)) * 100, 2),
                    "previous_pct": round(float(previous_target.get(label, 0)) * 100, 2),
                    "delta_pct": round(float(current_target.get(label, 0) - previous_target.get(label, 0)) * 100, 2),
                }
                for label in labels[:25]
            ]
        }

    return {
        "row_delta": int(len(current_df) - len(previous_df)),
        "current_rows": int(len(current_df)),
        "previous_rows": int(len(previous_df)),
        "current_columns": int(len(current_df.columns)),
        "previous_columns": int(len(previous_df.columns)),
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "missingness_changes": missing_delta[:30],
        "target_shift": target_shift,
    }


def summarize_experiment(metrics: Dict[str, Any]) -> str:
    best_model = metrics.get("best_model") or "The winning model"
    metric_name = metrics.get("metric_name") or "score"
    score = metrics.get("score")
    warnings = metrics.get("warnings") or []
    shap_summary = metrics.get("shap_summary") or {}
    top_feature = None
    if isinstance(shap_summary, dict) and shap_summary:
        try:
            top_feature = max(shap_summary, key=lambda key: abs(float(shap_summary[key])))
        except Exception:
            top_feature = next(iter(shap_summary.keys()), None)

    score_text = f"{score}" if score is not None else "its best score"
    parts = [f"{best_model} led this run with {metric_name} {score_text}."]
    if top_feature:
        parts.append(f"It was most sensitive to '{top_feature}'.")
    if warnings:
        parts.append(f"Main watch-outs: {', '.join(str(w.get('type') or w) for w in warnings[:3])}.")
    else:
        parts.append("No major training warnings were triggered.")
    return " ".join(parts)


def sanitize_report_json(report: Dict[str, Any]) -> str:
    return json.dumps(report or {}, default=str)

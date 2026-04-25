"""api/routes/predict.py — Live prediction and future sweep endpoints."""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json
import math
import os
import uuid
import joblib
import pandas as pd
import shap
import numpy as np

from infra.database import db_session, JobModel, DatasetModel, ScenarioPackModel
from infra.result_contract import normalize_results, sanitize_for_json
from infra.storage import get_schema_path
from api.routes.datasets import _stream_upload_to_file
from core.file_loader import load_dataframe
from services.data_sanitizer import sanitize_dataframe

router = APIRouter(prefix="/api", tags=["predict"])


def _build_inference_frame(features: Dict[str, Any], expected_features: List[str]) -> pd.DataFrame:
    incoming = set(features.keys())
    expected = set(expected_features or [])

    if expected:
        missing = sorted(expected - incoming)

        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # We now ignore extra features instead of raising a 422 error,
        # which is more resilient to noisy payloads in production.
        row = {name: features.get(name) for name in expected_features}
        frame = pd.DataFrame([row], columns=expected_features)
    else:
        frame = pd.DataFrame([features])

    return sanitize_dataframe(frame).df


def _coerce_prediction_value(value: Any):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float, np.number)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _predict_with_metadata(model, df: pd.DataFrame, expected_features: List[str]) -> Dict[str, Any]:
    pred = model.predict(df)
    raw = pred[0]

    result = {
        "prediction": _coerce_prediction_value(raw),
        "feature_names": expected_features,
    }

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(df)
            result["confidence_pct"] = round(float(max(proba[0])) * 100, 1)
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) == len(proba[0]):
                result["probabilities"] = {
                    str(label): round(float(score) * 100, 2)
                    for label, score in zip(classes, proba[0])
                }
        except Exception:
            pass

    try:
        if hasattr(model, "preprocess") and hasattr(model, "get_underlying_model"):
            processed = model.preprocess(df)
            explainer = shap.Explainer(model.get_underlying_model(), processed)
            shap_values = explainer(processed)
            feature_names = (
                list(model.get_feature_names_out())
                if hasattr(model, "get_feature_names_out")
                else expected_features
            )
            result["sensitivity"] = {
                name: round(float(abs(val)), 4)
                for name, val in zip(feature_names, shap_values.values[0])
            }
        elif hasattr(model, "predict"):
            explainer = shap.Explainer(model.predict, df)
            shap_values = explainer(df)
            result["sensitivity"] = {
                name: round(float(abs(val)), 4)
                for name, val in zip(expected_features, shap_values.values[0])
            }
    except Exception:
        pass

    return result


def _parse_filter_value(value: Any):
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def _apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    filtered = df.copy()
    for rule in filters or []:
        column = rule.get("column")
        operator = str(rule.get("operator") or "eq").lower()
        value = _parse_filter_value(rule.get("value"))
        if not column or column not in filtered.columns:
            continue

        series = filtered[column]
        if operator == "eq":
            filtered = filtered.loc[series == value]
        elif operator == "neq":
            filtered = filtered.loc[series != value]
        elif operator == "gt":
            filtered = filtered.loc[pd.to_numeric(series, errors="coerce") > pd.to_numeric(value, errors="coerce")]
        elif operator == "gte":
            filtered = filtered.loc[pd.to_numeric(series, errors="coerce") >= pd.to_numeric(value, errors="coerce")]
        elif operator == "lt":
            filtered = filtered.loc[pd.to_numeric(series, errors="coerce") < pd.to_numeric(value, errors="coerce")]
        elif operator == "lte":
            filtered = filtered.loc[pd.to_numeric(series, errors="coerce") <= pd.to_numeric(value, errors="coerce")]
        elif operator == "contains":
            filtered = filtered.loc[series.astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == "in":
            values = value if isinstance(value, list) else [item.strip() for item in str(value).split(",") if item.strip()]
            filtered = filtered.loc[series.astype(str).isin([str(item) for item in values])]
    return filtered


def _dataset_feature_ranges(df: pd.DataFrame, expected_features: List[str]) -> List[Dict[str, Any]]:
    suggestions = []
    for feature in expected_features:
        if feature not in df.columns:
            continue
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if numeric.empty:
                continue
            suggestions.append({
                "feature": feature,
                "kind": "numeric",
                "min": round(float(numeric.min()), 4),
                "max": round(float(numeric.max()), 4),
                "median": round(float(numeric.median()), 4),
                "mean": round(float(numeric.mean()), 4),
                "step": round(float(max((numeric.max() - numeric.min()) / 40, 0.01)), 4),
            })
        else:
            top_values = series.dropna().astype(str).value_counts().head(8).index.tolist()
            if top_values:
                suggestions.append({
                    "feature": feature,
                    "kind": "categorical",
                    "top_values": top_values,
                })
    return suggestions


def _coerce_jsonable(value: Any):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float, np.number)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_jsonable(item) for item in value]
    return value


def _feature_range_lookup(feature_ranges: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item.get("feature"): item for item in feature_ranges if item.get("feature")}


def _prediction_score(meta: Dict[str, Any]) -> Optional[float]:
    prediction = meta.get("prediction")
    if isinstance(prediction, (int, float)):
        numeric = float(prediction)
        return numeric if math.isfinite(numeric) else None
    confidence = meta.get("confidence_pct")
    if isinstance(confidence, (int, float)):
        numeric = float(confidence)
        return numeric if math.isfinite(numeric) else None
    return None


def _counterfactual_objective(meta: Dict[str, Any], is_classification: bool) -> Optional[float]:
    if is_classification:
        confidence = meta.get("confidence_pct")
        if isinstance(confidence, (int, float)):
            numeric = float(confidence)
            return numeric if math.isfinite(numeric) else None
        probabilities = meta.get("probabilities") or {}
        if isinstance(probabilities, dict) and probabilities:
            numeric = [float(value) for value in probabilities.values() if isinstance(value, (int, float)) and math.isfinite(float(value))]
            if numeric:
                return max(numeric)
        return None
    return _prediction_score(meta)


def _assess_scenario_guardrails(
    base_payload: Dict[str, Any],
    candidate_payload: Dict[str, Any],
    adjustments: Dict[str, Any],
    feature_ranges: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    feature_lookup = _feature_range_lookup(feature_ranges)
    blocked_features = {str(item).strip() for item in policy.get("blocked_features", []) if str(item).strip()}
    review_keywords = [str(item).strip().lower() for item in policy.get("review_keywords", []) if str(item).strip()]
    max_delta_ratio = max(float(policy.get("max_numeric_delta_ratio", 0.35) or 0.35), 0.0)
    hard_bounds = bool(policy.get("hard_bounds", True))

    touched_features = set()
    reasons = []
    review_required = False
    blocked = False
    risk_score = 0.0

    for bucket in ("set", "delta", "scale"):
        for feature in (adjustments.get(bucket) or {}).keys():
            touched_features.add(feature)

    for feature in touched_features:
        if feature in blocked_features:
            blocked = True
            reasons.append(f"{feature} is blocked by policy.")
            risk_score += 1.0

        lowered = feature.lower()
        if any(keyword and keyword in lowered for keyword in review_keywords):
            review_required = True
            reasons.append(f"{feature} matches a protected keyword and needs review.")
            risk_score += 0.45

        range_info = feature_lookup.get(feature) or {}
        base_value = base_payload.get(feature)
        candidate_value = candidate_payload.get(feature)
        if range_info.get("kind") != "numeric":
            continue

        try:
            base_num = float(base_value)
            candidate_num = float(candidate_value)
        except Exception:
            continue

        min_value = range_info.get("min")
        max_value = range_info.get("max")
        span = None
        if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
            span = max(float(max_value) - float(min_value), 1e-9)
            if hard_bounds and (candidate_num < float(min_value) or candidate_num > float(max_value)):
                blocked = True
                reasons.append(f"{feature} moved outside the trained range [{min_value}, {max_value}].")
                risk_score += 1.0

        denominator = max(abs(base_num), span or 0.0, 1.0)
        change_ratio = abs(candidate_num - base_num) / denominator
        if change_ratio > max_delta_ratio:
            review_required = True
            reasons.append(f"{feature} changed by {round(change_ratio * 100, 1)}% of its reference range/value.")
            risk_score += min(change_ratio, 1.5)

    status = "approved"
    if blocked:
        status = "blocked"
    elif review_required:
        status = "review_required"

    unique_reasons = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    return {
        "status": status,
        "approval_required": status == "review_required",
        "blocked": blocked,
        "risk_score": round(risk_score, 3),
        "touched_features": sorted(touched_features),
        "reasons": unique_reasons,
    }


def _generate_auto_suggestions(
    model,
    base_frame: pd.DataFrame,
    expected_features: List[str],
    feature_ranges: List[Dict[str, Any]],
    baseline_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    base_payload = _coerce_jsonable(base_frame.iloc[0].to_dict())
    feature_lookup = _feature_range_lookup(feature_ranges)
    sensitivities = baseline_meta.get("sensitivity") or {}
    candidate_rows = []
    baseline_score = _prediction_score(baseline_meta)

    for feature in expected_features:
        range_info = feature_lookup.get(feature) or {}
        if range_info.get("kind") != "numeric":
            continue
        try:
            base_value = float(base_payload.get(feature))
        except Exception:
            continue

        minimum = range_info.get("min")
        maximum = range_info.get("max")
        if minimum is None or maximum is None:
            continue

        span = max(float(maximum) - float(minimum), 1e-9)
        step = float(range_info.get("step") or max(span * 0.1, 0.01))
        directions = [("increase", min(base_value + step, float(maximum))), ("decrease", max(base_value - step, float(minimum)))]

        for direction, candidate_value in directions:
            if abs(candidate_value - base_value) < 1e-9:
                continue

            payload = dict(base_payload)
            payload[feature] = round(float(candidate_value), 6)
            frame = _build_inference_frame(payload, expected_features)
            meta = _predict_with_metadata(model, frame, expected_features)
            score = _prediction_score(meta)
            delta = None if baseline_score is None or score is None else round(float(score) - float(baseline_score), 4)
            effort = round(abs(candidate_value - base_value) / max(span, abs(base_value), 1.0), 4)
            candidate_rows.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "baseline_value": round(base_value, 6),
                    "candidate_value": round(float(candidate_value), 6),
                    "delta": delta,
                    "effort_score": effort,
                    "sensitivity": round(float(sensitivities.get(feature, 0) or 0), 4),
                    "prediction": meta.get("prediction"),
                    "confidence_pct": meta.get("confidence_pct"),
                    "scenario": {
                        "id": f"auto-{feature}-{direction}",
                        "name": f"{feature} {direction.title()}",
                        "description": f"{direction.title()} {feature} by one guided step.",
                        "adjustments": {"set": {feature: round(float(candidate_value), 6)}},
                    },
                }
            )

    positive_rows = [row for row in candidate_rows if isinstance(row.get("delta"), (int, float)) and row["delta"] > 0]
    ranked_uplift = sorted(positive_rows or candidate_rows, key=lambda row: ((row.get("delta") or -10**9), row.get("sensitivity", 0)), reverse=True)
    ranked_risk = sorted(
        positive_rows or candidate_rows,
        key=lambda row: (row.get("effort_score", 10**9), -(row.get("delta") or -10**9), -row.get("sensitivity", 0)),
    )

    suggestions = []
    if ranked_uplift:
        best = ranked_uplift[0]
        suggestions.append(
            {
                "type": "best_uplift",
                "title": "Best uplift",
                "summary": f"Push {best['feature']} {best['direction']} for the strongest modeled gain.",
                **best,
            }
        )

    if ranked_risk:
        safest = ranked_risk[0]
        if not suggestions or safest["scenario"]["id"] != suggestions[0]["scenario"]["id"]:
            suggestions.append(
                {
                    "type": "least_risk",
                    "title": "Least risk",
                    "summary": f"Adjust {safest['feature']} with the smallest normalized move that still improves the outcome.",
                    **safest,
                }
            )

    return suggestions[:2]


def _load_job_and_model(job_id: str):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed or not found")
        job_snapshot = {
            "id": job.id,
            "dataset_id": job.dataset_id,
            "status": job.status,
        }

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}

        results = normalize_results(raw_results)

    from infra.storage import resolve_model_path

    model_path = resolve_model_path(job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    return job_snapshot, results, model


@router.get("/scenario/context/{job_id}")
def get_scenario_context(job_id: str):
    job, results, _ = _load_job_and_model(job_id)
    expected_features = results.get("feature_names") or []
    target_name = results.get("target")
    dataset_id = job["dataset_id"]

    with db_session() as db:
        dataset_row = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        dataset_path = dataset_row.file_path if dataset_row else None

    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Training dataset for scenario analysis is not available")

    try:
        source_df = load_dataframe(filepath=dataset_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not reload dataset for scenario analysis: {e}")

    if target_name and target_name in source_df.columns:
        source_df = source_df.drop(columns=[target_name])

    feature_ranges = _dataset_feature_ranges(source_df, expected_features)
    default_payload = {}
    for item in feature_ranges:
        if item.get("kind") == "numeric":
            default_payload[item["feature"]] = item.get("median", 0)
        else:
            values = item.get("top_values") or []
            default_payload[item["feature"]] = values[0] if values else ""

    sample_rows = []
    preview_frame = source_df[expected_features].drop_duplicates().head(6) if expected_features else source_df.drop_duplicates().head(6)
    for index, (_, row) in enumerate(preview_frame.iterrows()):
        values = {}
        for key, value in row.to_dict().items():
            if hasattr(value, "item"):
                value = value.item()
            values[key] = value
        sample_rows.append(
            {
                "id": f"row-{index}",
                "label": f"Reference Row {index + 1}",
                "preview": ", ".join(f"{key}={values.get(key)}" for key in list(values.keys())[:3]),
                "values": values,
            }
        )

    sample_rows.insert(
        0,
        {
            "id": "median-profile",
            "label": "Median / Mode Profile",
            "preview": "Centered baseline generated from the training dataset",
            "values": default_payload,
        },
    )

    return sanitize_for_json({
        "job_id": job_id,
        "target": target_name,
        "model": results.get("best_model"),
        "is_classification": bool(results.get("is_classification")),
        "feature_names": expected_features,
        "feature_ranges": feature_ranges,
        "default_payload": default_payload,
        "sample_rows": sample_rows,
    })


class PredictRequest(BaseModel):
    features: Dict[str, Any]


@router.post("/predict/{job_id}")
def predict(job_id: str, req: PredictRequest):
    _, results, model = _load_job_and_model(job_id)

    try:
        expected_features = results.get("feature_names") or []
        df = _build_inference_frame(req.features, expected_features)
        return _predict_with_metadata(model, df, expected_features)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CounterfactualRequest(BaseModel):
    payload: Dict[str, Any]
    target_prediction: float


def _run_counterfactual(job_id: str, req: CounterfactualRequest):
    """
    Suggest minimal changes to reach a target prediction.
    Uses a simple greedy search for demonstration.
    """
    _, results, model = _load_job_and_model(job_id)

    try:
        expected_features = results.get("feature_names") or []
        is_classification = bool(results.get("is_classification"))
        base_df = _build_inference_frame(req.payload, expected_features)
        baseline_meta = _predict_with_metadata(model, base_df, expected_features)
        current_score = _counterfactual_objective(baseline_meta, is_classification)
        if current_score is None:
            return {"error": "Goal Seeker needs a numeric prediction or confidence score for this model."}
        
        # Simple heuristic: try ±10% on each numeric feature
        suggestions = []
        for col in base_df.columns:
            val = base_df[col].iloc[0]
            numeric_val = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            if pd.isna(numeric_val):
                continue

            base_value = float(numeric_val)
            candidates = []
            if abs(base_value) > 1e-9:
                candidates = [
                    ("Decrease by 10%", base_value * 0.9),
                    ("Increase by 10%", base_value * 1.1),
                ]
            else:
                candidates = [
                    ("Decrease by 1", -1.0),
                    ("Increase by 1", 1.0),
                ]

            for change_label, candidate_value in candidates:
                test_df = base_df.copy()
                test_df[col] = candidate_value
                meta = _predict_with_metadata(model, test_df, expected_features)
                new_score = _counterfactual_objective(meta, is_classification)
                if new_score is None:
                    continue

                if abs(new_score - req.target_prediction) < abs(current_score - req.target_prediction):
                    suggestions.append({
                        "feature": col,
                        "from": round(base_value, 4),
                        "to": round(float(candidate_value), 4),
                        "change": change_label,
                        "new_prediction": meta.get("prediction"),
                        "new_score": round(float(new_score), 4),
                        "impact": round(abs(float(new_score) - float(current_score)), 4),
                    })
        
        return {
            "current_prediction": baseline_meta.get("prediction"),
            "confidence_pct": baseline_meta.get("confidence_pct"),
            "current_score": round(float(current_score), 4),
            "target_prediction": req.target_prediction,
            "suggestions": sorted(suggestions, key=lambda x: (abs(x["new_score"] - req.target_prediction), -x["impact"]))[:5],
            "message": "Goal Seeker optimized confidence." if is_classification else "Goal Seeker optimized numeric prediction.",
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/counterfactual-lite/{job_id}")
def get_counterfactual_lite(job_id: str, req: CounterfactualRequest):
    return _run_counterfactual(job_id, req)


@router.post("/counterfactual/{job_id}")
def get_counterfactual(job_id: str, req: CounterfactualRequest):
    return _run_counterfactual(job_id, req)


class ScenarioFilter(BaseModel):
    column: str
    operator: str = "eq"
    value: Any = None


class ScenarioAdjustments(BaseModel):
    set: Dict[str, Any] = Field(default_factory=dict)
    delta: Dict[str, float] = Field(default_factory=dict)
    scale: Dict[str, float] = Field(default_factory=dict)


class ScenarioSpec(BaseModel):
    id: Optional[str] = None
    name: str
    description: str = ""
    adjustments: ScenarioAdjustments = Field(default_factory=ScenarioAdjustments)


class ScenarioApprovalPolicy(BaseModel):
    max_numeric_delta_ratio: float = 0.35
    hard_bounds: bool = True
    blocked_features: List[str] = Field(default_factory=list)
    review_keywords: List[str] = Field(
        default_factory=lambda: ["price", "discount", "limit", "exposure", "debt", "salary", "budget"]
    )


class ScenarioRequest(BaseModel):
    base_mode: str = "payload"
    row_index: Optional[int] = None
    base_payload: Dict[str, Any] = Field(default_factory=dict)
    filters: List[ScenarioFilter] = Field(default_factory=list)
    scenarios: List[ScenarioSpec] = Field(default_factory=list)
    sweep_feature: str = ""
    sweep_values: List[Any] = Field(default_factory=list)
    approval_policy: ScenarioApprovalPolicy = Field(default_factory=ScenarioApprovalPolicy)
    approved_scenarios: List[str] = Field(default_factory=list)
    enforce_guardrails: bool = True


@router.post("/scenarios/{job_id}")
def simulate_scenarios(job_id: str, req: ScenarioRequest):
    job, results, model = _load_job_and_model(job_id)
    expected_features = results.get("feature_names") or []
    target_name = results.get("target")

    with db_session() as db:
        dataset_row = db.query(DatasetModel).filter(DatasetModel.id == job["dataset_id"]).first()
        dataset_path = dataset_row.file_path if dataset_row else None

    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Training dataset for scenario analysis is not available")

    try:
        source_df = load_dataframe(filepath=dataset_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not reload dataset for scenario analysis: {e}")

    if target_name and target_name in source_df.columns:
        source_df = source_df.drop(columns=[target_name])

    if req.base_mode == "row":
        if req.row_index is None:
            raise HTTPException(status_code=422, detail="row_index is required for row mode")
        if req.row_index < 0 or req.row_index >= len(source_df):
            raise HTTPException(status_code=422, detail="row_index is outside the dataset bounds")
        base_record = source_df.iloc[int(req.row_index)].to_dict()
        cohort_df = source_df.iloc[[int(req.row_index)]].copy()
        cohort_meta = {"mode": "row", "row_index": int(req.row_index), "rows": 1}
    elif req.base_mode == "cohort":
        cohort_df = _apply_filters(source_df, [item.model_dump() for item in req.filters])
        if cohort_df.empty:
            raise HTTPException(status_code=422, detail="The selected cohort produced zero rows")
        numeric_cols = cohort_df.select_dtypes(include=[np.number]).columns.tolist()
        base_record = {}
        for feature in expected_features:
            if feature not in cohort_df.columns:
                continue
            series = cohort_df[feature]
            if feature in numeric_cols:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                base_record[feature] = float(numeric.median()) if not numeric.empty else 0.0
            else:
                mode = series.dropna().mode()
                base_record[feature] = mode.iloc[0] if not mode.empty else ""
        cohort_meta = {
            "mode": "cohort",
            "rows": int(len(cohort_df)),
            "filters": [item.model_dump() for item in req.filters],
        }
    else:
        base_record = dict(req.base_payload or {})
        cohort_df = source_df.copy()
        cohort_meta = {"mode": "payload", "rows": 1}

    try:
        base_frame = _build_inference_frame(base_record, expected_features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    baseline = _predict_with_metadata(model, base_frame, expected_features)
    feature_ranges = _dataset_feature_ranges(cohort_df if not cohort_df.empty else source_df, expected_features)
    approval_policy = req.approval_policy.model_dump() if hasattr(req.approval_policy, "model_dump") else dict(req.approval_policy or {})
    approved_scenarios = {str(item).strip() for item in req.approved_scenarios if str(item).strip()}
    scenario_rows = []
    guardrail_summary = {"approved": 0, "review_required": 0, "blocked": 0}

    for index, scenario in enumerate(req.scenarios or []):
        working = dict(base_frame.iloc[0].to_dict())
        adjustments = scenario.adjustments.model_dump() if hasattr(scenario.adjustments, "model_dump") else dict(scenario.adjustments or {})
        scenario_id = scenario.id or f"scenario-{index + 1}"

        for key, value in (adjustments.get("set") or {}).items():
            if key in working:
                working[key] = value
        for key, value in (adjustments.get("delta") or {}).items():
            if key in working:
                try:
                    working[key] = float(working[key]) + float(value)
                except Exception:
                    continue
        for key, value in (adjustments.get("scale") or {}).items():
            if key in working:
                try:
                    working[key] = float(working[key]) * float(value)
                except Exception:
                    continue

        guardrails = _assess_scenario_guardrails(
            _coerce_jsonable(base_frame.iloc[0].to_dict()),
            _coerce_jsonable(working),
            adjustments,
            feature_ranges,
            approval_policy,
        )
        guardrail_summary[guardrails["status"]] += 1
        approved = scenario_id in approved_scenarios or (scenario.name or "").strip() in approved_scenarios
        should_execute = not guardrails.get("blocked") and not (req.enforce_guardrails and guardrails.get("approval_required") and not approved)

        prediction = {}
        delta = None
        delta_pct = None
        if should_execute:
            frame = _build_inference_frame(working, expected_features)
            prediction = _predict_with_metadata(model, frame, expected_features)
            baseline_pred = baseline.get("prediction")
            current_pred = prediction.get("prediction")
            if isinstance(baseline_pred, (int, float)) and isinstance(current_pred, (int, float)):
                delta = round(float(current_pred) - float(baseline_pred), 4)
                if abs(float(baseline_pred)) > 1e-9:
                    delta_pct = round((delta / float(baseline_pred)) * 100, 2)

        scenario_rows.append({
            "id": scenario_id,
            "name": scenario.name or f"Scenario {index + 1}",
            "description": scenario.description,
            "prediction": prediction.get("prediction"),
            "confidence_pct": prediction.get("confidence_pct"),
            "delta": delta,
            "delta_pct": delta_pct,
            "payload": _coerce_jsonable(working),
            "probabilities": prediction.get("probabilities"),
            "executed": should_execute,
            "approved": approved,
            "approval_status": guardrails.get("status"),
            "guardrails": guardrails,
        })

    sweep_predictions = []
    if req.sweep_feature and req.sweep_values:
        for sweep_value in req.sweep_values:
            candidate = dict(base_frame.iloc[0].to_dict())
            if req.sweep_feature in candidate:
                candidate[req.sweep_feature] = _parse_filter_value(sweep_value)
            frame = _build_inference_frame(candidate, expected_features)
            pred_meta = _predict_with_metadata(model, frame, expected_features)
            sweep_predictions.append({
                "x": _parse_filter_value(sweep_value),
                "prediction": pred_meta.get("prediction"),
                "confidence": pred_meta.get("confidence_pct"),
            })

    recommendations = [
        "Cohort median mode creates a representative benchmark instead of relying on one lucky row.",
        "Scenario deltas are compared against the live baseline so you can watch uplift or downside instantly.",
        "Use sweep mode to stress-test one lever before locking a scenario into deployment rules.",
    ]

    return sanitize_for_json({
        "job_id": job_id,
        "target": target_name,
        "metric_name": results.get("metric_name"),
        "is_classification": results.get("is_classification"),
        "baseline": {
            **baseline,
            "payload": base_frame.iloc[0].to_dict(),
        },
        "cohort": cohort_meta,
        "feature_ranges": feature_ranges,
        "scenarios": scenario_rows,
        "sweep_feature": req.sweep_feature,
        "sweep_predictions": sweep_predictions,
        "recommended_features": recommendations,
        "approval_policy": approval_policy,
        "guardrail_summary": guardrail_summary,
        "auto_suggestions": _generate_auto_suggestions(model, base_frame, expected_features, feature_ranges, baseline),
    })


class ScenarioPackPayload(BaseModel):
    name: str
    description: str = ""
    base_mode: str = "payload"
    row_index: Optional[int] = None
    base_payload: Dict[str, Any] = Field(default_factory=dict)
    filters: List[ScenarioFilter] = Field(default_factory=list)
    scenarios: List[ScenarioSpec] = Field(default_factory=list)
    sweep_feature: str = ""
    sweep_values: List[Any] = Field(default_factory=list)
    approval_policy: ScenarioApprovalPolicy = Field(default_factory=ScenarioApprovalPolicy)
    approved_scenarios: List[str] = Field(default_factory=list)


@router.get("/scenario-packs/{job_id}")
def list_scenario_packs(job_id: str):
    with db_session() as db:
        rows = (
            db.query(ScenarioPackModel)
            .filter(ScenarioPackModel.job_id == job_id)
            .order_by(ScenarioPackModel.updated_at.desc(), ScenarioPackModel.created_at.desc())
            .all()
        )
        packs = []
        for row in rows:
            try:
                raw_scenarios = json.loads(row.scenarios_json or "[]")
            except Exception:
                raw_scenarios = []
            approval_policy = {}
            approved_scenarios = []
            scenarios = raw_scenarios
            if isinstance(raw_scenarios, dict):
                approval_policy = raw_scenarios.get("approval_policy") or {}
                approved_scenarios = raw_scenarios.get("approved_scenarios") or []
                scenarios = raw_scenarios.get("scenarios") or []
            packs.append(
                {
                    "id": row.id,
                    "job_id": row.job_id,
                    "name": row.name,
                    "description": row.description,
                    "base_mode": row.base_mode or "payload",
                    "row_index": int(row.row_index) if str(row.row_index or "").strip() else None,
                    "base_payload": json.loads(row.base_payload_json or "{}"),
                    "filters": json.loads(row.filters_json or "[]"),
                    "scenarios": scenarios,
                    "sweep_feature": row.sweep_feature or "",
                    "sweep_values": json.loads(row.sweep_values_json or "[]"),
                    "approval_policy": approval_policy,
                    "approved_scenarios": approved_scenarios,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
            )
    return {"job_id": job_id, "packs": packs}


@router.post("/scenario-packs/{job_id}")
def save_scenario_pack(job_id: str, req: ScenarioPackPayload):
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="Scenario pack name is required")

    with db_session() as db:
        row = (
            db.query(ScenarioPackModel)
            .filter(ScenarioPackModel.job_id == job_id, ScenarioPackModel.name == name)
            .first()
        )
        if row is None:
            row = ScenarioPackModel(id=uuid.uuid4().hex, job_id=job_id, name=name)
            db.add(row)

        row.description = req.description
        row.base_mode = req.base_mode
        row.row_index = str(req.row_index) if req.row_index is not None else None
        row.base_payload_json = json.dumps(_coerce_jsonable(req.base_payload or {}))
        row.filters_json = json.dumps([item.model_dump() for item in req.filters])
        row.scenarios_json = json.dumps(
            {
                "scenarios": [item.model_dump() for item in req.scenarios],
                "approval_policy": req.approval_policy.model_dump()
                if hasattr(req.approval_policy, "model_dump")
                else dict(req.approval_policy or {}),
                "approved_scenarios": [str(item).strip() for item in req.approved_scenarios if str(item).strip()],
            }
        )
        row.sweep_feature = req.sweep_feature
        row.sweep_values_json = json.dumps([_coerce_jsonable(item) for item in req.sweep_values])

    return {"saved": True, "job_id": job_id, "name": name}


class FutureRequest(BaseModel):
    job_id: str
    base_features: dict
    sweep_feature: str
    sweep_values: list


@router.post("/future")
def future_predict(req: FutureRequest):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == req.job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed")

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}

        results = normalize_results(raw_results)

    from infra.storage import resolve_model_path

    model_path = resolve_model_path(req.job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    expected_features = results.get("feature_names") or []

    predictions = []

    for val in req.sweep_values:
        features = dict(req.base_features or {})
        features[req.sweep_feature] = val

        try:
            df_row = _build_inference_frame(features, expected_features)

            pred = model.predict(df_row)[0]

            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(df_row)
                    conf = round(float(max(proba[0])) * 100, 1)
                except Exception:
                    conf = None

            predictions.append({
                "x": val,
                "prediction": float(pred) if hasattr(pred, "item") else pred,
                "confidence": conf
            })

        except Exception as e:
            predictions.append({
                "x": val,
                "error": str(e)
            })

    return {
        "sweep_feature": req.sweep_feature,
        "predictions": predictions
    }


@router.post("/contract-check/{job_id}")
async def contract_check(job_id: str, file: UploadFile = File(...)):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed or not found")

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}
        results = normalize_results(raw_results)

    expected_features = results.get("feature_names") or []
    schema_path = get_schema_path(job_id)
    contract_schema = {}
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r") as handle:
                contract_schema = json.load(handle).get("schema", {}) or {}
        except Exception:
            contract_schema = {}

    temp_path = None
    try:
        temp_path = await _stream_upload_to_file(
            f"contract_{job_id}_{os.urandom(4).hex()}",
            file.filename or "inference.csv",
            file,
        )
        df = load_dataframe(filepath=temp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read inference file: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    incoming_columns = list(df.columns)
    missing = sorted(set(expected_features) - set(incoming_columns))
    extra = sorted(set(incoming_columns) - set(expected_features))
    dtype_mismatches = []
    for col in incoming_columns:
        if col in contract_schema:
            expected_type = str(contract_schema[col].get("type", "unknown"))
            actual_type = str(df[col].dtype)
            if expected_type != "unknown" and actual_type != expected_type:
                dtype_mismatches.append(
                    {"column": col, "expected_type": expected_type, "actual_type": actual_type}
                )

    status = "aligned"
    if missing or dtype_mismatches:
        status = "drift_risk"
    elif extra:
        status = "warning"

    return {
        "job_id": job_id,
        "status": status,
        "rows": int(len(df)),
        "expected_feature_count": len(expected_features),
        "incoming_column_count": len(incoming_columns),
        "missing_features": missing,
        "extra_columns": extra,
        "dtype_mismatches": dtype_mismatches,
        "incoming_columns": incoming_columns,
    }


@router.post("/batch-predict/{job_id}")
async def batch_predict(job_id: str, file: UploadFile = File(...)):
    with db_session() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not completed or not found")

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}
        results = normalize_results(raw_results)

    from infra.storage import resolve_model_path
    model_path = resolve_model_path(job_id) or results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    temp_path = None
    try:
        temp_path = await _stream_upload_to_file(
            f"batch_{job_id}_{os.urandom(4).hex()}",
            file.filename or "batch.csv",
            file,
        )
        df = load_dataframe(filepath=temp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read batch file: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    try:
        expected_features = results.get("feature_names") or []
        sanitized = sanitize_dataframe(df, drop_duplicate_rows=False).df.reset_index(drop=True)
        if expected_features:
            missing = sorted(set(expected_features) - set(sanitized.columns))
            if missing:
                raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")
            model_input = sanitized[expected_features].copy()
        else:
            model_input = sanitized.copy()

        preds = model.predict(model_input)
        sanitized["prediction"] = preds

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(model_input)
            sanitized["confidence_pct"] = [round(float(max(p)) * 100, 1) for p in proba]

        preview = sanitize_for_json(sanitized.head(100).to_dict(orient="records"))
        return {
            "job_id": job_id,
            "row_count": len(sanitized),
            "preview": preview,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

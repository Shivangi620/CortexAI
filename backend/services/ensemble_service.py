"""
services/ensemble_service.py
Feature 7: Build ensemble models from multiple completed training jobs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from core.file_loader import load_dataframe
from infra.database import DatasetModel, JobModel, db_session
from infra.storage import ModelRegistry, get_model_path, resolve_model_path, save_metrics
from services.training.inference import PrefitStackingEnsemble, PrefitVotingEnsemble


def _safe_score(value: Any, default: float = 1.0) -> float:
    try:
        numeric = float(value)
        if np.isfinite(numeric):
            return max(numeric, 0.01)
    except Exception:
        pass
    return default


def _extract_feature_names(model: Any, results: Dict[str, Any]) -> List[str]:
    candidates = [
        results.get("feature_names"),
        getattr(model, "feature_names_in_", None),
        getattr(getattr(model, "model", None), "feature_names_in_", None),
    ]
    for value in candidates:
        if isinstance(value, (list, tuple)) and value:
            return [str(item) for item in value if str(item).strip()]
    return []


def _load_reference_dataset(dataset_id: Optional[str]) -> Optional[pd.DataFrame]:
    if not dataset_id:
        return None
    with db_session() as db:
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset or not dataset.file_path or not os.path.exists(dataset.file_path):
            return None
        try:
            return load_dataframe(filepath=dataset.file_path)
        except Exception:
            return None


def _resolve_feature_contract(feature_sets: List[List[str]], dataset_df: Optional[pd.DataFrame], target: Optional[str]) -> List[str]:
    normalized = [set(items) for items in feature_sets if items]
    if normalized:
        shared = set.intersection(*normalized) if len(normalized) > 1 else normalized[0]
        if shared:
            ordered = []
            for items in feature_sets:
                for item in items:
                    if item in shared and item not in ordered:
                        ordered.append(item)
            return ordered

    if dataset_df is not None:
        return [column for column in dataset_df.columns if column != target]
    return []


def _classification_labels(y: pd.Series, models: List[Any]) -> List[Any]:
    labels: List[Any] = []
    for item in pd.Series(y).dropna().tolist():
        if item not in labels:
            labels.append(item)
    for model in models:
        classes_attr = getattr(model, "classes_", None)
        for item in list(classes_attr) if classes_attr is not None else []:
            if item not in labels:
                labels.append(item)
    return labels


def _evaluate_classifier(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
    preds = pd.Series(model.predict(X)).astype(str)
    truth = pd.Series(y).astype(str)
    return round(float(accuracy_score(truth, preds)) * 100, 1)


def _evaluate_regressor(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
    preds = np.asarray(model.predict(X), dtype=float)
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    return round(float(r2_score(y, preds)) * 100, 1)


def _normalize_weights(weights: List[float]) -> List[float]:
    cleaned = []
    for value in weights:
        try:
            numeric = float(value)
            cleaned.append(max(numeric, 0.01) if np.isfinite(numeric) else 0.01)
        except Exception:
            cleaned.append(0.01)
    total = float(sum(cleaned)) or float(len(cleaned)) or 1.0
    return [round(value / total, 4) for value in cleaned]


def _fit_bagging_weights(
    models: List[Any],
    X: pd.DataFrame,
    y: pd.Series,
    is_classification: bool,
    rounds: int = 12,
    random_state: int = 42,
) -> List[float]:
    rng = np.random.default_rng(random_state)
    aggregate = np.zeros(len(models), dtype=float)
    row_count = len(X)
    if row_count == 0:
        return [1.0 for _ in models]

    for _ in range(max(rounds, 4)):
        sample_idx = rng.integers(0, row_count, size=row_count)
        sample_X = X.iloc[sample_idx]
        sample_y = y.iloc[sample_idx]
        for index, model in enumerate(models):
            try:
                score = (
                    _evaluate_classifier(model, sample_X, sample_y)
                    if is_classification
                    else _evaluate_regressor(model, sample_X, sample_y)
                )
                aggregate[index] += max(float(score), 0.01)
            except Exception:
                aggregate[index] += 0.01
    return aggregate.tolist()


def _fit_boosting_weights(
    models: List[Any],
    X: pd.DataFrame,
    y: pd.Series,
    is_classification: bool,
) -> List[float]:
    if len(X) == 0:
        return [1.0 for _ in models]

    sample_weights = np.full(len(X), 1.0 / len(X), dtype=float)
    model_weights: List[float] = []

    if is_classification:
        y_true = np.asarray(y)
        for model in models:
            preds = np.asarray(model.predict(X))
            misses = (preds != y_true).astype(float)
            weighted_error = float(np.sum(sample_weights * misses))
            weighted_error = min(max(weighted_error, 1e-6), 1 - 1e-6)
            alpha = 0.5 * np.log((1.0 - weighted_error) / weighted_error)
            if not np.isfinite(alpha) or alpha <= 0:
                alpha = 0.05
            model_weights.append(float(alpha))
            sample_weights *= np.exp(alpha * misses)
            sample_weights /= sample_weights.sum()
        return model_weights

    y_true = np.asarray(y, dtype=float)
    target_scale = max(float(np.nanstd(y_true)), 1.0)
    for model in models:
        preds = np.asarray(model.predict(X), dtype=float)
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        residual = np.abs(y_true - preds) / target_scale
        weighted_error = float(np.sum(sample_weights * np.clip(residual, 0.0, 1.0)))
        alpha = max(0.05, 1.0 - weighted_error)
        model_weights.append(alpha)
        sample_weights *= 1.0 + np.clip(residual, 0.0, 3.0)
        sample_weights /= sample_weights.sum()
    return model_weights


def _fit_stacking_meta_model(
    models: List[Any],
    X: pd.DataFrame,
    y: pd.Series,
    is_classification: bool,
    feature_names: List[str],
    target: str,
    class_labels: List[Any],
) -> Tuple[PrefitStackingEnsemble, float]:
    if is_classification and pd.Series(y).nunique() < 2:
        raise ValueError("Stacking requires at least two target classes in the reference dataset.")
    if len(X) < 2:
        raise ValueError("Reference dataset is too small to fit an ensemble.")

    can_stratify = is_classification and len(X) >= 10 and pd.Series(y).value_counts().min() >= 2
    can_split = len(X) >= 10

    if can_split:
        split_kwargs = {"test_size": 0.2, "random_state": 42}
        if can_stratify:
            split_kwargs["stratify"] = y
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    bootstrap = PrefitStackingEnsemble(
        models=models,
        meta_model=LogisticRegression(max_iter=1000) if is_classification else Ridge(),
        task_type="classification" if is_classification else "regression",
        feature_names=feature_names,
        class_labels=class_labels,
        target_name=target,
    )

    train_meta = bootstrap._build_meta_features(X_train)
    meta_model = LogisticRegression(max_iter=1000) if is_classification else Ridge()
    meta_model.fit(train_meta, y_train)

    ensemble = PrefitStackingEnsemble(
        models=models,
        meta_model=meta_model,
        task_type="classification" if is_classification else "regression",
        feature_names=feature_names,
        class_labels=class_labels,
        target_name=target,
        meta_feature_names=list(train_meta.columns),
    )

    score = (
        _evaluate_classifier(ensemble, X_test, y_test)
        if is_classification
        else _evaluate_regressor(ensemble, X_test, y_test)
    )
    return ensemble, score


def build_ensemble(
    job_ids: List[str],
    strategy: str = "voting",
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(job_ids, list) or len(job_ids) < 2:
        return {"error": "Need at least 2 completed jobs to build an ensemble."}

    loaded_models: List[Any] = []
    named_models: List[Tuple[str, Any]] = []
    individual_scores: List[Dict[str, Any]] = []
    feature_sets: List[List[str]] = []
    is_classification: Optional[bool] = None
    metric_name: Optional[str] = None
    target: Optional[str] = None
    reference_dataset_id = dataset_id

    with db_session() as db:
        for job_id in job_ids:
            try:
                job = db.query(JobModel).filter(JobModel.id == job_id).first()
                if not job or job.status != "completed":
                    continue

                try:
                    results = json.loads(job.results_json) if job.results_json else {}
                except Exception:
                    results = {}

                model_path = resolve_model_path(job_id)
                if not model_path:
                    fallback_path = results.get("model_path")
                    if fallback_path and os.path.exists(fallback_path):
                        model_path = fallback_path

                if not model_path or not os.path.exists(model_path):
                    continue

                model = joblib.load(model_path)
                model_name = str(results.get("best_model") or f"model_{job_id[:8]}")

                if is_classification is None:
                    is_classification = bool(results.get("is_classification", True))
                    metric_name = results.get("metric_name") or ("Accuracy" if is_classification else "R²")
                    target = results.get("target")
                    if not reference_dataset_id:
                        reference_dataset_id = job.dataset_id
                else:
                    if bool(results.get("is_classification", True)) != is_classification:
                        return {"error": "Selected jobs mix classification and regression models."}
                    incoming_target = results.get("target")
                    if target and incoming_target and incoming_target != target:
                        return {"error": "Selected jobs were trained with different target columns."}

                loaded_models.append(model)
                named_models.append((model_name, model))
                feature_names = _extract_feature_names(model, results)
                feature_sets.append(feature_names)
                individual_scores.append(
                    {
                        "job_id": job_id,
                        "model": model_name,
                        "score": results.get("score"),
                        "metric_name": results.get("metric_name"),
                        "results": results,
                        "feature_names": feature_names,
                    }
                )
            except Exception:
                continue

    if len(loaded_models) < 2:
        return {"error": "Could not load at least 2 valid models. Ensure jobs are completed."}

    reference_df = _load_reference_dataset(reference_dataset_id)
    feature_names = _resolve_feature_contract(feature_sets, reference_df, target)

    if reference_df is None or not isinstance(target, str) or target not in reference_df.columns:
        return {"error": "Reference dataset is unavailable or missing the target column needed for ensemble fitting."}
    if not feature_names:
        return {"error": "Selected jobs do not share a reusable feature contract."}

    X = reference_df.drop(columns=[target]).copy()
    y = reference_df[target].copy()
    if X.empty or y.empty:
        return {"error": "Reference dataset does not contain enough rows to fit an ensemble."}

    if not is_classification:
        y = pd.to_numeric(y, errors="coerce")
        valid_mask = y.notna()
        X = X.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)

    if len(X) < 2 or len(y) < 2:
        return {"error": "Reference dataset does not contain enough valid rows to fit an ensemble."}

    score_weights = [_safe_score(item.get("score")) for item in individual_scores]
    class_labels = _classification_labels(y, loaded_models) if is_classification else []
    strategy_weights = list(score_weights)

    try:
        normalized_strategy = str(strategy or "voting").strip().lower()
        if normalized_strategy in {"stacking", "stacked"}:
            ensemble_model, ensemble_score = _fit_stacking_meta_model(
                models=loaded_models,
                X=X,
                y=y,
                is_classification=bool(is_classification),
                feature_names=feature_names,
                target=target,
                class_labels=class_labels,
            )
            strategy = "stacking"
        elif normalized_strategy == "bagging":
            strategy_weights = _fit_bagging_weights(
                models=loaded_models,
                X=X,
                y=y,
                is_classification=bool(is_classification),
            )
            ensemble_model = PrefitVotingEnsemble(
                models=loaded_models,
                weights=strategy_weights,
                task_type="classification" if is_classification else "regression",
                feature_names=feature_names,
                class_labels=class_labels,
                target_name=target or "",
            )
            ensemble_score = (
                _evaluate_classifier(ensemble_model, X, y)
                if is_classification
                else _evaluate_regressor(ensemble_model, X, y)
            )
            strategy = "bagging"
        elif normalized_strategy == "boosting":
            ranked = sorted(
                zip(named_models, loaded_models, individual_scores, score_weights),
                key=lambda item: _safe_score(item[2].get("score"), default=0.0),
                reverse=True,
            )
            named_models = [item[0] for item in ranked]
            loaded_models = [item[1] for item in ranked]
            individual_scores = [item[2] for item in ranked]
            strategy_weights = _fit_boosting_weights(
                models=loaded_models,
                X=X,
                y=y,
                is_classification=bool(is_classification),
            )
            ensemble_model = PrefitVotingEnsemble(
                models=loaded_models,
                weights=strategy_weights,
                task_type="classification" if is_classification else "regression",
                feature_names=feature_names,
                class_labels=class_labels,
                target_name=target or "",
            )
            ensemble_score = (
                _evaluate_classifier(ensemble_model, X, y)
                if is_classification
                else _evaluate_regressor(ensemble_model, X, y)
            )
            strategy = "boosting"
        else:
            ensemble_model = PrefitVotingEnsemble(
                models=loaded_models,
                weights=score_weights,
                task_type="classification" if is_classification else "regression",
                feature_names=feature_names,
                class_labels=class_labels,
                target_name=target or "",
            )
            ensemble_score = (
                _evaluate_classifier(ensemble_model, X, y)
                if is_classification
                else _evaluate_regressor(ensemble_model, X, y)
            )
            strategy = "weighted_voting"
    except Exception as exc:
        return {"error": f"Failed to fit ensemble on the reference dataset: {exc}"}

    normalized_weights = _normalize_weights(strategy_weights)

    ensemble_id = str(uuid4())
    ensemble_path = None

    try:
        ModelRegistry.save_model(
            ensemble_id,
            ensemble_model,
            {
                "feature_names": feature_names,
                "target": target,
                "best_model": f"Ensemble ({strategy.replace('_', ' ').title()})",
                "metric_name": metric_name,
                "is_classification": is_classification,
                "preprocessor": "ensemble_pipeline",
                "ensemble_strategy": strategy,
            },
        )
        ensemble_path = get_model_path(ensemble_id)
    except Exception:
        ensemble_path = None

    leaderboard = [
        {"model": row.get("model"), "score": row.get("score")}
        for row in sorted(
            individual_scores,
            key=lambda item: _safe_score(item.get("score"), default=0.0),
            reverse=True,
        )
        if row.get("model")
    ]

    results = {
        "best_model": f"Ensemble ({strategy.title()})",
        "score": ensemble_score,
        "metric_name": metric_name,
        "leaderboard": leaderboard,
        "tested_models": individual_scores,
        "is_classification": is_classification,
        "feature_names": feature_names,
        "target": target,
        "model_path": ensemble_path,
        "execution_profile": {
            "source": "ensemble",
            "strategy": strategy,
            "base_jobs": job_ids,
            "weights": normalized_weights,
            "shared_feature_count": len(feature_names),
            "shared_features": feature_names,
            "ensemble_builder_version": "v3_prefit",
        },
        "sanitizer_report": {
            "shared_feature_count": len(feature_names),
            "shared_features": feature_names,
            "reference_dataset_id": reference_dataset_id,
        },
    }

    try:
        save_metrics(ensemble_id, results)
    except Exception:
        pass

    try:
        from core.integrations import MLTracking

        with db_session() as db:
            db.add(
                JobModel(
                    id=ensemble_id,
                    dataset_id=reference_dataset_id or "",
                    status="completed",
                    history_json=json.dumps(["Ensemble job created from completed runs."]),
                    results_json=json.dumps(results),
                    model_path=ensemble_path,
                    story=f"Ensemble built using {strategy} from {len(job_ids)} source models.",
                    params_json=json.dumps(
                        {
                            "job_ids": job_ids,
                            "strategy": strategy,
                            "dataset_id": reference_dataset_id,
                            "source": "ensemble",
                        }
                    ),
                )
            )

        MLTracking.log_run(
            ensemble_id,
            params={"goal": "Ensemble", "mode": strategy, "metric_name": metric_name},
            metrics=results,
            artifact_path=ensemble_path,
        )
    except Exception:
        pass

    return {
        "ensemble_id": ensemble_id,
        "job_id": ensemble_id,
        "strategy": strategy,
        "models_combined": [name for name, _ in named_models],
        "individual_scores": individual_scores,
        "component_weights": [
            {
                "job_id": row.get("job_id"),
                "model": row.get("model"),
                "score": row.get("score"),
                "weight": normalized_weights[index] if index < len(normalized_weights) else None,
            }
            for index, row in enumerate(individual_scores)
        ],
        "ensemble_score": ensemble_score,
        "metric_name": metric_name,
        "is_classification": is_classification,
        "ensemble_path": ensemble_path,
        "note": "Ensemble is built from already-fitted source runs and evaluated on the selected reference dataset.",
        "shared_feature_count": len(feature_names),
        "shared_features": feature_names,
    }

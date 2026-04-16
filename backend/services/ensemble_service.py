"""
services/ensemble_service.py
Feature 7: Build ensemble models from multiple completed training jobs.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional
from uuid import uuid4

import pandas as pd


def build_ensemble(
    job_ids: List[str],
    strategy: str = "voting",
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    import joblib
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score

    from infra.database import get_db, JobModel, DatasetModel
    from infra.storage import resolve_model_path
    from core.file_loader import load_dataframe

    if not isinstance(job_ids, list) or len(job_ids) < 2:
        return {"error": "Need at least 2 completed jobs to build an ensemble."}

    estimators = []
    individual_scores = []
    is_clf = None
    metric_name = None
    target = None
    feature_names = []
    test_df = None
    reference_dataset_id = dataset_id

    with get_db() as db:
        for jid in job_ids:
            try:
                job = db.query(JobModel).filter(JobModel.id == jid).first()
                if not job or job.status != "completed":
                    continue

                try:
                    results = json.loads(job.results_json) if job.results_json else {}
                except Exception:
                    results = {}

                model_path = resolve_model_path(jid)

                if not model_path:
                    db_path = results.get("model_path")
                    if db_path and os.path.exists(db_path):
                        model_path = db_path
                    elif db_path:
                        alt_path = os.path.join(os.getcwd(), db_path)
                        if os.path.exists(alt_path):
                            model_path = alt_path

                if not model_path or not os.path.exists(model_path):
                    continue

                model = joblib.load(model_path)

                name = results.get("best_model") or f"model_{jid[:6]}"
                name = str(name)

                estimators.append((name, model))

                individual_scores.append({
                    "job_id": jid,
                    "model": name,
                    "score": results.get("score"),
                    "metric_name": results.get("metric_name"),
                    "results": results,
                })

                if is_clf is None:
                    is_clf = bool(results.get("is_classification", True))
                    metric_name = results.get("metric_name", "Accuracy")
                    target = results.get("target")
                    feature_names = results.get("feature_names") or []
                    if not reference_dataset_id:
                        reference_dataset_id = job.dataset_id

                    dataset = db.query(DatasetModel).filter(
                        DatasetModel.id == reference_dataset_id
                    ).first()

                    if dataset and dataset.file_path and os.path.exists(dataset.file_path):
                        try:
                            test_df = load_dataframe(filepath=dataset.file_path)
                        except Exception:
                            test_df = None
                else:
                    if bool(results.get("is_classification", True)) != is_clf:
                        return {"error": "Selected jobs mix classification and regression models."}
                    if target and results.get("target") and results.get("target") != target:
                        return {"error": "Selected jobs were trained with different target columns."}

            except Exception:
                continue

    if len(estimators) < 2:
        return {"error": "Could not load at least 2 valid models. Ensure jobs are completed."}

    seen: Dict[str, int] = {}
    unique_estimators = []

    for name, model in estimators:
        if name in seen:
            seen[name] += 1
            unique_estimators.append((f"{name}_{seen[name]}", model))
        else:
            seen[name] = 0
            unique_estimators.append((name, model))

    estimators = unique_estimators

    try:
        numeric_scores = []
        for row in individual_scores:
            try:
                numeric_scores.append(max(float(row.get("score") or 0.0), 0.01))
            except Exception:
                numeric_scores.append(1.0)
        weight_sum = sum(numeric_scores) or len(numeric_scores) or 1
        normalized_weights = [round(score / weight_sum, 4) for score in numeric_scores]

        if strategy == "stacking":
            from sklearn.ensemble import StackingClassifier, StackingRegressor

            if is_clf:
                ensemble = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000),
                    cv=3,
                    passthrough=True,
                )
            else:
                ensemble = StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(),
                    cv=3,
                    passthrough=True,
                )
        else:
            if is_clf:
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting="soft",
                    weights=normalized_weights,
                )
            else:
                ensemble = VotingRegressor(estimators=estimators, weights=normalized_weights)

    except Exception as e:
        return {"error": f"Failed to build ensemble: {e}"}

    ensemble_score = None

    if test_df is not None and isinstance(target, str) and target in test_df.columns:
        try:
            from sklearn.preprocessing import LabelEncoder

            y = test_df[target]
            X = test_df.drop(columns=[target])

            if is_clf:
                le = LabelEncoder()
                y_enc = le.fit_transform(y.astype(str))
            else:
                y_enc = pd.to_numeric(y, errors="coerce").fillna(0)

            split_kwargs = {"test_size": 0.2, "random_state": 42}
            if is_clf and len(pd.Series(y_enc).unique()) > 1:
                split_kwargs["stratify"] = y_enc

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_enc, **split_kwargs
            )

            ensemble.fit(X_tr, y_tr)
            preds = ensemble.predict(X_te)

            raw_score = accuracy_score(y_te, preds) if is_clf else r2_score(y_te, preds)
            ensemble_score = round(float(raw_score) * 100, 1)

            # Persist a fitted artifact that can actually be reused later.
            ensemble.fit(X, y_enc)

        except Exception:
            ensemble_score = None

    if ensemble_score is None:
        return {"error": "Failed to fit ensemble on the reference dataset."}

    ensemble_id = str(uuid4())

    try:
        from infra.database import JobModel, get_db
        from infra.storage import ModelRegistry, get_model_path, save_metrics
        from core.integrations import MLTracking

        ensemble_path = get_model_path(ensemble_id)
        ModelRegistry.save_model(
            ensemble_id,
            ensemble,
            {
                "feature_names": feature_names,
                "target": target,
                "best_model": f"Ensemble ({strategy.title()})",
                "metric_name": metric_name,
                "is_classification": is_clf,
                "preprocessor": "ensemble_pipeline",
            },
        )
        ensemble_path = get_model_path(ensemble_id)
    except Exception:
        ensemble_path = None

    leaderboard = [
        {
            "model": row.get("model"),
            "score": row.get("score"),
        }
        for row in sorted(
            individual_scores,
            key=lambda x: float(x.get("score", 0) or 0),
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
        "is_classification": is_clf,
        "feature_names": feature_names,
        "target": target,
        "model_path": ensemble_path,
        "execution_profile": {
            "source": "ensemble",
            "strategy": strategy,
            "base_jobs": job_ids,
            "weights": normalized_weights,
        },
    }

    try:
        save_metrics(ensemble_id, results)
    except Exception:
        pass

    try:
        with get_db() as db:
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
        "models_combined": [name for name, _ in estimators],
        "individual_scores": individual_scores,
        "ensemble_score": ensemble_score,
        "metric_name": metric_name,
        "is_classification": is_clf,
        "ensemble_path": ensemble_path,
        "note": "Ensemble score is based on a holdout split from the reference dataset, then re-fit on all rows for reuse.",
    }

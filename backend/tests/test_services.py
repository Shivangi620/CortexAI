import pandas as pd
import numpy as np
import json
import os
import joblib
import optuna
import pytest
import core.meta_learning as meta_learning_module
import services.training.model_selector as model_selector_module
from sklearn.linear_model import LinearRegression, LogisticRegression
from services.training.preprocessing import (
    auto_clean_data,
    fuzzy_merge_labels,
    make_lite_preprocessor,
    make_preprocessor,
)
from services.data_sanitizer import sanitize_dataframe
from services.drift_service import get_drift_dashboard
from services.ensemble_service import build_ensemble
from core.synthetic import generate_synthetic
from services.training.forecasting import estimate_training_forecast
from services.training.evaluator import _resolve_scoring, stability_check, normalize_training_controls, detect_task_type
from services.training.components import (
    _best_completed_trial,
    _coerce_estimator_instance,
    _prune_correlated_candidates,
    _prune_optuna_candidates,
    _resolve_final_model_choice,
    _select_fallback_model,
    _score_for_ranking,
    _simple_model_is_good_enough,
    EvaluationComponent,
)
from services.training.components import _resolve_target_column_name
from services.training.model_selector import ModelSelector
from core.export import (
    _explain_script_content,
    _model_import_and_init,
    _pipeline_steps_content,
    _source_manifest_content,
    build_export_bundle_filename,
)
from services.explain_service import generate_counterfactual
from services.training.inference import TabularModelPipeline
from infra.database import DatasetModel, JobModel
from infra.storage import get_model_path
from core.file_loader import load_dataframe
from .conftest import TestingSessionLocal


class DummyInferenceArtifact:
    def __init__(self, bias: float = 0.0):
        self.bias = bias
        self.feature_names_in_ = ["age", "income"]
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, X):
        frame = pd.DataFrame(X).copy()
        age = pd.to_numeric(frame["age"], errors="coerce").fillna(0.0)
        income = pd.to_numeric(frame["income"], errors="coerce").fillna(0.0)
        logits = ((age - 40.0) / 12.0) + ((income - 60000.0) / 25000.0) + self.bias
        positive = 1.0 / (1.0 + np.exp(-logits))
        negative = 1.0 - positive
        return np.column_stack([negative.to_numpy(), positive.to_numpy()])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_fuzzy_merge_labels():
    s = pd.Series(["Apple", "apple", "banana", "Banana", "Apple", "orange"])
    merged = fuzzy_merge_labels(s, threshold=0.8)
    
    # "Apple" and "apple" should merge. "banana" and "Banana" should merge.
    cleaned_counts = merged.value_counts()
    
    assert len(cleaned_counts) == 3
    # Exact caps depend on which was more frequent or first, but counts group uniformly
    assert cleaned_counts.iloc[0] == 3  # apple group
    assert cleaned_counts.iloc[1] == 2  # banana group


def test_auto_clean_data_drops_constants_and_nulls():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "id_col": [1, 2, 3, 4, 5],                 # Should not be dropped if variance is perfectly 1, but let's see
        "constant_col": ["a", "a", "a", "a", "a"], # 100% constant, should drop
        "null_col": [np.nan, np.nan, np.nan, np.nan, "1"], # >90% null, should drop
        "score_col": [10.5, 12.3, 8.1, 7.5, 15.0]
    })

    cleaned_df, logs = auto_clean_data(df, target="target")

    # "constant_col" and "null_col" drop. "id_col" drops because unique == length and named "id_"
    assert "constant_col" not in cleaned_df.columns
    assert "null_col" not in cleaned_df.columns
    assert "id_col" not in cleaned_df.columns
    
    # Target and valid_col remain
    assert "target" in cleaned_df.columns
    assert "score_col" in cleaned_df.columns


def test_auto_clean_null_standardization():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4],
        "mixed": ["val1", "n/a", "null", "val2"]
    })
    
    cleaned, _ = auto_clean_data(df, "target")
    # n/a and null should become NaN
    assert pd.isna(cleaned.loc[1, "mixed"])
    assert pd.isna(cleaned.loc[2, "mixed"])
    assert cleaned.loc[0, "mixed"] == "val1"


def test_normalize_result_contract():
    from infra.result_contract import normalize_results

    partial_results = {
        "best_model": None,
        "score": "95.2",
        "leaderboard": [{"model": "A", "score": 95}],
        "shap_summary": [("f1", 0.2)],
        "reasoning": "Completed training",
    }

    normalized = normalize_results(partial_results)

    assert normalized["best_model"] == ""
    assert normalized["score"] == 95.2
    assert normalized["metric_name"] == ""
    assert normalized["leaderboard"] == [{"model": "A", "score": 95}]
    assert normalized["model_path"] == ""
    assert normalized["shap_summary"] == {}
    assert normalized["reasoning"] == ["Completed training"]


def test_normalize_result_contract_sanitizes_non_finite_values():
    from infra.result_contract import normalize_results

    partial_results = {
        "best_model": "LGBM",
        "score": np.nan,
        "leaderboard": [{"model": "A", "score": np.nan, "mse": np.inf}],
        "shap_summary": {"f1": np.nan, "f2": np.inf},
        "reasoning": ["ok", np.nan],
    }

    normalized = normalize_results(partial_results)

    assert normalized["score"] == 0.0
    assert normalized["leaderboard"] == [{"model": "A", "score": None, "mse": None}]
    assert normalized["shap_summary"] == {"f1": None, "f2": None}
    assert normalized["reasoning"] == ["ok", None]


@pytest.mark.parametrize("strategy", ["bagging", "boosting", "stacking"])
def test_build_ensemble_supports_prefit_artifact_strategies(tmp_path, strategy):
    dataset_id = f"ensemble-ds-{strategy}"
    first_job_id = f"ensemble-base-1-{strategy}"
    second_job_id = f"ensemble-base-2-{strategy}"
    dataset_path = tmp_path / "ensemble_reference.csv"
    frame = pd.DataFrame(
        {
            "age": [22, 25, 28, 31, 34, 37, 42, 48, 53, 59, 63, 68],
            "income": [28000, 32000, 36000, 39000, 43000, 47000, 61000, 68000, 74000, 82000, 91000, 99000],
            "target": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )
    frame.to_csv(dataset_path, index=False)

    first_path = get_model_path(first_job_id)
    second_path = get_model_path(second_job_id)
    joblib.dump(DummyInferenceArtifact(bias=-0.2), first_path)
    joblib.dump(DummyInferenceArtifact(bias=0.25), second_path)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id=dataset_id,
                file_path=os.fspath(dataset_path),
                profile_json=json.dumps({"rows": len(frame), "cols": len(frame.columns), "columns": list(frame.columns)}),
                source_type="upload",
            )
        )
        for job_id, model_name, model_path, score in [
            (first_job_id, "Hist Gradient Boosting", first_path, 87.4),
            (second_job_id, "LightGBM", second_path, 89.1),
        ]:
            db.add(
                JobModel(
                    id=job_id,
                    dataset_id=dataset_id,
                    status="completed",
                    model_path=model_path,
                    results_json=json.dumps(
                        {
                            "best_model": model_name,
                            "score": score,
                            "metric_name": "Accuracy",
                            "feature_names": ["age", "income"],
                            "target": "target",
                            "is_classification": True,
                            "model_path": model_path,
                        }
                    ),
                )
            )
        db.commit()

    body = build_ensemble(
        [first_job_id, second_job_id],
        strategy=strategy,
        dataset_id=dataset_id,
    )

    assert "error" not in body, body.get("error")
    assert body["strategy"] == strategy
    assert body["metric_name"] == "Accuracy"
    assert set(body["models_combined"]) == {"Hist Gradient Boosting", "LightGBM"}
    assert isinstance(body["ensemble_score"], float)
    assert os.path.exists(body["ensemble_path"])

    saved = joblib.load(body["ensemble_path"])
    proba = saved.predict_proba(frame[["age", "income"]].head(3))
    assert proba.shape == (3, 2)


def test_generate_synthetic_preserves_mixed_column_types():
    frame = pd.DataFrame(
        {
            "amount": [10.5, 12.0, np.nan, 18.75],
            "city": pd.Series(["Austin", "Boston", "Austin", None], dtype="object"),
            "flag": pd.Series([True, False, True, None], dtype="boolean"),
            "event_time": pd.to_datetime(["2024-01-01", "2024-01-03", None, "2024-01-08"]),
        }
    )

    expanded, synthetic = generate_synthetic(frame, 12, random_state=7)

    assert len(expanded) == len(frame) + 12
    assert len(synthetic) == 12
    assert synthetic["amount"].notna().any()
    assert synthetic["city"].dropna().isin(["Austin", "Boston"]).all()
    assert synthetic["city"].notna().any()
    assert str(synthetic["flag"].dtype) in {"boolean", "bool"}
    assert synthetic["flag"].dropna().isin([True, False]).all()
    assert pd.api.types.is_datetime64_any_dtype(synthetic["event_time"])


def test_generate_synthetic_preserves_boolean_like_object_columns_and_exact_count():
    frame = pd.DataFrame(
        {
            "flag": [True, False, None, True, None],
            "segment": ["A", "B", "A", "B", None],
            "value": [1.0, 2.5, 1.8, 2.2, 1.4],
        }
    )

    expanded, synthetic = generate_synthetic(frame, 9, random_state=3)

    assert len(synthetic) == 9
    assert len(expanded) == len(frame) + 9
    assert str(synthetic["flag"].dtype) == "boolean"
    assert synthetic["flag"].dropna().isin([True, False]).all()


def test_load_dataframe_supports_markdown_documents():
    markdown = b"# Churn Notes\n\nCustomer called support twice.\nLikely renewal risk.\n"

    df = load_dataframe(contents=markdown, filename="notes.md")

    assert not df.empty
    assert list(df.columns) == ["source_file", "segment_type", "segment_index", "text", "text_length"]
    assert df.iloc[0]["source_file"] == "notes.md"
    assert df.iloc[0]["segment_type"] == "block"
    assert "Churn Notes" in df.iloc[0]["text"]


def test_drift_dashboard_respects_custom_thresholds():
    baseline = {"age": {"mean": 10.0, "std": 1.0, "count": 200}}
    current_df = pd.DataFrame({"age": np.linspace(10.2, 10.8, 200)})

    report = get_drift_dashboard(
        current_df=current_df,
        baseline_stats=baseline,
        feature_names=["age"],
        warning_threshold=0.01,
        critical_threshold=0.02,
    )

    assert report["thresholds"]["warning_psi"] == 0.01
    assert report["thresholds"]["critical_psi"] == 0.02
    assert "alert_summary" in report
    assert report["alert_summary"]["level"] in {"warning", "critical", "stable"}


def test_drift_dashboard_includes_retrain_recommendation(monkeypatch):
    def fake_select_pool(rows, is_clf, goal, profile, mode=""):
        return (
            {
                "Random Forest": object(),
                "Hist Gradient Boosting": object(),
                "XGBoost": object(),
            },
            {
                "confidence": 78.0,
                "memory_signal": {"applied": True, "reordered_models": ["Random Forest"]},
            },
        )

    def fake_cross_dataset_insights(profile):
        return {
            "historical_runs": 12,
            "most_common_winner": {"model": "Hist Gradient Boosting", "count": 5},
        }

    monkeypatch.setattr(model_selector_module.ModelSelector, "select_pool", staticmethod(fake_select_pool))
    monkeypatch.setattr(meta_learning_module, "get_cross_dataset_insights", fake_cross_dataset_insights)

    baseline = {"age": {"mean": 10.0, "std": 1.0, "count": 200}}
    current_df = pd.DataFrame({"age": np.linspace(12.0, 14.0, 200)})

    report = get_drift_dashboard(
        current_df=current_df,
        baseline_stats=baseline,
        feature_names=["age"],
        task_type="classification",
        current_model="Logistic Regression",
        metric_name="F1-score",
        goal="Balanced",
        mode="Balanced",
        warning_threshold=0.01,
        critical_threshold=0.02,
    )

    recommendation = report["retrain_recommendation"]
    assert recommendation["recommended_goal"] in {"Balanced", "Performance"}
    assert recommendation["candidate_models"] == [
        "Random Forest",
        "Hist Gradient Boosting",
        "XGBoost",
    ]
    assert recommendation["historical_winner"] == "Hist Gradient Boosting"
    assert recommendation["memory_applied"] is True


def test_drift_dashboard_critical_drift_escalates_retrain_lane(monkeypatch):
    def fake_select_pool(rows, is_clf, goal, profile, mode=""):
        assert goal == "Performance"
        assert mode == "Full"
        return (
            {"Hist Gradient Boosting": object(), "XGBoost": object()},
            {"confidence": 66.0, "memory_signal": {"applied": False, "reordered_models": []}},
        )

    monkeypatch.setattr(model_selector_module.ModelSelector, "select_pool", staticmethod(fake_select_pool))
    monkeypatch.setattr(
        meta_learning_module,
        "get_cross_dataset_insights",
        lambda profile: {"historical_runs": 0, "most_common_winner": {"model": "", "count": 0}},
    )

    baseline = {"age": {"mean": 10.0, "std": 1.0, "count": 200}}
    current_df = pd.DataFrame({"age": np.linspace(40.0, 60.0, 200)})

    report = get_drift_dashboard(
        current_df=current_df,
        baseline_stats=baseline,
        feature_names=["age"],
        task_type="classification",
        current_model="Logistic Regression",
        goal="Balanced",
        mode="Balanced",
        warning_threshold=0.01,
        critical_threshold=0.02,
    )

    recommendation = report["retrain_recommendation"]
    assert recommendation["recommended_goal"] == "Performance"
    assert recommendation["recommended_mode"] == "Full"


def test_training_forecast_returns_runtime_and_budget():
    forecast = estimate_training_forecast(
        profile={
            "rows": 1200,
            "cols": 12,
            "columns": [f"f{i}" for i in range(12)],
            "task_type": "classification",
            "missing_pct": 4.5,
        },
        target_column="target",
        goal="Balanced",
        mode="Full",
        selected_features=["f1", "f2", "f3"],
        cv_folds=5,
        handle_imbalance=True,
        auto_clean=True,
        eval_metric="Accuracy",
    )

    assert forecast["goal"] == "Balanced"
    assert forecast["mode"] == "Full"
    assert forecast["estimated_duration_seconds"]["max"] >= forecast["estimated_duration_seconds"]["min"]
    assert forecast["optuna_trials"] == 32
    assert forecast["estimated_feature_count"] == 3


def test_classification_scoring_supports_accuracy_precision_recall():
    assert _resolve_scoring("Accuracy", True) == "accuracy"
    assert _resolve_scoring("Precision", True) == "precision_weighted"
    assert _resolve_scoring("Recall", True) == "recall_weighted"
    assert _resolve_scoring("F1-score", True) == "f1_weighted"
    assert _resolve_scoring("ROC-AUC", True) == "roc_auc_ovr_weighted"


def test_stability_check_uses_requested_classification_metric_and_returns_all_metrics():
    X = pd.DataFrame(
        {
            "x1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "x2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

    score, std_score, extras = stability_check(
        LogisticRegression(max_iter=1000),
        X,
        y,
        True,
        scoring_name="precision_weighted",
    )

    assert 0 <= score <= 1
    assert std_score >= 0
    assert {"accuracy", "precision", "recall", "f1"} <= set(extras.keys())


def test_stability_check_supports_roc_auc_metric():
    X = pd.DataFrame(
        {
            "x1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "x2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

    score, std_score, extras = stability_check(
        LogisticRegression(max_iter=1000),
        X,
        y,
        True,
        scoring_name="roc_auc_ovr_weighted",
    )

    assert 0 <= score <= 1
    assert std_score >= 0
    assert "roc_auc" in extras


def test_stability_check_uses_requested_regression_metric_and_returns_all_metrics():
    X = pd.DataFrame({"x1": np.arange(20), "x2": np.arange(20) * 2})
    y = pd.Series(np.arange(20) * 3 + 5)

    score, std_score, extras = stability_check(
        LinearRegression(),
        X,
        y,
        False,
        scoring_name="neg_mean_squared_error",
    )

    assert score <= 0
    assert std_score >= 0
    assert {"r2", "mse", "rmse", "mae"} <= set(extras.keys())


def test_normalize_training_controls_enforces_task_specific_metric_rules():
    classification = normalize_training_controls(
        task_type="classification",
        goal="performance",
        mode="full",
        eval_metric="RMSE",
        handle_imbalance=True,
    )
    assert classification["task_type"] == "classification"
    assert classification["goal"] == "Performance"
    assert classification["mode"] == "Full"
    assert classification["eval_metric"] == "F1-score"
    assert classification["handle_imbalance"] is True
    assert classification["warnings"]

    regression = normalize_training_controls(
        task_type="regression",
        goal="speed",
        mode="fast",
        eval_metric="Precision",
        handle_imbalance=True,
    )
    assert regression["task_type"] == "regression"
    assert regression["goal"] == "Speed"
    assert regression["mode"] == "Fast"
    assert regression["eval_metric"] == "RMSE"
    assert regression["handle_imbalance"] is False
    assert len(regression["warnings"]) >= 1


def test_normalize_training_controls_uses_quality_first_defaults_when_metric_not_provided():
    classification = normalize_training_controls(
        task_type="classification",
        goal="balanced",
        mode="balanced",
    )
    regression = normalize_training_controls(
        task_type="regression",
        goal="balanced",
        mode="balanced",
    )

    assert classification["eval_metric"] == "F1-score"
    assert regression["eval_metric"] == "RMSE"


def test_detect_task_type_keeps_discrete_numeric_scores_as_regression_without_classification_hints():
    y = pd.Series([1, 2, 3, 4, 5] * 40)

    decision = detect_task_type(y, target_name="customer_satisfaction_score")

    assert decision["task_type"] == "regression"


def test_sanitize_dataframe_preserves_original_target_label_case():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": ["Yes", "No", "Yes", "No"],
        }
    )

    result = sanitize_dataframe(df, target="target")

    assert result.df["target"].tolist() == ["Yes", "No", "Yes", "No"]


def test_validation_summary_marks_lower_is_better_holdout_regression_as_overfit():
    component = EvaluationComponent()

    summary = component._build_validation_summary(
        scoring_name="neg_root_mean_squared_error",
        score_meta={"label": "CV RMSE", "direction": "lower_is_better"},
        cv_mean_raw=-4.0,
        holdout_raw=-6.0,
        cv_display=4.0,
        holdout_display=6.0,
    )

    assert summary["status"] == "possible_overfit"
    assert summary["generalization_gap_display"] == 2.0


def test_validation_summary_marks_higher_is_better_holdout_gain_as_improvement():
    component = EvaluationComponent()

    summary = component._build_validation_summary(
        scoring_name="accuracy",
        score_meta={"label": "CV Accuracy", "direction": "higher_is_better"},
        cv_mean_raw=0.82,
        holdout_raw=0.88,
        cv_display=82.0,
        holdout_display=88.0,
    )

    assert summary["status"] == "holdout_outperformed_cv"


def test_regression_ranking_preserves_negative_error_scores():
    assert _score_for_ranking(-4.5, False) == -4.5
    assert _score_for_ranking(-1.2, False) > _score_for_ranking(-4.5, False)


def test_best_completed_trial_ignores_pruned_trials():
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        if trial.number == 0:
            raise optuna.TrialPruned("simulated failure")
        return -2.5

    study.optimize(objective, n_trials=2)
    best_trial = _best_completed_trial(study)

    assert best_trial is not None
    assert best_trial.state == optuna.trial.TrialState.COMPLETE
    assert float(best_trial.value) == -2.5


def test_tabular_pipeline_clamps_pca_for_small_training_folds():
    X = pd.DataFrame(
        {
            "f1": [1, 2, 3],
            "f2": [2, 3, 4],
            "f3": [3, 4, 5],
            "f4": [4, 5, 6],
            "f5": [5, 6, 7],
            "f6": [6, 7, 8],
        }
    )
    y = pd.Series([0, 1, 0])

    pipeline = TabularModelPipeline(
        base_estimator=LogisticRegression(max_iter=1000),
        feature_columns=list(X.columns),
        preprocessing="lite",
        pca_mode="always",
        pca_components=6,
    )

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(X)


def test_drift_dashboard_only_checks_model_feature_scope():
    baseline = {
        "used_feature": {"mean": 10.0, "std": 1.0, "count": 200},
        "unused_feature": {"mean": 50.0, "std": 1.0, "count": 200},
        "target": {"mean": 0.5, "std": 0.1, "count": 200},
    }
    current_df = pd.DataFrame(
        {
            "used_feature": np.linspace(12.0, 14.0, 200),
            "unused_feature": np.linspace(90.0, 110.0, 200),
            "target": np.linspace(0.0, 1.0, 200),
        }
    )

    report = get_drift_dashboard(
        current_df=current_df,
        baseline_stats=baseline,
        feature_names=["used_feature"],
        target_name="target",
        warning_threshold=0.01,
        critical_threshold=0.02,
    )

    assert report["total_features_checked"] == 1
    assert [item["feature"] for item in report["feature_drift"]] == ["used_feature"]


def test_generate_counterfactual_accepts_target_prediction_for_classification(tmp_path):
    csv_path = tmp_path / "counterfactual_classification.csv"
    csv_path.write_text("x1,target\n0,no\n0,no\n1,yes\n1,yes\n", encoding="utf-8")

    model = LogisticRegression().fit(pd.DataFrame({"x1": [0, 0, 1, 1]}), [0, 0, 1, 1])
    model_path = get_model_path("cf-service-job")
    joblib.dump(model, model_path)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="cf-service-ds",
                file_path=str(csv_path),
                profile_json=json.dumps({"columns": ["x1", "target"]}),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="cf-service-job",
                dataset_id="cf-service-ds",
                status="completed",
                model_path=model_path,
                results_json=json.dumps(
                    {
                        "feature_names": ["x1"],
                        "is_classification": True,
                        "model_path": model_path,
                    }
                ),
            )
        )
        db.commit()

    payload = generate_counterfactual(
        "cf-service-job",
        {"feature_names": ["x1"], "is_classification": True, "model_path": model_path},
        {"x1": 0},
        target_prediction=95,
    )

    assert "error" not in payload
    assert payload["current_prediction"] in {"0", "1"}
    assert "suggestions" in payload


def test_generate_counterfactual_supports_regression_goal_seeking(tmp_path):
    csv_path = tmp_path / "counterfactual_regression.csv"
    csv_path.write_text("x1,target\n0,0\n1,10\n2,20\n3,30\n", encoding="utf-8")

    model = LinearRegression().fit(pd.DataFrame({"x1": [0, 1, 2, 3]}), [0, 10, 20, 30])
    model_path = get_model_path("reg-cf-service-job")
    joblib.dump(model, model_path)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="reg-cf-service-ds",
                file_path=str(csv_path),
                profile_json=json.dumps({"columns": ["x1", "target"]}),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="reg-cf-service-job",
                dataset_id="reg-cf-service-ds",
                status="completed",
                model_path=model_path,
                results_json=json.dumps(
                    {
                        "feature_names": ["x1"],
                        "is_classification": False,
                        "model_path": model_path,
                    }
                ),
            )
        )
        db.commit()

    payload = generate_counterfactual(
        "reg-cf-service-job",
        {"feature_names": ["x1"], "is_classification": False, "model_path": model_path},
        {"x1": 0},
        target_prediction=25,
    )

    assert "error" not in payload
    assert payload["current_prediction"] == 0.0
    assert payload["target_prediction"] == 25.0
    assert isinstance(payload["suggestions"], list)


def test_preprocessor_respects_explicit_pca_controls():
    num_cols = [f"n{i}" for i in range(12)]
    full_pre = make_preprocessor(num_cols, [], pca_mode="always", pca_components=5)
    lite_pre = make_lite_preprocessor(num_cols, [], pca_mode="always", pca_components=4)
    off_pre = make_preprocessor(num_cols, [], pca_mode="off", pca_components=5)

    full_num_steps = full_pre.transformers[0][1].named_steps
    lite_num_steps = lite_pre.transformers[0][1].named_steps
    off_num_steps = off_pre.transformers[0][1].named_steps

    assert "pca" in full_num_steps
    assert full_num_steps["pca"].n_components == 5
    assert "pca" in lite_num_steps
    assert lite_num_steps["pca"].n_components == 4
    assert "pca" not in off_num_steps


def test_full_preprocessor_feature_names_survive_custom_numeric_steps():
    preprocessor = make_preprocessor(
        ["area", "rooms", "age"],
        [],
        pca_mode="off",
    )
    X = pd.DataFrame(
        {
            "area": [800, 1200, 1600, 2200],
            "rooms": [2, 3, 3, 4],
            "age": [10, 5, 2, 1],
        }
    )

    preprocessor.fit(X, pd.Series([100000, 150000, 210000, 280000]))
    names = preprocessor.get_feature_names_out()

    assert len(names) == preprocessor.transform(X).shape[1]
    assert any("area" in str(name) for name in names)


def test_tabular_pipeline_full_preprocessing_fits_with_feature_names():
    X = pd.DataFrame(
        {
            "area": [800, 1200, 1600, 2200, 950, 1800],
            "rooms": [2, 3, 3, 4, 2, 4],
            "age": [10, 5, 2, 1, 8, 3],
        }
    )
    y = pd.Series([100000, 150000, 210000, 280000, 120000, 240000])

    pipeline = TabularModelPipeline(
        base_estimator=LinearRegression(),
        feature_columns=list(X.columns),
        preprocessing="full",
        pca_mode="off",
    )

    pipeline.fit(X, y)

    assert len(pipeline.get_feature_names_out()) == pipeline.preprocess(X).shape[1]


def test_training_forecast_includes_pca_configuration_notes():
    forecast = estimate_training_forecast(
        profile={
            "rows": 3000,
            "cols": 48,
            "columns": [f"f{i}" for i in range(48)],
            "task_type": "classification",
            "missing_pct": 3.2,
        },
        target_column="target",
        goal="Performance",
        mode="Balanced",
        selected_features=[],
        cv_folds=3,
        handle_imbalance=False,
        auto_clean=True,
        eval_metric="Precision",
        pca_mode="always",
        pca_components=10,
    )

    assert forecast["pca_mode"] == "always"
    assert forecast["pca_components"] == 10
    assert any("PCA" in note for note in forecast["notes"])


def test_training_component_keeps_optimized_model_when_present():
    optimized_model = object()
    fallback_model = object()

    final_model, winner_name = _resolve_final_model_choice(
        optimized_model,
        "LightGBM",
        [{"model": fallback_model, "name": "Random Forest"}],
    )

    assert final_model is optimized_model
    assert winner_name == "LightGBM"


def test_training_component_falls_back_to_top_candidate_when_missing():
    fallback_model = object()

    final_model, winner_name = _resolve_final_model_choice(
        None,
        None,
        [{"model": fallback_model, "name": "Random Forest"}],
    )

    assert final_model is fallback_model
    assert winner_name == "Random Forest"


def test_training_component_coerces_string_estimator_name_back_to_model_pool():
    fallback_model = object()
    recovered = _coerce_estimator_instance(
        "Random Forest",
        "Random Forest",
        {"Random Forest": fallback_model},
        [{"model": fallback_model, "name": "Random Forest"}],
    )

    assert recovered is fallback_model


def test_target_resolution_matches_case_and_separator_variants():
    resolved = _resolve_target_column_name(
        ["customer_id", "Target Value", "prediction_score"],
        "TargetValue",
    )

    assert resolved == "Target Value"


def test_model_selector_includes_stronger_sklearn_families_for_quality_goals():
    clf_pool, _ = ModelSelector.select_pool(
        rows=1000,
        is_clf=True,
        goal="Performance",
        profile={"rows": 1000, "cols": 8},
    )
    reg_pool, _ = ModelSelector.select_pool(
        rows=1000,
        is_clf=False,
        goal="Performance",
        profile={"rows": 1000, "cols": 8},
    )

    assert "Decision Tree" not in clf_pool
    assert "Extra Trees" not in clf_pool
    assert "Hist Gradient Boosting" in clf_pool
    assert "Decision Tree" not in reg_pool
    assert "Extra Trees" not in reg_pool
    assert "ElasticNet" in reg_pool


def test_model_selector_prefers_small_dataset_pattern_models_for_classification():
    clf_pool, _ = ModelSelector.select_pool(
        rows=1200,
        is_clf=True,
        goal="Performance",
        profile={"rows": 1200, "cols": 10, "num_cols": ["age", "income"], "cat_cols": ["city"]},
    )

    assert "KNN" in clf_pool
    assert "SVM" in clf_pool
    assert "MLP" not in clf_pool
    assert "Random Forest" in clf_pool


def test_model_selector_avoids_slow_small_sample_models_on_large_data():
    clf_pool, clf_rec = ModelSelector.select_pool(
        rows=75000,
        is_clf=True,
        goal="Performance",
        profile={"rows": 75000, "cols": 24, "num_cols": list(range(20)), "cat_cols": ["segment"]},
    )
    reg_pool, _ = ModelSelector.select_pool(
        rows=75000,
        is_clf=False,
        goal="Performance",
        profile={"rows": 75000, "cols": 24, "num_cols": list(range(24)), "cat_cols": []},
    )

    assert "SVM" not in clf_pool
    assert "KNN" not in clf_pool
    assert "ElasticNet" in reg_pool
    assert sum(name in clf_pool for name in ["Hist Gradient Boosting", "LightGBM", "XGBoost"]) >= 2
    assert clf_rec["goal_profile"]["dataset_traits"]["large_dataset"] is True


def test_model_selector_high_dimensional_regression_keeps_linear_models_front_and_center():
    reg_pool, _ = ModelSelector.select_pool(
        rows=800,
        is_clf=False,
        goal="Balanced",
        profile={"rows": 800, "cols": 180, "num_cols": list(range(180)), "cat_cols": []},
    )

    assert "ElasticNet" in reg_pool
    assert "Ridge" in reg_pool
    assert "Lasso" not in reg_pool
    assert "KNN" not in reg_pool


def test_model_selector_balanced_mode_stays_clean_and_fast():
    clf_pool, _ = ModelSelector.select_pool(
        rows=3200,
        is_clf=True,
        goal="Balanced",
        profile={"rows": 3200, "cols": 14, "num_cols": list(range(10)), "cat_cols": ["segment"]},
    )
    reg_pool, _ = ModelSelector.select_pool(
        rows=3200,
        is_clf=False,
        goal="Balanced",
        profile={"rows": 3200, "cols": 14, "num_cols": list(range(14)), "cat_cols": []},
    )

    assert list(clf_pool.keys())[:3] == [
        "Logistic Regression",
        "Random Forest",
        "Hist Gradient Boosting",
    ]
    assert "Decision Tree" not in clf_pool
    assert "SVM" not in clf_pool
    assert "KNN" in clf_pool
    assert list(reg_pool.keys())[:4] == [
        "Linear Regression",
        "Ridge",
        "ElasticNet",
        "Random Forest",
    ]
    assert "Decision Tree" not in reg_pool


def test_model_selector_skips_knn_for_high_feature_count_and_mlp_for_small_data():
    clf_pool, clf_rec = ModelSelector.select_pool(
        rows=1200,
        is_clf=True,
        goal="Performance",
        profile={"rows": 1200, "cols": 64, "num_cols": list(range(64)), "cat_cols": []},
    )

    assert "KNN" not in clf_pool
    assert "MLP" not in clf_pool
    assert clf_rec["goal_profile"]["dataset_traits"]["knn_allowed"] is False


def test_model_selector_includes_mlp_for_complex_performance_datasets():
    clf_pool, _ = ModelSelector.select_pool(
        rows=4200,
        is_clf=True,
        goal="Performance",
        profile={"rows": 4200, "cols": 18, "num_cols": list(range(12)), "cat_cols": ["segment", "region"]},
    )
    reg_pool, _ = ModelSelector.select_pool(
        rows=4200,
        is_clf=False,
        goal="Performance",
        profile={"rows": 4200, "cols": 18, "num_cols": list(range(16)), "cat_cols": ["segment", "region"]},
    )

    assert "MLP" in clf_pool
    assert "MLP" in reg_pool


def test_export_helper_supports_mlp_and_elasticnet():
    mlp_import, mlp_init = _model_import_and_init("MLP", True)
    elastic_import, elastic_init = _model_import_and_init("ElasticNet", False)

    assert "MLPClassifier" in mlp_import
    assert "MLPClassifier" in mlp_init
    assert "ElasticNet" in elastic_import
    assert "ElasticNet" in elastic_init


def test_model_selector_skips_advanced_boosters_for_low_complexity_datasets():
    clf_pool, clf_rec = ModelSelector.select_pool(
        rows=600,
        is_clf=True,
        goal="Performance",
        profile={
            "rows": 600,
            "cols": 4,
            "num_cols": ["age", "income"],
            "cat_cols": [],
            "target_entropy": 0.08,
            "numeric_max_corr": 0.05,
        },
    )

    assert "Hist Gradient Boosting" in clf_pool
    assert "XGBoost" not in clf_pool
    assert "LightGBM" not in clf_pool
    assert clf_rec["goal_profile"]["dataset_traits"]["low_complexity"] is True


def test_model_selector_uses_meta_memory_inside_safe_hierarchy(monkeypatch):
    def fake_zero_shot(profile, model_pool):
        return {
            "rankings": [
                {"model": "ElasticNet", "pred_score": 93.0},
                {"model": "Ridge", "pred_score": 81.0},
                {"model": "Random Forest", "pred_score": 99.0},
                {"model": "Hist Gradient Boosting", "pred_score": 88.0},
            ],
            "confidence": 84.0,
            "source": "Test Meta",
            "reason": "Historical winners favor ElasticNet here.",
        }

    monkeypatch.setattr(model_selector_module, "zero_shot_recommend", fake_zero_shot)

    reg_pool, reg_rec = ModelSelector.select_pool(
        rows=9000,
        is_clf=False,
        goal="Performance",
        profile={"rows": 9000, "cols": 18, "num_cols": list(range(18)), "cat_cols": []},
    )

    ordered = list(reg_pool.keys())
    assert ordered[0] == "Linear Regression"
    assert ordered[1] == "ElasticNet"
    assert ordered[2] == "Ridge"
    assert ordered.index("Random Forest") < ordered.index("Hist Gradient Boosting")
    assert reg_rec["memory_signal"]["applied"] is True
    assert "ElasticNet" in reg_rec["memory_signal"]["reordered_models"]


def test_model_selector_keeps_original_order_when_meta_confidence_is_low(monkeypatch):
    def fake_zero_shot(profile, model_pool):
        return {
            "rankings": [
                {"model": "ElasticNet", "pred_score": 99.0},
                {"model": "Ridge", "pred_score": 75.0},
            ],
            "confidence": 12.0,
            "source": "Test Meta",
            "reason": "Too little history.",
        }

    monkeypatch.setattr(model_selector_module, "zero_shot_recommend", fake_zero_shot)

    reg_pool, reg_rec = ModelSelector.select_pool(
        rows=9000,
        is_clf=False,
        goal="Performance",
        profile={"rows": 9000, "cols": 18, "num_cols": list(range(18)), "cat_cols": []},
    )

    ordered = list(reg_pool.keys())
    assert ordered[0:3] == ["Linear Regression", "Ridge", "ElasticNet"]
    assert reg_rec["memory_signal"]["applied"] is False


def test_simple_model_early_stop_requires_low_variance_too():
    stable = {"name": "Logistic Regression", "f1": 92.1, "stability_std": 1.8}
    unstable = {"name": "Logistic Regression", "f1": 92.1, "stability_std": 5.4}

    assert _simple_model_is_good_enough(stable, is_classification=True) is True
    assert _simple_model_is_good_enough(unstable, is_classification=True) is False


def test_optuna_candidate_pruning_skips_redundant_boosters_and_heavy_models_after_strong_simple_win():
    top_candidates = [
        {
            "name": "Logistic Regression",
            "score": 0.93,
            "accuracy": 93.1,
            "f1": 92.8,
            "roc_auc": 94.0,
            "stability_std": 1.4,
        },
        {"name": "XGBoost", "score": 0.925},
        {"name": "LightGBM", "score": 0.921},
    ]
    execution_profile = {"top_k": 3}

    pruned, notes = _prune_optuna_candidates(
        top_candidates,
        sweep_results=top_candidates,
        execution_profile=execution_profile,
        is_classification=True,
    )

    assert [row["name"] for row in pruned] == ["Logistic Regression"]
    assert any("EarlyStop" in note for note in notes)


def test_optuna_candidate_pruning_keeps_only_best_advanced_booster():
    sweep_results = [
        {"name": "Random Forest", "score": 0.84, "accuracy": 84.0, "f1": 83.7},
        {"name": "XGBoost", "score": 0.88},
        {"name": "LightGBM", "score": 0.865},
        {"name": "Hist Gradient Boosting", "score": 0.86},
    ]
    top_candidates = list(sweep_results[:3])
    execution_profile = {"top_k": 3}

    pruned, notes = _prune_optuna_candidates(
        top_candidates,
        sweep_results=sweep_results,
        execution_profile=execution_profile,
        is_classification=True,
    )

    assert "XGBoost" in [row["name"] for row in pruned]
    assert "LightGBM" not in [row["name"] for row in pruned]
    assert len(pruned) == 3
    assert any("Hierarchy" in note for note in notes)


def test_diversity_guard_prunes_near_duplicate_candidates():
    sweep_results = [
        {"name": "Random Forest", "score": 0.871, "cv_scores": [87.1, 87.3, 87.0]},
        {"name": "Extra Trees", "score": 0.869, "cv_scores": [87.0, 87.2, 86.9]},
        {"name": "Hist Gradient Boosting", "score": 0.861, "cv_scores": [86.1, 86.2, 86.0]},
    ]

    pruned, notes = _prune_correlated_candidates(
        top_candidates=sweep_results,
        sweep_results=sweep_results,
        execution_profile={"top_k": 3},
    )

    assert "Random Forest" in [row["name"] for row in pruned]
    assert "Extra Trees" not in [row["name"] for row in pruned]
    assert any("DiversityGuard" in note for note in notes)


def test_tuning_budget_allocator_prefers_heavier_boosters():
    rf_budget = ModelSelector.get_tuning_budget("Random Forest", 32, 360, {"low_complexity": False})
    xgb_budget = ModelSelector.get_tuning_budget("XGBoost", 32, 360, {"low_complexity": False})

    assert rf_budget["trials"] < xgb_budget["trials"]
    assert rf_budget["timeout"] < xgb_budget["timeout"]


def test_fallback_model_prefers_baseline_family():
    name, model = _select_fallback_model(
        {
            "Random Forest": LinearRegression(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
        }
    )

    assert name == "Logistic Regression"
    assert isinstance(model, LogisticRegression)


def test_export_documents_real_pipeline_source_and_steps():
    manifest = _source_manifest_content()
    steps = json.loads(
        _pipeline_steps_content(
            {
                "target": "price",
                "feature_names": ["area"],
                "sanitizer_report": {"numeric_coercions": ["area"]},
                "performance_metrics": {
                    "optimized_metric": {"requested": "RMSE"}
                },
            },
            {
                "task_type": "regression",
                "preprocessor": "lite_column_transformer",
                "pca_applied": False,
            },
        )
    )

    assert "services/training/inference.py" in manifest
    assert steps["artifact_type"] == "end_to_end_raw_tabular_pipeline"
    assert steps["preprocessing"]["sanitizer"] == "services.data_sanitizer.sanitize_dataframe"
    assert steps["cleaning_policy_observed"]["numeric_coercions"] == ["area"]


def test_explain_script_content_renders_literal_rows_dict():
    content = _explain_script_content(["age", "income"])

    assert 'FEATURE_NAMES = ["age", "income"]' in content
    assert '{"feature": str(name), "importance": round(float(val), 6)}' in content


def test_export_bundle_filename_reflects_launch_origin():
    assert (
        build_export_bundle_filename(
            "job-drift-card",
            {"launch_source": "drift_recommendation"},
        )
        == "drift_reopen_automl_export_job-drif.zip"
    )
    assert (
        build_export_bundle_filename(
            "job-manual-card",
            {"launch_source": "manual"},
        )
        == "manual_automl_export_job-manu.zip"
    )

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from services.training.preprocessing import (
    auto_clean_data,
    fuzzy_merge_labels,
    make_lite_preprocessor,
    make_preprocessor,
)
from services.drift_service import get_drift_dashboard
from services.training.forecasting import estimate_training_forecast
from services.training.evaluator import _resolve_scoring, stability_check
from services.training.components import _coerce_estimator_instance, _resolve_final_model_choice
from services.training.components import _resolve_target_column_name
from services.explain_service import generate_counterfactual
from infra.database import DatasetModel, JobModel
from infra.storage import get_model_path
from core.file_loader import load_dataframe
from .conftest import TestingSessionLocal


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

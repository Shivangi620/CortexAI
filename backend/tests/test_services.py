import pandas as pd
import numpy as np
from services.training.preprocessing import auto_clean_data, fuzzy_merge_labels
from services.drift_service import get_drift_dashboard
from services.training.forecasting import estimate_training_forecast
from core.file_loader import load_dataframe


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

import pandas as pd
import numpy as np
from services.training.preprocessing import auto_clean_data, fuzzy_merge_labels


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

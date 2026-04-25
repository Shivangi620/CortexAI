import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_TYPES = (LGBMClassifier, LGBMRegressor)
except Exception:
    LGBM_TYPES = tuple()


CLASSIFICATION_HINTS = {
    "class",
    "label",
    "status",
    "outcome",
    "segment",
    "category",
    "grade",
    "tier",
    "flag",
    "fraud",
    "churn",
    "default",
    "approved",
    "rejected",
}

CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1-score",
    "f1 score",
    "roc_auc",
    "roc-auc",
    "roc auc",
    "auc",
}
REGRESSION_METRICS = {"r2", "r²", "r2 score", "mae", "mean absolute error", "mse", "mean squared error", "rmse", "root mean squared error"}


def _default_metric_for_task(task_type: str) -> str:
    task_norm = str(task_type or "").strip().lower()
    if task_norm == "regression":
        return "RMSE"
    return "F1-score"


def detect_task_type(y, target_name: str = "", override: str = ""):
    override_norm = str(override or "").strip().lower()
    if override_norm in {"classification", "regression"}:
        return {
            "task_type": override_norm,
            "source": "user_override",
            "reason": f"Task forced to {override_norm} by request.",
        }

    series = pd.Series(y).dropna()
    if series.empty:
        return {
            "task_type": "classification",
            "source": "fallback",
            "reason": "Target is empty after dropping nulls; defaulting to classification.",
        }

    lowered_name = str(target_name or "").strip().lower().replace(" ", "_")
    hinted = any(token in lowered_name for token in CLASSIFICATION_HINTS)

    if (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return {
            "task_type": "classification",
            "source": "semantic_dtype",
            "reason": "Target is categorical/string-like, so classification was selected.",
        }

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return {
            "task_type": "classification",
            "source": "fallback",
            "reason": "Target could not be interpreted numerically; using classification.",
        }

    unique_count = int(numeric.nunique(dropna=True))
    unique_ratio = float(unique_count / max(len(numeric), 1))
    fractional_mass = float(np.mean(np.abs(numeric - np.round(numeric)) > 1e-9))

    if fractional_mass > 0.05:
        return {
            "task_type": "regression",
            "source": "continuous_target",
            "reason": "Target contains genuine fractional values, which strongly suggests regression.",
        }

    small_cardinality = unique_count <= max(5, min(20, int(len(numeric) * 0.02) or 5))
    if unique_count <= 2:
        return {
            "task_type": "classification",
            "source": "binary_numeric_target",
            "reason": "Target has only two numeric states, so classification was selected.",
        }
    if hinted and unique_count <= 50:
        return {
            "task_type": "classification",
            "source": "target_semantics",
            "reason": "Target name suggests discrete labels and cardinality is low enough for classification.",
        }
    if small_cardinality and unique_ratio <= 0.02:
        return {
            "task_type": "classification",
            "source": "cardinality",
            "reason": "Target has extremely low cardinality relative to dataset size, so classification was selected.",
        }

    return {
        "task_type": "regression",
        "source": "numeric_default",
        "reason": "Target behaves like a numeric continuous outcome, so regression was selected.",
    }

def _resolve_scoring(eval_metric: str, is_clf: bool) -> str:
    metric = (eval_metric or "").strip().lower()
    if is_clf:
        if metric in {"f1", "f1-score", "f1 score"}:
            return "f1_weighted"
        if metric in {"precision"}:
            return "precision_weighted"
        if metric in {"recall"}:
            return "recall_weighted"
        if metric in {"roc_auc", "roc-auc", "roc auc", "auc"}:
            return "roc_auc_ovr_weighted"
        return "accuracy"
    if metric in {"mae", "mean absolute error"}:
        return "neg_mean_absolute_error"
    if metric in {"mse", "mean squared error"}:
        return "neg_mean_squared_error"
    if metric in {"rmse", "root mean squared error"}:
        return "neg_root_mean_squared_error"
    return "r2"


def normalize_training_controls(
    task_type: str,
    goal: str,
    mode: str,
    eval_metric: str = "",
    handle_imbalance: bool = False,
) -> dict:
    task_norm = str(task_type or "classification").strip().lower()
    if task_norm not in {"classification", "regression"}:
        task_norm = "classification"

    goal_lookup = {
        "speed": "Speed",
        "balanced": "Balanced",
        "performance": "Performance",
        "full": "Performance",
    }
    mode_lookup = {
        "fast": "Fast",
        "balanced": "Balanced",
        "full": "Full",
        "performance": "Full",
        "speed": "Fast",
    }

    resolved_goal = goal_lookup.get(str(goal or "").strip().lower(), "Balanced")
    resolved_mode = mode_lookup.get(str(mode or "").strip().lower(), "Balanced")

    metric_raw = str(eval_metric or "").strip()
    metric_norm = metric_raw.lower()
    warnings = []

    if task_norm == "classification":
        if metric_norm and metric_norm in REGRESSION_METRICS:
            replacement = _default_metric_for_task(task_norm)
            warnings.append(
                f"Requested metric '{metric_raw}' is regression-only; switching to {replacement} for classification."
            )
            metric_raw = replacement
        elif not metric_raw:
            metric_raw = _default_metric_for_task(task_norm)
    else:
        if metric_norm and metric_norm in CLASSIFICATION_METRICS:
            warnings.append(
                f"Requested metric '{metric_raw}' is classification-only; switching to RMSE for regression."
            )
            metric_raw = "RMSE"
        elif not metric_raw:
            metric_raw = "RMSE"

    resolved_handle_imbalance = bool(handle_imbalance and task_norm == "classification")
    if handle_imbalance and task_norm != "classification":
        warnings.append(
            "Imbalance handling is only applied to classification tasks; disabling it for regression."
        )

    return {
        "task_type": task_norm,
        "goal": resolved_goal,
        "mode": resolved_mode,
        "eval_metric": metric_raw,
        "handle_imbalance": resolved_handle_imbalance,
        "warnings": warnings,
    }


def stability_check(model, X, y, is_clf, seeds=(42, 123, 999), scoring_name: str = ""):
    """Re-fit with multiple seeds; returns mean primary metric + optional detail metrics."""
    scores = []
    accs, precs, recs, f1s, roc_aucs = [], [], [], [], []
    r2s, mses, rmses, maes = [], [], [], []
    y_series = pd.Series(y)
    unique_classes = int(y_series.nunique()) if is_clf else 0
    for seed in seeds:
        if hasattr(model, 'random_state'):
            try:
                model.set_params(random_state=seed)
            except Exception:
                pass
        test_size = max(1, int(round(len(y_series) * 0.2)))
        if len(y_series) <= 3:
            test_size = 1
        split_kwargs = {"test_size": test_size, "random_state": seed}
        if is_clf and unique_classes > 1 and test_size >= unique_classes:
            min_class_count = int(y_series.value_counts().min())
            if min_class_count >= 2:
                split_kwargs["stratify"] = y
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, **split_kwargs)
        except Exception:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)
        if LGBM_TYPES and isinstance(model, LGBM_TYPES):
            X_tr_in = pd.DataFrame(X_tr)
            X_val_in = pd.DataFrame(X_val, columns=X_tr_in.columns)
        else:
            X_tr_in, X_val_in = X_tr, X_val
        model.fit(X_tr_in, y_tr)
        pred = model.predict(X_val_in)
        if is_clf:
            accuracy = accuracy_score(y_val, pred)
            precision = precision_score(
                y_val, pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_val, pred, average="weighted", zero_division=0)
            f1 = f1_score(y_val, pred, average="weighted", zero_division=0)
            roc_auc = None
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val_in)
                    if probs.shape[1] == 2:
                        roc_auc = roc_auc_score(y_val, probs[:, 1])
                    else:
                        roc_auc = roc_auc_score(
                            y_val, probs, multi_class="ovr", average="weighted"
                        )
            except Exception:
                roc_auc = None

            accs.append(float(accuracy))
            precs.append(float(precision))
            recs.append(float(recall))
            f1s.append(float(f1))
            if roc_auc is not None:
                roc_aucs.append(float(roc_auc))

            if scoring_name == "f1_weighted":
                scores.append(float(f1))
            elif scoring_name == "precision_weighted":
                scores.append(float(precision))
            elif scoring_name == "recall_weighted":
                scores.append(float(recall))
            elif scoring_name == "roc_auc_ovr_weighted":
                scores.append(float(roc_auc if roc_auc is not None else accuracy))
            else:
                scores.append(float(accuracy))
        else:
            r2 = r2_score(y_val, pred)
            mse = mean_squared_error(y_val, pred)
            mae = mean_absolute_error(y_val, pred)
            rmse = float(np.sqrt(mse))

            r2s.append(float(r2))
            mses.append(float(mse))
            rmses.append(float(rmse))
            maes.append(float(mae))

            if scoring_name == "neg_mean_absolute_error":
                scores.append(-float(mae))
            elif scoring_name == "neg_mean_squared_error":
                scores.append(-float(mse))
            elif scoring_name == "neg_root_mean_squared_error":
                scores.append(-float(rmse))
            else:
                scores.append(float(r2))
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    extras = {}
    if is_clf and accs:
        extras = {
            "accuracy": round(float(np.mean(accs)) * 100, 1),
            "precision": round(float(np.mean(precs)) * 100, 1),
            "recall": round(float(np.mean(recs)) * 100, 1),
            "f1": round(float(np.mean(f1s)) * 100, 1),
        }
        if roc_aucs:
            extras["roc_auc"] = round(float(np.mean(roc_aucs)) * 100, 1)
    elif not is_clf and mses:
        extras = {
            "r2": round(float(np.mean(r2s)), 6),
            "mse": round(float(np.mean(mses)), 6),
            "rmse": round(float(np.mean(rmses)), 6),
            "mae": round(float(np.mean(maes)), 6),
        }
    return mean_score, std_score, extras

class DiagnosticAgent:
    """Agent that predicts dataset performance and risk before training."""
    def predict_risk(self, df: pd.DataFrame, target: str, health_metadata: dict = None):
        rows, cols = df.shape
        reasoning = []

        # Default high risk if metadata missing
        health_score = health_metadata.get("score", 50) if health_metadata else 50
        
        if rows < 500:
            risk = "🔴 High Overfitting Risk"
        elif rows / cols < 10:
            risk = "🟡 Medium Overfitting Risk"
        else:
            risk = "✅ Low Risk"

        # Dynamic accuracy estimate based on health score
        # Formula: Base 85% reduced by health score gap, clamped to 10-95%
        base_est = 88 - (100 - health_score) * 0.6
        est_min = max(10, int(base_est - 5))
        est_max = min(98, int(base_est + 5))
        
        acc_est = f"{est_min}-{est_max}%"
        
        difficulty = "Low" if health_score > 80 else "Medium" if health_score > 60 else "High"
        
        reasoning.append(f"Diagnostic: {risk} | Health Score: {health_score} | Estimated Difficulty: {difficulty}")
        return {"risk": risk, "estimate": acc_est, "difficulty": difficulty, "health_score": health_score}, reasoning

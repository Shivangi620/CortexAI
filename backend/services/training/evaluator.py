import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_TYPES = (LGBMClassifier, LGBMRegressor)
except Exception:
    LGBM_TYPES = tuple()

def _resolve_scoring(eval_metric: str, is_clf: bool) -> str:
    metric = (eval_metric or "").strip().lower()
    if is_clf:
        if metric in {"f1", "f1-score", "f1 score"}:
            return "f1_weighted"
        if metric in {"precision"}:
            return "precision_weighted"
        if metric in {"recall"}:
            return "recall_weighted"
        return "accuracy"
    if metric in {"mae", "mean absolute error"}:
        return "neg_mean_absolute_error"
    if metric in {"mse", "mean squared error"}:
        return "neg_mean_squared_error"
    if metric in {"rmse", "root mean squared error"}:
        return "neg_root_mean_squared_error"
    return "r2"


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
            msg = "Small dataset size (<500 rows) suggests the model may memorize rather than learn."
        elif rows / cols < 10:
            risk = "🟡 Medium Overfitting Risk"
            msg = "High dimensionality (few rows per feature) may cause variance issues."
        else:
            risk = "✅ Low Risk"
            msg = "Solid data-to-feature ratio detected."

        # Dynamic accuracy estimate based on health score
        # Formula: Base 85% reduced by health score gap, clamped to 10-95%
        base_est = 88 - (100 - health_score) * 0.6
        est_min = max(10, int(base_est - 5))
        est_max = min(98, int(base_est + 5))
        
        acc_est = f"{est_min}-{est_max}%"
        
        difficulty = "Low" if health_score > 80 else "Medium" if health_score > 60 else "High"
        
        reasoning.append(f"Diagnostic: {risk} | Health Score: {health_score} | Estimated Difficulty: {difficulty}")
        return {"risk": risk, "estimate": acc_est, "difficulty": difficulty, "health_score": health_score}, reasoning

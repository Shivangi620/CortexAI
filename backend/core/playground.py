import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# ✅ FIX 1: optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

from services.training.preprocessing import make_preprocessor


AVAILABLE_MODELS = {
    "Logistic Regression": "classification",
    "Random Forest": "both",
    "Decision Tree": "both",
    "XGBoost": "both",
    "Linear Regression": "regression",
    "Ridge Regression": "regression",
    "Lasso Regression": "regression",
}


def quick_train(
    df: pd.DataFrame,
    target: str,
    selected_features: list,
    selected_model_names: list,
) -> dict:

    if target not in df.columns:
        return {"error": f"Target '{target}' not found"}

    # Default to the dataset's non-target columns when no explicit feature list is supplied.
    selected_features = selected_features or [column for column in df.columns if column != target]

    valid_features = [f for f in selected_features if f in df.columns and f != target]
    cols_to_use = valid_features + [target]
    df = df[cols_to_use].dropna(subset=[target])

    if df.empty:
        return {"error": "No data left after filtering"}

    y = df[target]
    X = df.drop(columns=[target])

    if X.empty:
        return {"error": "Quicktrain needs at least one feature column besides the target."}

    is_clf = (y.nunique() < 20) or (not pd.api.types.is_numeric_dtype(y))

    # ✅ FIX 3: preserve index
    le = None
    if is_clf:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

    X = X.copy()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    datetime_cols = X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    for col in bool_cols:
        X[col] = X[col].astype(int)
    for col in datetime_cols:
        X[col] = X[col].astype(str)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        X[col] = X[col].astype(str)

    usable_cols = num_cols + cat_cols
    if not usable_cols:
        return {"error": "Quicktrain could not find usable numeric, categorical, boolean, or datetime feature columns."}

    X = X[usable_cols]

    if len(X) < 10:
        return {"error": "Not enough rows after filtering (need at least 10)."}

    # ✅ FIX 5: safe split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if is_clf and y.nunique() > 1 else None
        )
    except Exception:
        return {"error": "Train/test split failed (check class distribution)"}

    model_registry = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42),
        "Random Forest": (
            RandomForestClassifier(n_estimators=30, random_state=42)
            if is_clf else RandomForestRegressor(n_estimators=30, random_state=42)
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=8, random_state=42)
            if is_clf else DecisionTreeRegressor(max_depth=8, random_state=42)
        ),
    }

    # ✅ FIX 6: add XGBoost only if available
    if XGBClassifier and XGBRegressor:
        model_registry["XGBoost"] = (
            XGBClassifier(n_estimators=30, eval_metric="logloss", random_state=42)
            if is_clf else XGBRegressor(n_estimators=30, random_state=42)
        )

    leaderboard = []
    detailed_results = []
    best_pipeline = None

    for name in selected_model_names:
        if name not in model_registry:
            continue

        # skip incompatible models
        if is_clf and name == "Linear Regression":
            continue
        if not is_clf and name == "Logistic Regression":
            continue

        model = model_registry[name]

        try:
            preprocessor = make_preprocessor(num_cols, cat_cols)
        except ValueError as e:
            return {"error": str(e)}

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            if is_clf:
                score = round(accuracy_score(y_test, preds) * 100, 1)
                detail = {
                    "model": name,
                    "score": score,
                    "accuracy": score,
                    "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)) * 100, 1),
                    "recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)) * 100, 1),
                    "f1": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)) * 100, 1),
                    "test_rows": int(len(X_test)),
                    "train_rows": int(len(X_train)),
                }
                labels = sorted(pd.Series(y_test).dropna().unique().tolist())
                try:
                    matrix = confusion_matrix(y_test, preds, labels=labels)
                    detail["confusion_matrix"] = matrix.tolist()
                    detail["class_labels"] = (
                        [str(le.inverse_transform([int(label)])[0]) for label in labels]
                        if le is not None else [str(label) for label in labels]
                    )
                except Exception:
                    pass
            else:
                score = round(r2_score(y_test, preds) * 100, 1)
                detail = {
                    "model": name,
                    "score": score,
                    "r2": score,
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
                    "mae": round(float(mean_absolute_error(y_test, preds)), 4),
                    "test_rows": int(len(X_test)),
                    "train_rows": int(len(X_train)),
                }

            sample_df = X_test.head(5).copy()
            sample_preds = pipe.predict(sample_df)
            detail["sample_predictions"] = [
                {
                    "row_index": int(idx),
                    "actual": (
                        str(le.inverse_transform([int(y_test.loc[idx])])[0])
                        if is_clf and le is not None else float(y_test.loc[idx]) if hasattr(y_test.loc[idx], "item") else y_test.loc[idx]
                    ),
                    "predicted": (
                        str(le.inverse_transform([int(pred)])[0])
                        if is_clf and le is not None else float(pred) if hasattr(pred, "item") else pred
                    ),
                }
                for idx, pred in zip(sample_df.index.tolist(), sample_preds)
            ]

            if hasattr(pipe, "predict_proba"):
                try:
                    proba = pipe.predict_proba(sample_df)
                    for row, p_row in zip(detail["sample_predictions"], proba):
                        row["confidence_pct"] = round(float(max(p_row)) * 100, 1)
                except Exception:
                    pass

            leaderboard.append({"model": name, "score": score})
            detailed_results.append(detail)

            if best_pipeline is None or score > best_pipeline["score"]:
                best_pipeline = {"model": name, "score": score, "pipeline": pipe}

        except Exception as e:
            leaderboard.append({"model": name, "score": None, "error": str(e)})
            detailed_results.append({"model": name, "score": None, "error": str(e)})

    if not leaderboard:
        return {"error": "No valid models ran. Check feature/model selection."}

    # ✅ FIX 7: correct sorting
    def safe_score(x):
        try:
            return float(x.get("score", -999))
        except Exception:
            return -999
    
    leaderboard.sort(key=safe_score, reverse=True)

    best = leaderboard[0]
    best_detail = next((row for row in detailed_results if row.get("model") == best["model"]), {})

    feature_importance = []
    if best_pipeline:
        try:
            model = best_pipeline["pipeline"].named_steps["model"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = model.coef_[0] if np.ndim(model.coef_) > 1 else model.coef_
                importances = np.abs(coef)
            else:
                importances = None

            if importances is not None:
                transformed_names = best_pipeline["pipeline"].named_steps["preprocessor"].get_feature_names_out()
                feature_importance = [
                    {"feature": str(name), "importance": round(float(val), 6)}
                    for name, val in sorted(
                        zip(transformed_names, importances),
                        key=lambda x: abs(float(x[1])),
                        reverse=True,
                    )[:15]
                ]
        except Exception:
            feature_importance = []

    return {
        "is_classification": is_clf,
        "metric_name": "Accuracy" if is_clf else "R²",
        "leaderboard": leaderboard,
        "model_details": detailed_results,
        "best_model": best["model"],
        "score": best["score"],
        "best_detail": best_detail,
        "feature_importance": feature_importance,
        "n_features": len(selected_features),
        "n_rows": len(df),
        "selected_features": selected_features,
        "target": target,
    }

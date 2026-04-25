import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional, Any

try:
    import lightgbm as lgb
except Exception:
    lgb = None
from infra.database import db_session, MetaLearningRecord

# ── Meta-Feature Extraction ───────────────────────────────────────────────────


def extract_meta_features(profile: dict) -> dict:
    """Compute an enhanced meta-feature vector from a dataset profile."""
    n_rows = profile.get("rows", 0)
    n_cols = profile.get("cols", 1)  # avoid div by zero inside
    if n_cols == 0:
        n_cols = 1

    col_stats = profile.get("column_stats", {})

    # Extract structural ratios
    n_num = len(profile.get("num_cols") or [])
    n_cat = len(profile.get("cat_cols") or [])

    # Extract semantic ratios
    c_binary = c_id = c_datetime = c_continuous = c_discrete = c_nominal = 0
    for col, stat in col_stats.items():
        stype = stat.get("semantic_type", "")
        if stype == "Binary":
            c_binary += 1
        elif stype == "ID/Index":
            c_id += 1
        elif stype == "DateTime":
            c_datetime += 1
        elif stype == "Continuous":
            c_continuous += 1
        elif stype == "Discrete/Ordinal":
            c_discrete += 1
        elif stype == "Nominal Category":
            c_nominal += 1

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "num_ratio": round(n_num / n_cols, 4),
        "cat_ratio": round(n_cat / n_cols, 4),
        "binary_ratio": round(c_binary / n_cols, 4),
        "datetime_ratio": round(c_datetime / n_cols, 4),
        "continuous_ratio": round(c_continuous / n_cols, 4),
        "missing_pct": profile.get("missing_pct", 0),
        "is_imbalanced": 1 if "High" in str(profile.get("imbalance", "")) else 0,
    }


class MetaLearner:
    def __init__(self):
        self.model = None
        if lgb is not None:
            try:
                self.model = lgb.LGBMRegressor(
                    n_estimators=100, learning_rate=0.1, random_state=42
                )
            except Exception:
                self.model = None
        self.is_trained = False
        self.min_records = 10
        self.val_error = 1.0  # High error initially
        self.feature_columns: List[str] = []

    def prepare_data(self, records):
        data = []

        for r in records:
            try:
                mf = json.loads(r.get("meta_features_json") or "{}")
                leaderboard = json.loads(r.get("leaderboard_json") or "[]")
            except Exception:
                continue

            for entry in leaderboard:
                try:
                    if entry.get("phase") not in {None, "cross_validation"}:
                        continue
                    row = mf.copy()
                    row["model_type"] = entry.get("model")
                    row["score"] = entry.get("score", 0)
                    row["task_type"] = entry.get("task_type") or r.get("task_type")
                    row["metric_name"] = entry.get("metric_name") or r.get("metric_name")
                    data.append(row)
                except Exception:
                    continue

        if not data:
            return pd.DataFrame(), None

        df = pd.DataFrame(data)
        X = self._encode_features(df.drop(columns=["score"], errors="ignore"), fit=True)
        return X, df.get("score")

    def _encode_features(
        self,
        frame: pd.DataFrame,
        fit: bool = True,
        expected_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame

        encoded = frame.copy()
        for col in encoded.columns:
            if pd.api.types.is_bool_dtype(encoded[col]):
                encoded[col] = encoded[col].astype(int)

        categorical_cols = encoded.select_dtypes(include=["object", "string", "category"]).columns
        if len(categorical_cols) > 0:
            encoded = pd.get_dummies(
                encoded,
                columns=list(categorical_cols),
                dummy_na=True,
                dtype=int,
            )

        encoded = encoded.fillna(0)

        if fit:
            self.feature_columns = list(encoded.columns)
            return encoded

        columns = expected_columns if expected_columns is not None else self.feature_columns
        if columns:
            encoded = encoded.reindex(columns=columns, fill_value=0)
        return encoded

    def train(self):
        with db_session() as db:
            raw_records = db.query(MetaLearningRecord).all()

            records = [
                {
                    "meta_features_json": r.meta_features_json,
                    "leaderboard_json": r.leaderboard_json,
                    "task_type": r.task_type,
                    "metric_name": r.metric_name,
                }
                for r in raw_records
            ]

        if len(records) < self.min_records:
            return
        if self.model is None:
            return

        X, y = self.prepare_data(records)
        if X.empty:
            return

        self.model.fit(X, y)
        self.is_trained = True
        # Simple heuristic for validation error: use 10% of training data as pseudo-val
        preds = self.model.predict(X)
        self.val_error = np.mean(np.abs(preds - y)) / 100.0  # Normalized error

    def predict_rankings(self, profile: dict, model_pool: List[str]) -> Dict:
        """Predicts expected scores for each model and returns ranked list."""
        if not self.is_trained:
            return self.get_heuristics(profile, model_pool)

        mf = extract_meta_features(profile)
        pred_data = [mf.copy() for _ in model_pool]
        for i, m in enumerate(model_pool):
            pred_data[i]["model_type"] = m

        X_pred = self._encode_features(
            pd.DataFrame(pred_data),
            fit=False,
            expected_columns=self.feature_columns,
        )

        preds = self.model.predict(X_pred)

        # Confidence logic: 1.0 - val_error
        confidence = max(0, min(1, 1.0 - self.val_error))

        # Rankings
        results = []
        for m, score in zip(model_pool, preds):
            results.append({"model": m, "pred_score": round(float(score), 2)})

        results.sort(key=lambda x: x["pred_score"], reverse=True)

        # Switch to heuristics if confidence is too low
        if confidence < 0.6:
            return self.get_heuristics(
                profile, model_pool, confidence, "Low Meta-Confidence"
            )

        return {
            "rankings": results,
            "confidence": round(confidence * 100, 1),
            "source": "LightGBM Meta-Learner",
            "reason": f"Meta-learner confident ({round(confidence*100)}%) based on {X_pred.shape[0]} historical trials.",
        }

    def get_heuristics(
        self, profile: dict, model_pool: List[str], confidence=0, reason="Cold Start"
    ) -> Dict:
        """Fallback heuristics (the existing rules)."""
        rows = profile.get("rows", 0)
        # Simple ordering based on rows
        if rows > 10000:
            pivot = ["XGBoost", "LightGBM", "Random Forest"]
        elif rows < 500:
            pivot = ["Logistic Regression", "Ridge", "Random Forest"]
        else:
            pivot = ["Random Forest", "XGBoost", "SVM"]

        # Sort pool based on pivot
        rankings = []
        for m in model_pool:
            score = 70.0  # Default base
            if m in pivot:
                score += (len(pivot) - pivot.index(m)) * 5
            rankings.append({"model": m, "pred_score": score})

        rankings.sort(key=lambda x: x["pred_score"], reverse=True)

        return {
            "rankings": rankings,
            "confidence": confidence,
            "source": "Rule-based Heuristics",
            "reason": f"Initial guess. {reason}",
        }


# Singleton instance for the system
meta_engine = MetaLearner()


def _default_model_pool(profile: dict) -> List[str]:
    """Pick a reasonable candidate pool when the API does not specify models."""
    cols = profile.get("columns") or []
    n = max(len(cols), 1)
    cat_ratio = len(profile.get("cat_cols") or []) / n
    rows = profile.get("rows") or 0
    if cat_ratio >= 0.25 or rows < 5000:
        pool = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]
        if rows < 5000:
            pool.append("SVM")
        return pool
    return [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
    ]


def get_cross_dataset_insights(profile: dict) -> Dict[str, Any]:
    """
    Summarize historical training runs (meta_learning) and compare to the current profile.
    """
    mf = extract_meta_features(profile)
    rows_cur = mf["n_rows"]
    cols_cur = mf["n_cols"]

    with db_session() as db:
        records = (
            db.query(MetaLearningRecord)
            .order_by(MetaLearningRecord.created_at.desc())
            .limit(500)
            .all()
        )

    if not records:
        return {
            "historical_runs": 0,
            "message": "No historical training runs yet. Finish at least one job to unlock cross-dataset insights.",
            "your_dataset": {"rows": rows_cur, "columns": cols_cur},
        }

    model_counts = Counter(r.best_model for r in records if r.best_model)
    task_counts = Counter(r.task_type for r in records if r.task_type)
    top_model, top_model_n = model_counts.most_common(1)[0]

    rows_hist = []
    cols_hist = []
    for r in records:
        try:
            fj = json.loads(r.meta_features_json)
            rows_hist.append(int(fj.get("n_rows", 0)))
            cols_hist.append(int(fj.get("n_cols", 0)))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    def _band(val: int, values: List[int]) -> str:
        if not values:
            return "unknown"
        lo, hi = min(values), max(values)
        if val < lo:
            return "smaller than most past runs"
        if val > hi:
            return "larger than most past runs"
        return "within the range seen in past runs"

    return {
        "historical_runs": len(records),
        "unique_jobs_recorded": len({r.id for r in records}),
        "task_mix": dict(task_counts),
        "most_common_winner": {"model": top_model, "count": top_model_n},
        "your_dataset": {
            "rows": rows_cur,
            "columns": cols_cur,
            "size_vs_history": {
                "rows": _band(rows_cur, rows_hist),
                "columns": _band(cols_cur, cols_hist),
            },
        },
        "hint": "Winners reflect your machine's past AutoML jobs on similar-sized data; they are hints, not guarantees.",
    }


def zero_shot_recommend(profile: dict, model_pool: Optional[List[str]] = None) -> dict:
    """Entry point for the recommendation engine."""
    pool = model_pool if model_pool else _default_model_pool(profile)
    if not meta_engine.is_trained:
        try:
            meta_engine.train()
        except Exception:
            meta_engine.is_trained = False
            return meta_engine.get_heuristics(
                profile, pool, confidence=0, reason="Meta training unavailable"
            )
    return meta_engine.predict_rankings(profile, pool)


def save_meta_record(profile: dict, results: dict):
    """Persist a meta-learning record."""
    mf = extract_meta_features(profile)
    task_type = "classification" if results.get("is_classification") else "regression"
    metric_name = results.get("metric_name", "Score")
    normalized_leaderboard = []
    for entry in results.get("leaderboard", []) or []:
        if not isinstance(entry, dict):
            continue
        normalized_leaderboard.append(
            {
                "model": entry.get("model"),
                "score": entry.get("score", 0),
                "phase": entry.get("phase"),
                "metric_name": metric_name,
                "task_type": task_type,
            }
        )
    with db_session() as db:
        record = MetaLearningRecord(
            meta_features_json=json.dumps(mf),
            best_model=results.get("best_model", ""),
            best_score=results.get("score", 0),
            task_type=task_type,
            metric_name=metric_name,
            leaderboard_json=json.dumps(normalized_leaderboard),
        )
        db.add(record)
        db.commit()
    # Trigger retrain
    try:
        meta_engine.train()
    except Exception:
        pass

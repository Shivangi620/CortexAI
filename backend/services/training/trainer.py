import pandas as pd
import numpy as np
import os
import joblib
import json
import shap
import optuna
import time
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from infra.database import get_db, JobModel
from core.integrations import MLTracking
from core.feature_engine import ManagedFeatureEngine
from core.meta_learning import zero_shot_recommend
from infra.storage import get_model_path, get_run_dir
from services.training.preprocessing import make_lite_preprocessor, DataAgent
from services.training.evaluator import _resolve_scoring, stability_check

def _update_history_db(job_id: str, history: list):
    """Write history snapshot to the DB using its own isolated session."""
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if job:
            job.history_json = json.dumps(history)
            db.commit()


def get_cheap_config(model_name, is_clf):
    """Returns 'Stage 1' parameters for rapid family evaluation."""
    if "Forest" in model_name:
        return {"n_estimators": 20, "max_depth": 5}
    if (
        "Boosting" in model_name
        or "XGB" in model_name
        or "LGBM" in model_name
        or "LightGBM" in model_name
    ):
        # Check for model family to avoid parameter errors
        return {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1}
    if "SVM" in model_name:
        return {"C": 0.1, "max_iter": 500}
    return {}

class ModelAgent:
    """Agent responsible for model selection and reasoning."""
    def select_pool(self, rows, is_clf, goal, mode, profile):
        """Uses meta-learning to rank and select the model pool."""
        pool = {
            "Logistic Regression" if is_clf else "Linear Regression": 
                LogisticRegression(max_iter=1000) if is_clf else LinearRegression(),
            "Random Forest": RandomForestClassifier() if is_clf else RandomForestRegressor(),
            "XGBoost": XGBClassifier(eval_metric='logloss') if is_clf else XGBRegressor(),
            "LightGBM": LGBMClassifier(verbose=-1) if is_clf else LGBMRegressor(verbose=-1),
        }
        
        # Regression specific
        if not is_clf:
            pool["Ridge"] = Ridge()
            pool["Lasso"] = Lasso()
        
        # Classification specific (speed/sample check)
        if is_clf and rows < 10000:
            pool["SVM"] = SVC(probability=True)

        pool = {k: v for k, v in pool.items() if v is not None}
        
        # Filter by Goal
        if goal == "Speed":
            speed_whitelist = ["Logistic Regression", "Linear Regression", "Ridge", "Lasso"]
            pool = {k: v for k, v in pool.items() if k in speed_whitelist}
            
        # Meta-Learning Ranking
        candidate_names = list(pool.keys())
        recommendation = zero_shot_recommend(profile, candidate_names)
        
        # Re-sort pool based on meta-learning scores if possible (prediction would be better, but ranking works)
        # For now, just return the pool and the recommendation reasoning.
        return pool, recommendation

def train_automl(
    df: pd.DataFrame, target: str, goal: str, mode: str, job_id: str,
    eval_metric: str = "",
    handle_imbalance: bool = False,
    auto_clean: bool = True,
    health_metadata: dict = None,
    cv_folds: int = 0
):
    full_reasoning = []
    start_time = time.time()
    health_metadata = health_metadata or {}
    eda_summary = {}

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    if auto_clean:
        da = DataAgent()
        df, clean_logs = da.clean(df, target)
        full_reasoning.extend(clean_logs)
    else:
        full_reasoning.append("DataCleaner: Auto-clean disabled by user.")
    
    # ── 1. Advanced Feature Engine (V3) ───────────────────────────────────────
    full_reasoning.append(f"Engine V3: Initiating Managed Feature Engine (Mode: {mode})")
    fe = ManagedFeatureEngine(target_col=target, task_type="classification" if health_metadata.get("task_type") == "classification" else "regression")
    
    # Leakage check
    leaks = fe.detect_leakage(df)
    if leaks:
        full_reasoning.append(f"LeakageGuard: Dropped suspicious columns: {leaks}")
        df = df.drop(columns=leaks)
        
    # Generate & Prune
    if mode in ["Balanced", "Full"]:
        df = fe.generate_features(df)
        full_reasoning.append(f"FeatureEngine: Generated {df.shape[1]} features (Budget enforced).")
    
    y_raw = df[target]
    X = df.drop(columns=[target])

    # TargetEncoder (and many estimators) require y without missing values.
    invalid_target = y_raw.isna()
    if y_raw.dtype == object or pd.api.types.is_string_dtype(y_raw):
        sr = y_raw.astype(str).str.strip().str.lower()
        invalid_target = invalid_target | sr.isin(
            ("nan", "none", "", "na", "n/a", "null", "?", "unknown", "??", "invalid")
        )
    dropped_target = int(invalid_target.sum())
    if dropped_target:
        full_reasoning.append(
            f"TargetCleaner: Removed {dropped_target} row(s) with missing or invalid target."
        )
    X = X.loc[~invalid_target].reset_index(drop=True)
    y = y_raw.loc[~invalid_target].reset_index(drop=True)
    if len(y) == 0:
        raise ValueError(
            "No rows left after removing missing or invalid target values. "
            "Clean the target column or choose another target."
        )

    # EDA summary exposed in job results for transparency.
    eda_summary = {
        "rows_after_target_cleaning": int(len(y)),
        "columns_after_feature_engineering": int(X.shape[1]),
        "target_missing_removed": dropped_target,
        "numeric_features": int(X.select_dtypes(include=[np.number]).shape[1]),
        "categorical_features": int(
            X.select_dtypes(include=["object", "category", "bool"]).shape[1]
        ),
    }

    # Task Detection
    is_clf = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20
    if is_clf:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Initial Split
    split_kwargs = {"test_size": 0.2, "random_state": 42}
    if is_clf:
        split_kwargs["stratify"] = y
    try:
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, **split_kwargs)
    except Exception:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # ── 2. Stage 1: Exploration Sweep ───────────────────────────────────────
    full_reasoning.append("🏁 Stage 1: Starting Exploration Sweep (Exploration Phase)")
    
    # Stratified sample for sweep (10-30%)
    sweep_size = 0.3 if len(X_train_full) < 5000 else 0.1
    X_sweep, _, y_sweep, _ = train_test_split(X_train_full, y_train_full, train_size=sweep_size, random_state=42)
    
    ma = ModelAgent()
    profile = {"rows": len(X), "cols": len(X.columns), "num_cols_count": len(num_cols), "cat_cols_count": len(cat_cols)}
    model_pool, meta_rec = ma.select_pool(len(X), is_clf, goal, mode, profile)
    full_reasoning.append(f"Meta-Learner: {meta_rec['reason']} (Source: {meta_rec['source']})")
    
    sweep_results = []
    pre_lite = make_lite_preprocessor(num_cols, cat_cols)
    X_sweep_proc = pre_lite.fit_transform(X_sweep, y_sweep)
    
    current_history = []
    for name, model in model_pool.items():
        try:
            # Apply cheap config for sweep
            model.set_params(**get_cheap_config(name, is_clf))
            model.fit(X_sweep_proc, y_sweep)
            
            # Fast eval on sweep sample
            score, _, metric_extras = stability_check(model, X_sweep_proc, y_sweep, is_clf)
            row = {"name": name, "score": score, "model": model}
            row.update(metric_extras)
            sweep_results.append(row)
            
            # Telemetry update
            metric_val = round(score * 100, 1)
            full_reasoning.append(f"Sweep: {name} scored {score:.3f}")
            current_history.append({"time": name, "metric": metric_val})
            _update_history_db(job_id, current_history)
            
        except Exception as e:
            full_reasoning.append(f"Sweep Check Failed for {name}: {e}")

    sweep_results.sort(key=lambda x: x['score'], reverse=True)
    top_candidates = sweep_results[:3]

    winner_pool_name: str | None = None

    if mode == "Fast":
        full_reasoning.append("Execution Gear: 'Fast' mode enabled. Finishing after Stage 1.")
        best_entry = top_candidates[0]
        final_model = best_entry['model']
        winner_pool_name = best_entry['name']
    else:
        # ── 3. Stage 2: Exploitation Deep Dive ──────────────────────────────
        full_reasoning.append(f"🚀 Stage 2: Starting Deep Dive optimization for: {[c['name'] for c in top_candidates]}")
        
        best_overall_score = -1
        final_model = None
        
        # Deep Dive Loop
        for candidate in top_candidates:
            name = candidate['name']
            full_reasoning.append(f"Deep Dive: Optimizing {name} with Bayesian Search...")
            
            def objective(trial):
                try:
                    model_params = {}
                    if "Forest" in name:
                        model_params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                            "max_depth": trial.suggest_int("max_depth", 5, 12)
                        }
                    elif "XGB" in name or "LGBM" in name or "LightGBM" in name:
                        model_params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                            "max_depth": trial.suggest_int("max_depth", 3, 8)
                        }
                    
                    m = model_pool[name].__class__(**model_params)
                    if isinstance(m, (LGBMClassifier, LGBMRegressor)):
                        try:
                            m.set_params(verbose=-1)
                        except Exception:
                            pass
                    n_splits = max(2, int(cv_folds)) if cv_folds else 3
                    cv = (
                        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        if is_clf else KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    )
                    
                    pipe = Pipeline([('pre', pre_lite), ('m', m)])
                    scores = cross_val_score(
                        pipe,
                        X_train_full,
                        y_train_full,
                        cv=cv,
                        scoring=_resolve_scoring(eval_metric, is_clf),
                    )
                    return scores.mean()
                except Exception:
                    return 0

            n_trials = 15 if mode == "Balanced" else 30
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, timeout=240)
            
            if study.best_value > best_overall_score:
                best_overall_score = study.best_value
                final_model = model_pool[name].__class__(**study.best_params)
                if isinstance(final_model, (LGBMClassifier, LGBMRegressor)):
                    try:
                        final_model.set_params(verbose=-1)
                    except Exception:
                        pass
                winner_pool_name = name
                
                # Telemetry update
                metric_val = round(study.best_value * 100, 1)
                current_history.append({"time": f"DeepDive:{name}", "metric": metric_val})
                _update_history_db(job_id, current_history)

        # Fallback if Deep Dive failed to find a model
        if final_model is None:
            final_model = top_candidates[0]['model']
            winner_pool_name = top_candidates[0]['name']

    if winner_pool_name is None:
        for k, v in model_pool.items():
            if v.__class__ is final_model.__class__:
                winner_pool_name = k
                break
        winner_pool_name = winner_pool_name or str(final_model).split("(")[0]

    # Final Telemetry
    current_history.append({"time": "Final", "metric": "🏁 Deployment"})
    _update_history_db(job_id, current_history)

    # ── 4. Final Training & Deployment ──────────────────────────────────────
    full_reasoning.append("🏁 Training final production pipe on full dataset...")
    final_pipe = Pipeline([('preprocessor', pre_lite), ('model', final_model)])
    final_pipe.fit(X_train_full, y_train_full)
    
    # ── 5. Data Drift Baseline ──────────────────────────────────────────────
    drift_baseline = X_train_full.describe().to_dict()
    
    # Metrics
    from sklearn.metrics import r2_score, accuracy_score  # reinforcing scope for worker
    preds = final_pipe.predict(X_test)
    score = accuracy_score(y_test, preds) if is_clf else r2_score(y_test, preds)
    
    # MLflow Logging
    MLTracking.log_run(
        job_id=job_id,
        params={"best_model": str(final_model), "mode": mode, "sweep_size": sweep_size},
        metrics={"test_score": score},
        model=final_pipe
    )

    # persistence
    model_path = get_model_path(job_id)
    get_run_dir(job_id) # ensures dir exists
    joblib.dump(final_pipe, model_path)

    # SHAP (Simplified for V3 refactor)
    shap_summary = {}
    try:
        X_test_proc = pre_lite.transform(X_test)
        explainer = shap.Explainer(final_model, X_test_proc)
        shap_vals = explainer(X_test_proc[:50])
        # Average importance
        importances = np.abs(shap_vals.values).mean(axis=0)
        if len(importances.shape) > 1: importances = importances.mean(axis=1) # multi-class
        
        feature_names = pre_lite.get_feature_names_out()
        for i, f in enumerate(feature_names[:8]):
            shap_summary[f.split("__")[-1]] = float(importances[i])
    except Exception as e:
        full_reasoning.append(f"Explainability: SHAP skipped ({e})")

    final_metric_scaled = round(score * 100, 1)
    # Leaderboard: row 1 = deployed model on hold-out test (same as `score`); following rows =
    # Stage 1 sweep scores (quick subsample — can disagree strongly with final test).
    leaderboard_rows = [
        {
            "model": winner_pool_name,
            "score": final_metric_scaled,
            "phase": "holdout_test",
        }
    ]
    for r in sweep_results:
        if r["name"] == winner_pool_name:
            continue
        row = {
            "model": r["name"],
            "score": round(r["score"] * 100, 1),
            "phase": "stage1_sweep",
        }
        for k in ("precision", "recall", "f1", "mse", "mae"):
            if k in r:
                row[k] = r[k]
        leaderboard_rows.append(row)
    tail = sorted(leaderboard_rows[1:], key=lambda x: x["score"], reverse=True)
    leaderboard_out = [leaderboard_rows[0]] + tail
    if is_clf:
        leaderboard_out[0]["precision"] = round(
            float(precision_score(y_test, preds, average="weighted", zero_division=0)) * 100,
            1,
        )
        leaderboard_out[0]["recall"] = round(
            float(recall_score(y_test, preds, average="weighted", zero_division=0)) * 100,
            1,
        )
        leaderboard_out[0]["f1"] = round(
            float(f1_score(y_test, preds, average="weighted", zero_division=0)) * 100,
            1,
        )
    else:
        leaderboard_out[0]["mse"] = round(float(mean_squared_error(y_test, preds)), 6)
        leaderboard_out[0]["mae"] = round(float(mean_absolute_error(y_test, preds)), 6)

    return {
        "best_model": winner_pool_name,
        "metric_name": "Accuracy" if is_clf else "R² Score",
        "score": final_metric_scaled,
        "leaderboard": leaderboard_out,
        "is_classification": is_clf,
        "shap_summary": shap_summary,
        "model_path": model_path,
        "feature_names": num_cols + cat_cols,
        "target": target,
        "eda_summary": eda_summary,
        "model_metadata": {
            "task_type": "classification" if is_clf else "regression",
            "eval_metric_requested": eval_metric or ("Accuracy" if is_clf else "R²"),
            "cv_folds_used": max(2, int(cv_folds)) if cv_folds else 3,
            "preprocessor": "lite_column_transformer_target_encoder",
            "feature_count": int(len(num_cols) + len(cat_cols)),
        },
        "reasoning": full_reasoning,
        "drift_baseline_json": json.dumps(drift_baseline)
    }

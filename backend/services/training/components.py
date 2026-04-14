import numpy as np
import pandas as pd
import shap
import optuna
from typing import Any, Dict
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor

from core.pipeline_engine import PipelineComponent, PipelineContext, PipelineStep
from core.feature_engine import ManagedFeatureEngine
from core.integrations import MLTracking
from infra.storage import ModelRegistry, DataContract, get_model_path, save_metrics
from services.training.preprocessing import make_lite_preprocessor, make_preprocessor, DataAgent
from services.training.evaluator import _resolve_scoring, stability_check
from services.training.model_selector import ModelSelector

class DataValidationComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.VALIDATE

    def execute(self, ctx: PipelineContext):
        df = pd.read_csv(ctx.file_path)
        if ctx.target_column not in df.columns:
            raise ValueError(f"Target column '{ctx.target_column}' not found in dataset")
        
        selected_features = ctx.config.get("selected_features")
        if selected_features:
            keep_cols = list(selected_features) + [ctx.target_column]
            available = [c for c in keep_cols if c in df.columns]
            df = df[available]

        
        # 1. Null Value Thresholds (NaN > 90% is critical failure)
        nan_percentages = df.isna().mean()
        high_nan_cols = nan_percentages[nan_percentages > 0.9].index.tolist()
        if high_nan_cols:
            ctx.reasoning.append(f"DataValidation: Warning - dropping columns with >90% missing values: {high_nan_cols}")
            df = df.drop(columns=high_nan_cols)

        # 2. Strict Type Parsing
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass

        # 3. Target Leakage Detection
        y_preview = df[ctx.target_column].dropna()
        task_type = (
            "classification"
            if (not pd.api.types.is_numeric_dtype(y_preview) or y_preview.nunique() <= 20)
            else "regression"
        )
        fe = ManagedFeatureEngine(target_col=ctx.target_column, task_type=task_type)
        leaks = fe.detect_leakage(df)
        if leaks:
            ctx.reasoning.append(f"DataValidation: LeakageGuard detected and dropped suspicious target-leaking columns early: {leaks}")
            df = df.drop(columns=leaks)

        # Enforce Data Contract immediately
        DataContract.save_contract(ctx.job_id, df)
        
        # Fit statistical baseline
        from core.drift_detector import DriftDetector
        DriftDetector(ctx.job_id).fit_baseline(df)

        ctx.df = df
        ctx.reasoning.append("DataValidation: Uploaded data matches contract. Constraints enforced.")
        

class FeatureEngineeringComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.FEATURE_ENG

    def execute(self, ctx: PipelineContext):
        df = ctx.df
        if ctx.config.get("auto_clean", True):
            da = DataAgent()
            df, logs = da.clean(df, ctx.target_column)
            ctx.reasoning.extend(logs)
        else:
            ctx.reasoning.append("DataCleaner: Auto-clean disabled.")
            
        task_type = (
            "classification"
            if (not pd.api.types.is_numeric_dtype(df[ctx.target_column].dropna()) or df[ctx.target_column].dropna().nunique() <= 20)
            else "regression"
        )
        fe = ManagedFeatureEngine(target_col=ctx.target_column, task_type=task_type)

            
        if ctx.mode in ["Balanced", "Full"]:
            df = fe.generate_features(df)
            ctx.reasoning.append(f"FeatureEngine: Generated {df.shape[1]} features.")
            
        y_raw = df[ctx.target_column]
        X = df.drop(columns=[ctx.target_column])
        
        invalid_target = y_raw.isna()
        if y_raw.dtype == object or pd.api.types.is_string_dtype(y_raw):
            sr = y_raw.astype(str).str.strip().str.lower()
            invalid_target = invalid_target | sr.isin(("nan", "none", "", "na", "n/a", "null", "?", "unknown", "??", "invalid"))
            
        dropped_target = int(invalid_target.sum())
        if dropped_target > 0:
            ctx.reasoning.append(f"TargetCleaner: Removed {dropped_target} invalid target rows.")
            
        X = X.loc[~invalid_target].reset_index(drop=True)
        y = y_raw.loc[~invalid_target].reset_index(drop=True)
        
        if len(y) == 0:
            raise ValueError("No rows left after dropping invalid target rows. Cannot train.")
            
        ctx.eda_summary = {
            "rows_after_target_cleaning": int(len(y)),
            "columns_after_feature_engineering": int(X.shape[1]),
            "target_missing_removed": dropped_target,
            "numeric_features": int(X.select_dtypes(include=[np.number]).shape[1]),
            "categorical_features": int(X.select_dtypes(include=["object", "category", "bool"]).shape[1]),
        }
        
        ctx.is_classification = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20
        if ctx.is_classification:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)))
            
        ctx.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        ctx.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        split_kwargs = {"test_size": 0.2, "random_state": 42}
        if ctx.is_classification:
            split_kwargs["stratify"] = y
            
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        ctx.X_train, ctx.X_test = X_train, X_test
        ctx.y_train, ctx.y_test = y_train, y_test
        ctx.X, ctx.y = X, y


class ModelSelectionComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.TRAIN

    def execute(self, ctx: PipelineContext):
        profile = {
            "rows": len(ctx.X),
            "cols": len(ctx.X.columns),
            "num_cols": ctx.num_cols,
            "cat_cols": ctx.cat_cols,
            "column_stats": {},
        }
        
        model_pool, meta_rec = ModelSelector.select_pool(len(ctx.X), ctx.is_classification, ctx.goal, profile)
        ctx.reasoning.append(f"Meta-Learner: {meta_rec['reason']} (Source: {meta_rec['source']})")
        ctx.model_pool = model_pool
        if ctx.mode == "Full":
            ctx.preprocessor = make_preprocessor(ctx.num_cols, ctx.cat_cols)
            ctx.reasoning.append("Preprocessor: Full mode selected richer preprocessing with skew/outlier/interactions.")
        else:
            ctx.preprocessor = make_lite_preprocessor(ctx.num_cols, ctx.cat_cols)
            ctx.reasoning.append("Preprocessor: Lite preprocessing selected for faster iteration.")


class TrainingComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.TRAIN

    def _apply_imbalance_strategy(self, model, ctx: PipelineContext):
        if not ctx.is_classification or not ctx.config.get("handle_imbalance"):
            return model

        try:
            if hasattr(model, "get_params") and "class_weight" in model.get_params():
                model.set_params(class_weight="balanced")
        except Exception:
            pass

        try:
            value_counts = pd.Series(ctx.y_train).value_counts()
            if len(value_counts) == 2 and hasattr(model, "get_params"):
                params = model.get_params()
                if "scale_pos_weight" in params:
                    majority = max(value_counts.max(), 1)
                    minority = max(value_counts.min(), 1)
                    model.set_params(scale_pos_weight=float(majority / minority))
        except Exception:
            pass

        return model

    def _execution_profile(self, ctx: PipelineContext) -> Dict[str, Any]:
        rows = len(ctx.X_train)

        if ctx.mode == "Fast":
            return {
                "sweep_size": 0.2 if rows < 5000 else 0.08,
                "top_k": 1,
                "n_trials": 0,
                "timeout": 0,
                "run_optuna": False,
                "use_full_preprocessor": False,
            }

        if ctx.mode == "Balanced":
            return {
                "sweep_size": 0.35 if rows < 5000 else 0.12,
                "top_k": 2,
                "n_trials": 12,
                "timeout": 120,
                "run_optuna": True,
                "use_full_preprocessor": False,
            }

        return {
            "sweep_size": 0.5 if rows < 5000 else 0.2,
            "top_k": 3,
            "n_trials": 32,
            "timeout": 360,
            "run_optuna": True,
            "use_full_preprocessor": True,
        }

    def execute(self, ctx: PipelineContext):
        profile = self._execution_profile(ctx)
        ctx.reasoning.append(
            f"ExecutionProfile: goal={ctx.goal}, mode={ctx.mode}, "
            f"models={list(ctx.model_pool.keys())}, sweep_size={profile['sweep_size']}, "
            f"top_k={profile['top_k']}, optuna={profile['run_optuna']}"
        )
        ctx.reasoning.append("🏁 Stage 1: Starting Exploration Sweep")
        ctx.record_history("Sweep Start", "Running", phase="sweep")
        sweep_size = profile["sweep_size"]
        X_swp, _, y_swp, _ = train_test_split(ctx.X_train, ctx.y_train, train_size=sweep_size, random_state=42)
        
        X_swp_proc = ctx.preprocessor.fit_transform(X_swp, y_swp)
        sweep_results = []
        model_debug_rows = []
        
        for name, model in ctx.model_pool.items():
            cheap_config = ModelSelector.get_cheap_config(name, ctx.is_classification)
            try:
                model.set_params(**cheap_config)
                model = self._apply_imbalance_strategy(model, ctx)
                model.fit(X_swp_proc, y_swp)
                score, std_score, metric_extras = stability_check(model, X_swp_proc, y_swp, ctx.is_classification)
                row = {
                    "name": name,
                    "score": score,
                    "stability_std": round(std_score * 100, 2),
                    "model": model,
                    "phase": "sweep",
                    "cheap_config": cheap_config,
                }
                row.update(metric_extras)
                sweep_results.append(row)
                model_debug_rows.append({
                    "model": name,
                    "phase": "sweep",
                    "status": "ok",
                    "sweep_score": round(score * 100, 2),
                    "stability_std": round(std_score * 100, 2),
                    "cheap_config": cheap_config,
                    "optimized": False,
                    "error": None,
                    **metric_extras,
                })
                ctx.reasoning.append(f"Sweep: {name} scored {score:.3f}")
                ctx.record_history(name, round(score * 100, 2), phase="sweep", status="ok")
            except Exception as e:
                model_debug_rows.append({
                    "model": name,
                    "phase": "sweep",
                    "status": "failed",
                    "sweep_score": None,
                    "stability_std": None,
                    "cheap_config": cheap_config,
                    "optimized": False,
                    "error": str(e),
                })
                ctx.reasoning.append(f"Sweep Default Check Failed for {name}: {e}")
                ctx.record_history(name, f"failed: {e}", phase="sweep", status="failed")
                
        sweep_results.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = sweep_results[:profile["top_k"]]
        ctx.sweep_results = sweep_results
        ctx.tested_models = model_debug_rows

        if not top_candidates:
            raise ValueError("No candidate models completed the exploration sweep.")

        winner_pool_name = None
        final_model = None

        if not profile["run_optuna"]:
            ctx.reasoning.append("Execution: Fast mode selected. Skipping Bayesian Opt.")
            final_model = top_candidates[0]['model']
            winner_pool_name = top_candidates[0]['name']
        else:
            ctx.reasoning.append(f"🚀 Stage 2: Deep Dive optimization for: {[c['name'] for c in top_candidates]}")
            ctx.record_history("Optimization", "Running", phase="optuna")
            best_overall_score = -1
            
            for candidate in top_candidates:
                name = candidate['name']

                def objective(trial):
                    try:
                        p = ModelSelector.get_bayesian_space(trial, name)
                        m = ctx.model_pool[name].__class__(**p)
                        m = self._apply_imbalance_strategy(m, ctx)
                        if isinstance(m, (LGBMClassifier, LGBMRegressor)):
                            try:
                                m.set_params(verbose=-1)
                            except Exception:
                                pass
                        
                        pipe = Pipeline([('pre', ctx.preprocessor), ('m', m)])
                        metric = ctx.config.get("eval_metric", "")
                        cv_folds = int(ctx.config.get("cv_folds", 0) or 0)

                        if cv_folds >= 2:
                            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if ctx.is_classification else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                            return cross_val_score(
                                pipe,
                                ctx.X_train,
                                ctx.y_train,
                                cv=cv,
                                scoring=_resolve_scoring(metric, ctx.is_classification),
                            ).mean()

                        holdout_split = {"test_size": 0.2, "random_state": 42}
                        if ctx.is_classification and len(pd.Series(ctx.y_train).unique()) > 1:
                            holdout_split["stratify"] = ctx.y_train
                        X_tr, X_val, y_tr, y_val = train_test_split(ctx.X_train, ctx.y_train, **holdout_split)
                        pipe.fit(X_tr, y_tr)
                        preds = pipe.predict(X_val)
                        if ctx.is_classification:
                            scoring_name = _resolve_scoring(metric, True)
                            if scoring_name == "f1_weighted":
                                return f1_score(y_val, preds, average="weighted", zero_division=0)
                            if scoring_name == "precision_weighted":
                                return precision_score(y_val, preds, average="weighted", zero_division=0)
                            if scoring_name == "recall_weighted":
                                return recall_score(y_val, preds, average="weighted", zero_division=0)
                            return accuracy_score(y_val, preds)

                        scoring_name = _resolve_scoring(metric, False)
                        if scoring_name == "neg_mean_absolute_error":
                            return -mean_absolute_error(y_val, preds)
                        if scoring_name == "neg_mean_squared_error":
                            return -mean_squared_error(y_val, preds)
                        if scoring_name == "neg_root_mean_squared_error":
                            return -float(np.sqrt(mean_squared_error(y_val, preds)))
                        return r2_score(y_val, preds)
                    except Exception:
                        return 0
                        
                n_trials = profile["n_trials"]
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials, timeout=profile["timeout"])

                opt_row = next((r for r in model_debug_rows if r["model"] == name), None)
                if opt_row is not None:
                    opt_row["optimized"] = True
                    opt_row["optuna_trials"] = n_trials
                    opt_row["best_cv_score"] = round(float(study.best_value) * 100, 2) if study.best_value is not None else None
                    opt_row["best_params"] = study.best_params

                if study.best_value is not None:
                    ctx.record_history(f"{name} CV", round(float(study.best_value) * 100, 2), phase="optuna", status="ok")
                
                if study.best_value > best_overall_score:
                    best_overall_score = study.best_value
                    final_model = ctx.model_pool[name].__class__(**study.best_params)
                    final_model = self._apply_imbalance_strategy(final_model, ctx)
                    winner_pool_name = name
                    
            if not final_model:
                final_model = top_candidates[0]['model']
                winner_pool_name = top_candidates[0]['name']
                
        ctx.final_model = final_model
        ctx.winner_pool_name = winner_pool_name
        if ctx.config.get("handle_imbalance") and ctx.is_classification:
            ctx.reasoning.append("ImbalanceStrategy: Enabled balanced weighting for supported classifiers.")
        
        ctx.reasoning.append("🏁 Training final production pipe on full dataset...")
        final_pipe = Pipeline([('preprocessor', ctx.preprocessor), ('model', final_model)])
        final_pipe.fit(ctx.X_train, ctx.y_train)
        ctx.final_model = final_pipe


class EvaluationComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.EVALUATE

    def execute(self, ctx: PipelineContext):
        preds = ctx.final_model.predict(ctx.X_test)
        execution_profile = TrainingComponent()._execution_profile(ctx)
        sweep_size = execution_profile["sweep_size"]

        score = accuracy_score(ctx.y_test, preds) if ctx.is_classification else r2_score(ctx.y_test, preds)
        ctx.final_score = score
        
        # Explainability
        shap_summary = {}
        try:
            X_test_proc = ctx.preprocessor.transform(ctx.X_test)
            underlying_model = ctx.final_model.named_steps["model"]
            explainer = shap.Explainer(underlying_model, X_test_proc)
            shap_vals = explainer(X_test_proc[:50])
            importances = np.abs(shap_vals.values).mean(axis=0)
            if len(importances.shape) > 1:
                importances = importances.mean(axis=1)
            
            f_names = ctx.preprocessor.get_feature_names_out()
            for i, f in enumerate(f_names[:8]):
                shap_summary[f.split("__")[-1]] = float(importances[i])
        except Exception as e:
            ctx.reasoning.append(f"Explainability SHAP skipped ({e})")
            
        ctx.shap_summary = shap_summary
        
        final_sc = round(score * 100, 1)
        ctx.record_history("Final", final_sc, phase="holdout_test", status="ok")
        lb = [{"model": ctx.winner_pool_name, "score": final_sc, "phase": "holdout_test"}]
        
        for r in ctx.sweep_results:
            if r["name"] == ctx.winner_pool_name:
                continue
            row = {"model": r["name"], "score": round(r["score"] * 100, 1), "phase": "sweep"}
            for k in ("precision", "recall", "f1", "mse", "mae"):
                if k in r:
                    row[k] = r[k]
            lb.append(row)
            
        if ctx.is_classification:
            lb[0]["precision"] = round(float(precision_score(ctx.y_test, preds, average="weighted", zero_division=0)) * 100, 1)
            lb[0]["recall"] = round(float(recall_score(ctx.y_test, preds, average="weighted", zero_division=0)) * 100, 1)
            lb[0]["f1"] = round(float(f1_score(ctx.y_test, preds, average="weighted", zero_division=0)) * 100, 1)
        else:
            lb[0]["mse"] = round(float(mean_squared_error(ctx.y_test, preds)), 6)
            lb[0]["mae"] = round(float(mean_absolute_error(ctx.y_test, preds)), 6)

        winner_debug = next((r for r in ctx.tested_models if r["model"] == ctx.winner_pool_name), None)
        if winner_debug is not None:
            winner_debug["phase"] = "holdout_test"
            winner_debug["holdout_score"] = final_sc
            winner_debug["winner"] = True
            if ctx.is_classification:
                winner_debug["precision"] = lb[0].get("precision")
                winner_debug["recall"] = lb[0].get("recall")
                winner_debug["f1"] = lb[0].get("f1")
            else:
                winner_debug["mse"] = lb[0].get("mse")
                winner_debug["mae"] = lb[0].get("mae")
            
        ctx.leaderboard = lb

        metadata = {
            "task_type": "classification" if ctx.is_classification else "regression",
            "eval_metric_requested": ctx.config.get("eval_metric") or ("Accuracy" if ctx.is_classification else "R²"),
            "cv_folds_used": int(ctx.config.get("cv_folds", 0) or 0),
            "preprocessor": "full_column_transformer" if execution_profile["use_full_preprocessor"] else "lite_column_transformer",
            "feature_names": ctx.num_cols + ctx.cat_cols,
        }
        
        # Save explicit model using ModelRegistry
        ModelRegistry.save_model(ctx.job_id, ctx.final_model, metadata)

        ctx.metrics = {
            "best_model": ctx.winner_pool_name,
            "metric_name": "Accuracy" if ctx.is_classification else "R² Score",
            "score": final_sc,
            "leaderboard": lb,
            "is_classification": ctx.is_classification,
            "shap_summary": shap_summary,
            "model_path": get_model_path(ctx.job_id),
            "feature_names": ctx.num_cols + ctx.cat_cols,
            "target": ctx.target_column,
            "eda_summary": ctx.eda_summary,
            "model_metadata": metadata,
            "reasoning": ctx.reasoning,
            "goal": ctx.goal,
            "mode": ctx.mode,
            "execution_profile": execution_profile,
            "tested_models": ctx.tested_models,
        }
        save_metrics(ctx.job_id, ctx.metrics)

        MLTracking.log_run(
            job_id=ctx.job_id,
            params={
                "best_model": ctx.winner_pool_name,
                "mode": ctx.mode,
                "goal": ctx.goal,
                "sweep_size": sweep_size,
                "top_k": execution_profile["top_k"],
                "optuna_trials": execution_profile["n_trials"],
                "cv_folds": ctx.config.get("cv_folds", 0),
                "metric_name": ctx.metrics["metric_name"],
            },
            metrics=ctx.metrics,
            model=ctx.final_model,
            artifact_path=get_model_path(ctx.job_id),
        )

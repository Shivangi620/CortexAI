import numpy as np
import pandas as pd
import shap
import optuna
import json
import platform
import sys
from typing import Any, Dict
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_val_predict,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LGBM_TYPES = (LGBMClassifier, LGBMRegressor)
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None
    LGBM_TYPES = tuple()

from core.pipeline_engine import PipelineComponent, PipelineContext, PipelineStep
from core.feature_engine import ManagedFeatureEngine
from core.integrations import MLTracking
from infra.database import DatasetModel, db_session
from infra.storage import (
    ModelRegistry,
    DataContract,
    get_model_path,
    get_schema_path,
    save_metrics,
)
from services.data_sanitizer import sanitize_dataframe, summarize_experiment
from services.leakage_service import run_leakage_report
from services.training.evaluator import _resolve_scoring, detect_task_type, normalize_training_controls
from services.training.inference import InferenceArtifact, TabularModelPipeline
from services.training.model_selector import ModelSelector
from core.file_loader import load_dataframe


def _score_for_ranking(score: float, is_classification: bool) -> float:
    """Keep sklearn's scoring orientation intact when ranking candidates."""
    return float(score)


def _classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    roc_auc = None
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
        except Exception:
            roc_auc = None

    return {
        "accuracy": round(accuracy * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1 * 100, 1),
        "roc_auc": round(roc_auc * 100, 1) if roc_auc is not None else None,
    }


def _regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "r2": round(r2, 6),
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
    }


def _best_completed_trial(study):
    completed_trials = [
        trial
        for trial in getattr(study, "trials", [])
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        return None
    return max(completed_trials, key=lambda trial: float(trial.value))


def _resolve_target_column_name(columns, requested_target):
    requested = (requested_target or "").strip()
    if not requested:
        return None
    if requested in columns:
        return requested

    normalized_requested = requested.casefold().replace(" ", "").replace("_", "")
    for column in columns:
        normalized_column = str(column).casefold().replace(" ", "").replace("_", "")
        if normalized_column == normalized_requested:
            return column
    return None


def _resolve_final_model_choice(final_model, winner_pool_name, top_candidates):
    if final_model is not None:
        return final_model, winner_pool_name
    if top_candidates:
        return top_candidates[0]["model"], top_candidates[0]["name"]
    return None, winner_pool_name


def _coerce_estimator_instance(
    final_model, winner_pool_name, model_pool, top_candidates
):
    if final_model is not None and not isinstance(final_model, str):
        return final_model

    candidate_name = winner_pool_name if winner_pool_name else None
    if isinstance(final_model, str) and final_model:
        candidate_name = final_model

    if candidate_name and candidate_name in model_pool:
        return model_pool[candidate_name]

    if top_candidates:
        return top_candidates[0]["model"]

    return final_model


_SIMPLE_MODEL_NAMES = {
    "Logistic Regression",
    "Linear Regression",
    "Ridge",
    "ElasticNet",
    "Random Forest",
    "Hist Gradient Boosting",
}
_ADVANCED_BOOSTER_NAMES = {"XGBoost", "LightGBM"}


def _is_simple_model_candidate(name: str) -> bool:
    return str(name or "") in _SIMPLE_MODEL_NAMES


def _simple_model_is_good_enough(candidate: Dict[str, Any], is_classification: bool) -> bool:
    stability_std = candidate.get("stability_std")
    if stability_std is None or float(stability_std) > 3.0:
        return False

    score = candidate.get("score")
    if score is not None:
        score = float(score)
        if score >= 90.0 or score >= 0.9:
            return True

    if is_classification:
        for key in ("f1", "accuracy", "roc_auc"):
            value = candidate.get(key)
            if value is not None and float(value) >= 90.0:
                return True
        return False

    r2 = candidate.get("r2")
    if r2 is not None and float(r2) >= 0.9:
        return True
    return False


def _refill_candidates_from_sweep(top_candidates, sweep_results, top_k, excluded_names=None):
    selected_names = {row["name"] for row in top_candidates}
    excluded = set(excluded_names or [])
    refilled = list(top_candidates)
    if len(refilled) >= top_k:
        return refilled[:top_k]
    for row in sweep_results:
        if row["name"] in selected_names or row["name"] in excluded:
            continue
        refilled.append(row)
        selected_names.add(row["name"])
        if len(refilled) >= top_k:
            break
    return refilled


def _prune_correlated_candidates(top_candidates, sweep_results, execution_profile):
    if len(top_candidates) < 2:
        return top_candidates, []

    notes = []
    kept = []
    excluded_names = set()
    for candidate in top_candidates:
        name = candidate["name"]
        if name in excluded_names:
            continue
        kept.append(candidate)
        candidate_scores = np.asarray(candidate.get("cv_scores") or [], dtype=float)
        if candidate_scores.size < 3:
            continue
        for other in top_candidates:
            other_name = other["name"]
            if other_name == name or other_name in excluded_names:
                continue
            other_scores = np.asarray(other.get("cv_scores") or [], dtype=float)
            if other_scores.size != candidate_scores.size or other_scores.size < 3:
                continue
            try:
                corr = float(np.corrcoef(candidate_scores, other_scores)[0, 1])
            except Exception:
                corr = 0.0
            if np.isfinite(corr) and corr >= 0.985:
                score_gap = abs(float(candidate.get("score", 0.0)) - float(other.get("score", 0.0)))
                if score_gap <= 0.02:
                    excluded_names.add(other_name)
                    notes.append(
                        f"DiversityGuard: {other_name} tracked {name} almost identically across folds, so only one was kept for deep tuning."
                    )
        if len(kept) >= execution_profile["top_k"]:
            break

    refilled = _refill_candidates_from_sweep(
        kept,
        sweep_results,
        execution_profile["top_k"],
        excluded_names=excluded_names,
    )
    return refilled, notes


def _select_fallback_model(model_pool):
    for name in [
        "Logistic Regression",
        "Linear Regression",
        "Ridge",
        "ElasticNet",
        "Random Forest",
        "Hist Gradient Boosting",
    ]:
        if name in model_pool:
            model = model_pool[name]
            return name, model.__class__(**model.get_params())
    for name, model in model_pool.items():
        return name, model.__class__(**model.get_params())
    return None, None


def _prune_optuna_candidates(top_candidates, sweep_results, execution_profile, is_classification):
    if not top_candidates:
        return top_candidates, []

    notes = []
    pruned = list(top_candidates)

    top_simple = next((row for row in pruned if _is_simple_model_candidate(row["name"])), None)
    if top_simple and _simple_model_is_good_enough(top_simple, is_classification):
        simple_only = [row for row in pruned if _is_simple_model_candidate(row["name"])]
        if simple_only:
            pruned = simple_only[: max(1, min(len(simple_only), execution_profile["top_k"]))]
            notes.append(
                f"EarlyStop: {top_simple['name']} already cleared the quality threshold, so heavy candidates were skipped for deep tuning."
            )
            return pruned, notes

    advanced_boosters = [row for row in pruned if row["name"] in _ADVANCED_BOOSTER_NAMES]
    if len(advanced_boosters) > 1:
        best_booster = max(advanced_boosters, key=lambda row: float(row.get("score", float("-inf"))))
        removed_boosters = {
            row["name"] for row in advanced_boosters if row["name"] != best_booster["name"]
        }
        pruned = [
            row
            for row in pruned
            if row["name"] not in _ADVANCED_BOOSTER_NAMES or row["name"] == best_booster["name"]
        ]
        pruned = _refill_candidates_from_sweep(
            pruned,
            sweep_results,
            execution_profile["top_k"],
            excluded_names=removed_boosters,
        )
        notes.append(
            f"Hierarchy: {best_booster['name']} led the advanced booster sweep, so redundant booster tuning was skipped."
        )

    pruned, diversity_notes = _prune_correlated_candidates(
        pruned,
        sweep_results,
        execution_profile,
    )
    notes.extend(diversity_notes)

    return pruned, notes


def _training_profile_from_frame(X: pd.DataFrame, y: pd.Series, is_classification: bool) -> Dict[str, Any]:
    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = [column for column in X.columns if column not in num_cols]
    numeric_max_corr = 0.0
    if len(num_cols) >= 2:
        numeric_frame = X[num_cols].apply(pd.to_numeric, errors="coerce")
        numeric_frame = numeric_frame.fillna(numeric_frame.median(numeric_only=True))
        try:
            corr = numeric_frame.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            finite = upper.to_numpy()
            finite = finite[np.isfinite(finite)]
            if finite.size:
                numeric_max_corr = float(finite.max())
        except Exception:
            numeric_max_corr = 0.0

    target_entropy = 0.0
    clean_target = pd.Series(y).dropna()
    unique_count = int(clean_target.nunique())
    if is_classification and unique_count > 1:
        probs = clean_target.astype(str).value_counts(normalize=True)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        target_entropy = entropy / max(np.log(len(probs)), 1e-12)
    elif unique_count > 20:
        target_entropy = min(unique_count / max(len(clean_target), 1) * 20.0, 1.0)

    return {
        "rows": len(X),
        "cols": len(X.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "column_stats": {},
        "target_entropy": round(float(target_entropy), 4),
        "numeric_max_corr": round(float(numeric_max_corr), 4),
    }


class DataValidationComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.VALIDATE

    def _normalize_columns_only(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        normalized = df.copy()
        logs = []
        original_columns = list(normalized.columns)
        normalized.columns = [
            str(col).strip().replace("\n", " ") for col in normalized.columns
        ]
        if list(normalized.columns) != original_columns:
            logs.append("Validation: Normalized column names.")
        return normalized, logs

    def execute(self, ctx: PipelineContext):
        df = load_dataframe(filepath=ctx.file_path)
        ctx.reasoning.append(
            f"DataValidation: Loaded dataset with {df.shape[1]} columns: {list(df.columns)}"
        )

        df, normalization_logs = self._normalize_columns_only(df)
        ctx.reasoning.extend(normalization_logs)

        resolved_target = _resolve_target_column_name(df.columns, ctx.target_column)
        if resolved_target is None:
            raise ValueError(
                f"Target column '{ctx.target_column}' not found in dataset"
            )
        if resolved_target != ctx.target_column:
            ctx.reasoning.append(
                f"DataValidation: Using matched target column '{resolved_target}' for requested input '{ctx.target_column}'."
            )
            ctx.target_column = resolved_target

        selected_features = ctx.config.get("selected_features")
        if selected_features:
            keep_cols = list(selected_features) + [ctx.target_column]
            available = [c for c in keep_cols if c in df.columns]
            df = df[available]
            ctx.reasoning.append(
                f"DataValidation: Feature selection applied, keeping {len(available)} columns: {available}"
            )

        if ctx.config.get("auto_clean", True):
            sanitized = sanitize_dataframe(df, target=ctx.target_column)
            df = sanitized.df
            ctx.reasoning.extend(sanitized.logs)
            ctx.sanitizer_report = sanitized.report
            ctx.reasoning.append(
                "DataValidation: Applied the shared sanitizer/validator before profiling and training."
            )
            if not sanitized.report.get("target_valid", True):
                raise ValueError(
                    sanitized.report.get("target_issue")
                    or "Invalid target column after sanitization."
                )

            nan_percentages = df.isna().mean()
            high_nan_cols = nan_percentages[nan_percentages > 0.9].index.tolist()
            if high_nan_cols:
                ctx.reasoning.append(
                    f"DataValidation: Warning - dropping columns with >90% missing values: {high_nan_cols}"
                )
                df = df.drop(columns=high_nan_cols)

            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception:
                        pass
        else:
            ctx.sanitizer_report = {
                "rows_before": int(len(df)),
                "rows_after": int(len(df)),
                "columns_before": int(len(df.columns)),
                "columns_after": int(len(df.columns)),
                "target_valid": True,
                "target_issue": None,
                "duplicate_rows_removed": 0,
                "numeric_coercions": [],
                "datetime_columns": [],
                "categorical_columns": [],
                "empty_string_cells_cleaned": 0,
                "dropped_target_rows": 0,
            }
            if ctx.target_column not in df.columns:
                ctx.sanitizer_report["target_valid"] = False
                ctx.sanitizer_report["target_issue"] = (
                    f"Target column '{ctx.target_column}' not found."
                )
            elif df[ctx.target_column].dropna().empty:
                ctx.sanitizer_report["target_valid"] = False
                ctx.sanitizer_report["target_issue"] = (
                    "Target has no non-null values."
                )
            ctx.reasoning.append(
                "DataValidation: Auto-clean disabled; keeping raw values and applying only minimal validation."
            )
            if not ctx.sanitizer_report.get("target_valid", True):
                raise ValueError(
                    ctx.sanitizer_report.get("target_issue")
                    or "Invalid target column before training."
                )

        y_preview = df[ctx.target_column].dropna()
        task_type = detect_task_type(
            y_preview,
            target_name=ctx.target_column,
            override=ctx.config.get("task_type", ""),
        )["task_type"]
        ctx.reasoning.append(
            f"DataValidation: Initial task hint is {task_type}; leakage checks will run on the training split only."
        )

        ctx.reasoning.append(
            f"DataValidation: After validation, {df.shape[1]} columns remain: {list(df.columns)}"
        )

        # Enforce Data Contract immediately
        DataContract.save_contract(ctx.job_id, df)

        ctx.df = df
        ctx.reasoning.append(
            "DataValidation: Uploaded data matches contract. Constraints enforced."
        )


class FeatureEngineeringComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.FEATURE_ENG

    def _detect_temporal_order(self, ctx: PipelineContext, X: pd.DataFrame) -> tuple[bool, str]:
        datetime_candidates = list(
            (getattr(ctx, "sanitizer_report", {}) or {}).get("datetime_columns") or []
        )
        for column in datetime_candidates:
            if column not in X.columns:
                continue
            series = pd.to_datetime(X[column], errors="coerce", format="mixed")
            valid = series.dropna()
            if len(valid) < max(8, int(len(series) * 0.6)):
                continue
            if valid.is_monotonic_increasing:
                return True, column
            if valid.is_monotonic_decreasing:
                return True, column
        return False, ""

    def execute(self, ctx: PipelineContext):
        df = ctx.df
        ctx.reasoning.append(
            f"FeatureEngineer: Starting with {df.shape[1]} columns: {list(df.columns)}"
        )

        y_raw = df[ctx.target_column]
        X = df.drop(columns=[ctx.target_column])

        invalid_target = y_raw.isna()
        if y_raw.dtype == object or pd.api.types.is_string_dtype(y_raw):
            sr = y_raw.astype(str).str.strip().str.lower()
            invalid_target = invalid_target | sr.isin(
                (
                    "nan",
                    "none",
                    "",
                    "na",
                    "n/a",
                    "null",
                    "?",
                    "unknown",
                    "??",
                    "invalid",
                )
            )

        dropped_target = int(invalid_target.sum())
        if dropped_target > 0:
            ctx.reasoning.append(
                f"TargetCleaner: Removed {dropped_target} invalid target rows."
            )

        X = X.loc[~invalid_target].reset_index(drop=True)
        y = y_raw.loc[~invalid_target].reset_index(drop=True)

        if len(y) == 0:
            raise ValueError(
                "No rows left after dropping invalid target rows. Cannot train."
            )

        ctx.eda_summary = {
            "rows_after_target_cleaning": int(len(y)),
            "columns_after_feature_engineering": int(X.shape[1]),
            "target_missing_removed": dropped_target,
            "numeric_features": int(X.select_dtypes(include=[np.number]).shape[1]),
            "categorical_features": int(
                X.select_dtypes(include=["object", "category", "bool"]).shape[1]
            ),
            "feature_synthesis_applied": False,
        }
        task_decision = detect_task_type(
            y,
            target_name=ctx.target_column,
            override=ctx.config.get("task_type", ""),
        )
        ctx.task_decision = task_decision
        ctx.is_classification = task_decision["task_type"] == "classification"
        ctx.reasoning.append(
            f"TaskDetection: {task_decision['reason']} (source={task_decision['source']})."
        )
        if ctx.is_classification:
            ctx.label_encoder = LabelEncoder()
            y = pd.Series(
                ctx.label_encoder.fit_transform(y.astype(str)),
                index=y.index,
            )
            ctx.class_labels = list(ctx.label_encoder.classes_)
        else:
            ctx.label_encoder = None
            ctx.class_labels = []

        # Validate that we have at least some columns
        if X.shape[1] == 0:
            error_msg = (
                f"No usable columns found after feature engineering. X shape: {X.shape}, dtypes:\n{X.dtypes}\n\n"
                "This usually means:\n"
                "1. All features were dropped during data cleaning (identifiers, constants, high missing values)\n"
                "2. Target leakage detection removed all features\n"
                "3. Feature selection was too restrictive\n"
                "4. The dataset only contains the target column\n\n"
                "Check the reasoning logs above for details on what columns were dropped and why."
            )
            raise ValueError(error_msg)

        temporal_validation, temporal_column = self._detect_temporal_order(ctx, X)
        ctx.config["temporal_validation"] = temporal_validation
        ctx.config["temporal_order_column"] = temporal_column

        if temporal_validation:
            order = pd.to_datetime(X[temporal_column], errors="coerce", format="mixed")
            sorted_index = order.sort_values(kind="mergesort").index
            X = X.loc[sorted_index].reset_index(drop=True)
            y = y.loc[sorted_index].reset_index(drop=True)

            split_at = max(1, min(len(X) - 1, int(np.floor(len(X) * 0.8))))
            X_train, X_test = X.iloc[:split_at].copy(), X.iloc[split_at:].copy()
            y_train, y_test = y.iloc[:split_at].copy(), y.iloc[split_at:].copy()
            ctx.reasoning.append(
                f"ValidationSplit: Detected temporal ordering in '{temporal_column}', so the last {len(X_test)} rows are held out chronologically."
            )
        else:
            split_kwargs = {"test_size": 0.2, "random_state": 42}
            if ctx.is_classification:
                split_kwargs["stratify"] = y

            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

        train_for_leakage = X_train.copy()
        train_for_leakage[ctx.target_column] = y_train.values
        leaks = ManagedFeatureEngine(
            target_col=ctx.target_column,
            task_type="classification" if ctx.is_classification else "regression",
        ).detect_leakage(train_for_leakage)
        if leaks:
            X = X.drop(columns=leaks, errors="ignore")
            X_train = X_train.drop(columns=leaks, errors="ignore")
            X_test = X_test.drop(columns=leaks, errors="ignore")
            ctx.reasoning.append(
                f"LeakageGuard: Dropped suspicious features using training-split-only checks: {leaks}"
            )

        ctx.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        ctx.cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        ctx.raw_feature_names = list(X.columns)

        if len(ctx.num_cols) == 0 and len(ctx.cat_cols) == 0:
            raise ValueError(
                "All candidate features were removed after train-only leakage checks."
            )

        from core.drift_detector import DriftDetector

        DriftDetector(ctx.job_id).fit_baseline(X_train[ctx.raw_feature_names])
        ctx.reasoning.append(
            f"DriftBaseline: Fitted baseline on {len(ctx.raw_feature_names)} training feature(s)."
        )
        ctx.reasoning.append(
            f"FeatureEngineer: Identified {len(ctx.num_cols)} numeric and {len(ctx.cat_cols)} categorical features."
        )
        ctx.reasoning.append(
            "FeatureEngineer: Deferred deterministic raw-input feature engineering to the model artifact so CV, training, and inference use the same path."
        )

        ctx.X_train, ctx.X_test = X_train, X_test
        ctx.y_train, ctx.y_test = y_train, y_test
        ctx.X, ctx.y = X, y


class ModelSelectionComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.TRAIN

    def execute(self, ctx: PipelineContext):
        controls = normalize_training_controls(
            task_type="classification" if ctx.is_classification else "regression",
            goal=ctx.goal,
            mode=ctx.mode,
            eval_metric=ctx.config.get("eval_metric", ""),
            handle_imbalance=ctx.config.get("handle_imbalance", False),
        )
        ctx.goal = controls["goal"]
        ctx.mode = controls["mode"]
        ctx.config["eval_metric"] = controls["eval_metric"]
        ctx.config["handle_imbalance"] = controls["handle_imbalance"]
        for warning in controls["warnings"]:
            ctx.reasoning.append(f"TrainingConfig: {warning}")

        profile = _training_profile_from_frame(ctx.X, ctx.y, ctx.is_classification)

        model_pool, meta_rec = ModelSelector.select_pool(
            len(ctx.X), ctx.is_classification, ctx.goal, profile, mode=ctx.mode
        )
        ctx.reasoning.append(
            f"Meta-Learner Advisory: {meta_rec['reason']} (Source: {meta_rec['source']})"
        )
        ctx.dataset_traits = meta_rec.get("goal_profile", {}).get("dataset_traits", {})
        memory_signal = meta_rec.get("memory_signal") or {}
        if memory_signal.get("applied"):
            reordered = memory_signal.get("reordered_models") or []
            if reordered:
                ctx.reasoning.append(
                    "Meta-Learner Memory: historical winners nudged the safe registry order for "
                    + ", ".join(reordered[:4])
                    + "."
                )
        complexity_score = ctx.dataset_traits.get("complexity_score")
        if complexity_score is not None:
            ctx.reasoning.append(
                f"ComplexityScore: dataset complexity estimated at {complexity_score:.3f}; low-complexity datasets can skip advanced boosters."
            )
        ctx.reasoning.append(
            "ModelSelection: Meta-learning can gently reorder candidates inside the safe registry when historical confidence is strong, but every selected model still runs under the same metric and validation scheme."
        )
        ctx.model_pool = model_pool
        pca_mode = str(ctx.config.get("pca_mode") or "auto").lower()
        pca_components = int(ctx.config.get("pca_components", 0) or 0)
        if ctx.mode == "Full":
            ctx.preprocessing_kind = "full"
            ctx.reasoning.append(
                "Preprocessor: Full mode will use richer numeric preprocessing inside the end-to-end artifact."
            )
        else:
            ctx.preprocessing_kind = "lite"
            ctx.reasoning.append(
                "Preprocessor: Lite preprocessing will be used inside the end-to-end artifact."
            )

        if pca_mode == "always":
            if pca_components > 1:
                ctx.reasoning.append(
                    f"Dimensionality: PCA forced on using up to {pca_components} components for the numeric branch."
                )
            else:
                ctx.reasoning.append(
                    "Dimensionality: PCA forced on for the numeric branch."
                )
        elif pca_mode == "off":
            ctx.reasoning.append(
                "Dimensionality: PCA disabled by the training configuration."
            )
        elif len(ctx.num_cols) >= 40:
            ctx.reasoning.append(
                "Dimensionality: PCA compression enabled for the numeric branch to keep wide datasets responsive."
            )


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
        goal = str(ctx.goal or "Balanced")

        if ctx.mode == "Fast":
            profile = {
                "sweep_size": 0.2 if rows < 5000 else 0.08,
                "top_k": 1,
                "n_trials": 0,
                "timeout": 0,
                "run_optuna": False,
                "use_full_preprocessor": False,
                "mode": "Fast",
            }
        elif ctx.mode == "Balanced":
            profile = {
                "sweep_size": 0.35 if rows < 5000 else 0.12,
                "top_k": 2,
                "n_trials": 12,
                "timeout": 120,
                "run_optuna": True,
                "use_full_preprocessor": False,
                "mode": "Balanced",
            }
        else:
            profile = {
                "sweep_size": 0.5 if rows < 5000 else 0.2,
                "top_k": 3,
                "n_trials": 32,
                "timeout": 360,
                "run_optuna": True,
                "use_full_preprocessor": True,
                "mode": "Full",
            }

        if goal == "Speed":
            profile["sweep_size"] = round(float(profile["sweep_size"]) * 0.7, 4)
            profile["top_k"] = max(1, min(int(profile["top_k"]), 2))
            profile["n_trials"] = int(max(0, round(profile["n_trials"] * 0.35)))
            profile["timeout"] = int(max(0, round(profile["timeout"] * 0.5)))
        elif goal == "Performance":
            profile["sweep_size"] = min(0.75, round(float(profile["sweep_size"]) * 1.15, 4))
            if profile["run_optuna"]:
                profile["top_k"] = min(int(profile["top_k"]) + 1, 4)
                profile["n_trials"] = int(round(profile["n_trials"] * 1.35))
                profile["timeout"] = int(round(profile["timeout"] * 1.3))

        profile["goal"] = goal
        return profile

    def _safe_train_subset(
        self, X, y, train_size: float, is_classification: bool, random_state: int = 42
    ):
        y_series = pd.Series(y)
        total_rows = len(y_series)
        if total_rows <= 2:
            return X, y

        desired_train = max(2, min(total_rows - 1, int(round(total_rows * train_size))))
        test_size = max(1, total_rows - desired_train)

        split_kwargs = {"train_size": desired_train, "random_state": random_state}
        if is_classification and y_series.nunique() > 1:
            min_class_count = int(y_series.value_counts().min())
            if test_size >= y_series.nunique() and min_class_count >= 2:
                split_kwargs["stratify"] = y

        try:
            X_train_slice, _, y_train_slice, _ = train_test_split(X, y, **split_kwargs)
            return X_train_slice, y_train_slice
        except Exception:
            try:
                X_train_slice, _, y_train_slice, _ = train_test_split(
                    X, y, train_size=desired_train, random_state=random_state
                )
                return X_train_slice, y_train_slice
            except Exception:
                return X, y

    def _safe_cv(self, y, requested_folds: int, is_classification: bool):
        requested_folds = int(requested_folds or 5)
        if requested_folds < 2:
            requested_folds = 2
        if not is_classification:
            return KFold(
                n_splits=min(requested_folds, max(2, len(pd.Series(y)) // 2)),
                shuffle=True,
                random_state=42,
            )

        y_series = pd.Series(y)
        min_class_count = (
            int(y_series.value_counts().min()) if not y_series.empty else 0
        )
        safe_folds = min(requested_folds, min_class_count)
        if safe_folds < 2:
            return None
        return StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42)

    def _build_cv_splitter(self, ctx: PipelineContext, y):
        requested_folds = int(ctx.config.get("cv_folds", 0) or 5)
        if ctx.config.get("temporal_validation"):
            n_splits = min(requested_folds, max(2, len(pd.Series(y)) - 1))
            if n_splits < 2:
                return None
            return TimeSeriesSplit(n_splits=n_splits)
        return self._safe_cv(y, requested_folds, ctx.is_classification)

    def _build_candidate_pipeline(self, estimator, ctx: PipelineContext):
        return TabularModelPipeline(
            base_estimator=estimator,
            feature_columns=getattr(ctx, "raw_feature_names", []),
            preprocessing=getattr(ctx, "preprocessing_kind", "lite"),
            pca_mode=ctx.config.get("pca_mode", "auto"),
            pca_components=int(ctx.config.get("pca_components", 0) or 0),
        )

    def execute(self, ctx: PipelineContext):
        execution_profile = self._execution_profile(ctx)

        if not ctx.model_pool:
            raise ValueError(
                "ModelSelectionComponent failed to populate model_pool. No models available for training."
            )

        ctx.reasoning.append(
            f"ExecutionProfile: goal={ctx.goal}, mode={ctx.mode}, "
            f"models={list(ctx.model_pool.keys())}, sweep_size={execution_profile['sweep_size']}, "
            f"top_k={execution_profile['top_k']}, optuna={execution_profile['run_optuna']}"
        )
        if ctx.goal == "Speed":
            ctx.reasoning.append(
                "ExecutionProfile: Speed goal trimmed sweep depth and optimization budget for faster turnaround."
            )
        elif ctx.goal == "Performance":
            ctx.reasoning.append(
                "ExecutionProfile: Performance goal expanded shortlist and tuning budget to push for higher model quality."
            )
        ctx.reasoning.append("🏁 Stage 1: Starting Exploration Sweep")
        ctx.record_history("Sweep Start", "Running", phase="sweep")
        sweep_size = execution_profile["sweep_size"]
        requested_rows = max(64, int(len(ctx.X_train) * sweep_size))
        sample_rows = min(requested_rows, 5000 if ctx.mode == "Fast" else 8000)
        effective_train_size = min(sample_rows / max(len(ctx.X_train), 1), 0.95)
        execution_profile["effective_sweep_rows"] = sample_rows
        X_swp, y_swp = self._safe_train_subset(
            ctx.X_train,
            ctx.y_train,
            train_size=effective_train_size,
            is_classification=ctx.is_classification,
            random_state=42,
        )

        # Validate sweep data
        ctx.reasoning.append(f"Sweep Data Shape: X={X_swp.shape}, y={y_swp.shape}")
        if len(X_swp) == 0:
            raise ValueError("Sweep subset is empty after train/test split.")
        if len(y_swp) == 0:
            raise ValueError("Sweep target is empty after train/test split.")

        sweep_results = []
        model_debug_rows = []
        scoring_name = _resolve_scoring(
            ctx.config.get("eval_metric", ""), ctx.is_classification
        )
        cv = self._build_cv_splitter(ctx, y_swp)
        if cv is None:
            raise ValueError(
                "Not enough label support to run consistent cross-validation. Add more rows per class or lower task complexity."
            )

        for name, model in ctx.model_pool.items():
            cheap_config = ModelSelector.get_cheap_config(name, ctx.is_classification)
            try:
                candidate_estimator = model.__class__(**{
                    **model.get_params(),
                    **cheap_config,
                })
                candidate_estimator = self._apply_imbalance_strategy(
                    candidate_estimator, ctx
                )
                candidate_pipeline = self._build_candidate_pipeline(
                    candidate_estimator, ctx
                )
                cv_scores = cross_val_score(
                    candidate_pipeline,
                    X_swp,
                    y_swp,
                    cv=cv,
                    scoring=scoring_name,
                )
                score = float(np.mean(cv_scores))
                std_score = float(np.std(cv_scores))
                score_for_sort = _score_for_ranking(score, ctx.is_classification)
                metric_details = {}
                try:
                    cv_pred = cross_val_predict(
                        candidate_pipeline,
                        X_swp,
                        y_swp,
                        cv=cv,
                    )
                    if ctx.is_classification:
                        cv_proba = None
                        try:
                            cv_proba = cross_val_predict(
                                candidate_pipeline,
                                X_swp,
                                y_swp,
                                cv=cv,
                                method="predict_proba",
                            )
                        except Exception:
                            cv_proba = None
                        metric_details = _classification_metrics(
                            y_swp, cv_pred, y_proba=cv_proba
                        )
                    else:
                        metric_details = _regression_metrics(y_swp, cv_pred)
                except Exception as metric_exc:
                    ctx.reasoning.append(
                        f"Sweep Metrics: {name} metric bundle skipped ({type(metric_exc).__name__}: {metric_exc})."
                    )
                row = {
                    "name": name,
                    "score": score_for_sort,
                    "cv_mean_raw": score,
                    "stability_std": round(std_score * 100, 2),
                    "model": candidate_estimator,
                    "phase": "cross_validation",
                    "cheap_config": cheap_config,
                    "cv_scores": [round(float(item) * 100, 2) for item in cv_scores],
                }
                row.update(metric_details)
                sweep_results.append(row)
                model_debug_rows.append(
                    {
                        "model": name,
                        "phase": "cross_validation",
                        "status": "ok",
                        "sweep_score": round(score_for_sort * 100, 2),
                        "stability_std": round(std_score * 100, 2),
                        "cheap_config": cheap_config,
                        "optimized": False,
                        "error": None,
                        "cv_scores": [round(float(item) * 100, 2) for item in cv_scores],
                        **metric_details,
                    }
                )
                ctx.reasoning.append(
                    f"Sweep CV: {name} scored {score:.3f} with std {std_score:.3f}."
                )
                ctx.record_history(
                    name, round(score_for_sort * 100, 2), phase="cross_validation", status="ok"
                )
            except Exception as e:
                error_detail = f"{type(e).__name__}: {str(e)}"
                model_debug_rows.append(
                    {
                        "model": name,
                        "phase": "sweep",
                        "status": "failed",
                        "sweep_score": None,
                        "stability_std": None,
                        "cheap_config": cheap_config,
                        "optimized": False,
                        "error": error_detail,
                    }
                )
                ctx.reasoning.append(f"Sweep Failed for {name}: {error_detail}")
                ctx.record_history(
                    name, f"failed: {error_detail}", phase="sweep", status="failed"
                )

        sweep_results.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = sweep_results[: execution_profile["top_k"]]
        ctx.sweep_results = sweep_results
        ctx.tested_models = model_debug_rows

        if not top_candidates:
            fallback_name, fallback_model = _select_fallback_model(ctx.model_pool)
            if fallback_model is None:
                failed_models = [
                    (r["model"], r["error"])
                    for r in model_debug_rows
                    if r["status"] == "failed"
                ]
                error_summary = "\n".join(
                    [f"  {name}: {error}" for name, error in failed_models]
                )
                raise ValueError(
                    f"No candidate models completed the exploration sweep.\nFailed models:\n{error_summary}"
                )
            fallback_model = self._apply_imbalance_strategy(fallback_model, ctx)
            ctx.reasoning.append(
                f"FallbackSafety: Every sweep candidate failed, so training will continue with baseline fallback '{fallback_name}'."
            )
            ctx.final_model = fallback_model
            ctx.record_history(fallback_name or "Fallback", "baseline_fallback", phase="fallback", status="ok")
            return

        winner_pool_name = None
        final_model = None

        if not execution_profile["run_optuna"]:
            ctx.reasoning.append(
                "Execution: Fast mode selected. Skipping Bayesian Opt."
            )
            final_model = top_candidates[0]["model"].__class__(**top_candidates[0]["model"].get_params())
            winner_pool_name = top_candidates[0]["name"]
        else:
            top_candidates, optuna_notes = _prune_optuna_candidates(
                top_candidates,
                sweep_results,
                execution_profile,
                ctx.is_classification,
            )
            ctx.reasoning.extend(optuna_notes)
            ctx.reasoning.append(
                f"🚀 Stage 2: Deep Dive optimization for: {[c['name'] for c in top_candidates]}"
            )
            ctx.record_history("Optimization", "Running", phase="optuna")
            best_overall_score = float("-inf")

            for candidate in top_candidates:
                name = candidate["name"]

                def objective(trial):
                    try:
                        p = ModelSelector.get_bayesian_space(trial, name)
                        m = ctx.model_pool[name].__class__(**p)
                        m = self._apply_imbalance_strategy(m, ctx)
                        if LGBM_TYPES and isinstance(m, LGBM_TYPES):
                            try:
                                m.set_params(verbose=-1)
                            except Exception:
                                pass
                        candidate_pipeline = self._build_candidate_pipeline(m, ctx)
                        return float(
                            cross_val_score(
                                candidate_pipeline,
                                ctx.X_train,
                                ctx.y_train,
                                cv=self._build_cv_splitter(ctx, ctx.y_train),
                                scoring=scoring_name,
                            ).mean()
                        )
                    except Exception as exc:
                        raise optuna.TrialPruned(
                            f"{name} trial failed: {type(exc).__name__}: {exc}"
                        ) from exc

                budget = ModelSelector.get_tuning_budget(
                    name,
                    execution_profile["n_trials"],
                    execution_profile["timeout"],
                    getattr(ctx, "dataset_traits", None),
                )
                n_trials = budget["trials"]
                timeout = budget["timeout"]
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    objective, n_trials=n_trials, timeout=timeout
                )
                opt_row = next(
                    (r for r in model_debug_rows if r["model"] == name), None
                )
                if opt_row is not None:
                    opt_row["optimized"] = True
                    opt_row["optuna_trials"] = n_trials
                    opt_row["optuna_timeout"] = timeout
                    opt_row["best_cv_score"] = None
                    opt_row["best_params"] = None

                best_trial = _best_completed_trial(study)
                if best_trial is None:
                    ctx.reasoning.append(
                        f"Optuna: {name} produced no completed trials; keeping sweep configuration."
                    )
                    continue

                if opt_row is not None:
                    opt_row["best_cv_score"] = round(float(best_trial.value) * 100, 2)
                    opt_row["best_params"] = best_trial.params

                if best_trial.value is not None:
                    ctx.record_history(
                        f"{name} CV",
                        round(float(best_trial.value) * 100, 2),
                        phase="optuna",
                        status="ok",
                    )

                if float(best_trial.value) > best_overall_score:
                    best_overall_score = float(best_trial.value)
                    final_model = ctx.model_pool[name].__class__(**best_trial.params)
                    final_model = self._apply_imbalance_strategy(final_model, ctx)
                    winner_pool_name = name

        if final_model is None:
            final_model = top_candidates[0]["model"].__class__(
                **top_candidates[0]["model"].get_params()
            )
            winner_pool_name = top_candidates[0]["name"]
        ctx.final_model = final_model
        ctx.winner_pool_name = winner_pool_name
        if ctx.config.get("handle_imbalance") and ctx.is_classification:
            ctx.reasoning.append(
                "ImbalanceStrategy: Enabled balanced weighting for supported classifiers."
            )

        ctx.reasoning.append(
            "🏁 Training final end-to-end production artifact on the full training split..."
        )
        final_pipe = self._build_candidate_pipeline(final_model, ctx)
        final_pipe.fit(ctx.X_train, ctx.y_train)
        ctx.final_model = final_pipe
        ctx.execution_profile = execution_profile


class EvaluationComponent(PipelineComponent):
    def get_step_type(self) -> PipelineStep:
        return PipelineStep.EVALUATE

    def _default_metric_label(self, is_classification: bool) -> str:
        return "F1-score" if is_classification else "RMSE"

    def _score_metadata(self, scoring_name: str, is_classification: bool) -> Dict[str, str]:
        if is_classification:
            if scoring_name == "f1_weighted":
                return {"label": "CV F1", "direction": "higher_is_better"}
            if scoring_name == "precision_weighted":
                return {"label": "CV Precision", "direction": "higher_is_better"}
            if scoring_name == "recall_weighted":
                return {"label": "CV Recall", "direction": "higher_is_better"}
            if scoring_name == "roc_auc_ovr_weighted":
                return {"label": "CV ROC-AUC", "direction": "higher_is_better"}
            return {"label": "CV Accuracy", "direction": "higher_is_better"}

        if scoring_name == "neg_mean_absolute_error":
            return {"label": "CV MAE", "direction": "lower_is_better"}
        if scoring_name == "neg_mean_squared_error":
            return {"label": "CV MSE", "direction": "lower_is_better"}
        if scoring_name == "neg_root_mean_squared_error":
            return {"label": "CV RMSE", "direction": "lower_is_better"}
        return {"label": "CV R² Score", "direction": "higher_is_better"}

    def _score_display_value(
        self, scoring_name: str, raw_score: float, metric_details: Dict[str, Any]
    ) -> float:
        if scoring_name == "neg_mean_absolute_error":
            return float(metric_details.get("mae", abs(raw_score)))
        if scoring_name == "neg_mean_squared_error":
            return float(metric_details.get("mse", abs(raw_score)))
        if scoring_name == "neg_root_mean_squared_error":
            return float(metric_details.get("rmse", abs(raw_score)))
        if scoring_name == "f1_weighted":
            return float(metric_details.get("f1", raw_score * 100.0))
        if scoring_name == "precision_weighted":
            return float(metric_details.get("precision", raw_score * 100.0))
        if scoring_name == "recall_weighted":
            return float(metric_details.get("recall", raw_score * 100.0))
        if scoring_name == "roc_auc_ovr_weighted":
            return float(metric_details.get("roc_auc", raw_score * 100.0))
        if scoring_name == "accuracy":
            return float(metric_details.get("accuracy", raw_score * 100.0))
        if scoring_name == "r2":
            return float(metric_details.get("r2", raw_score))
        return float(raw_score)

    def _package_versions(self) -> Dict[str, Any]:
        versions = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": getattr(pd, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "scikit_learn": getattr(__import__("sklearn"), "__version__", None),
            "xgboost": getattr(__import__("xgboost"), "__version__", None),
            "optuna": getattr(optuna, "__version__", None),
            "shap": getattr(shap, "__version__", None),
            "lightgbm": None,
        }
        try:
            import lightgbm as lgb

            versions["lightgbm"] = getattr(lgb, "__version__", None)
        except Exception:
            versions["lightgbm"] = "not_installed"
        return versions

    def _compute_permutation_importance(self, ctx: PipelineContext) -> Dict[str, float]:
        try:
            from sklearn.inspection import permutation_importance

            sample_X = ctx.X_test.head(min(len(ctx.X_test), 200))
            sample_y = ctx.y_test[: len(sample_X)]
            result = permutation_importance(
                ctx.final_model,
                sample_X,
                sample_y,
                n_repeats=5,
                random_state=42,
                scoring=_resolve_scoring(
                    ctx.config.get("eval_metric", ""), ctx.is_classification
                ),
            )
            ranked = sorted(
                zip(sample_X.columns, result.importances_mean),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            return {str(name): round(float(score), 6) for name, score in ranked[:15]}
        except Exception:
            return {}

    def _primary_metric_label(self, ctx: PipelineContext) -> str:
        scoring_name = _resolve_scoring(
            ctx.config.get("eval_metric", ""), ctx.is_classification
        )
        return self._score_metadata(scoring_name, ctx.is_classification)["label"]

    def _classification_primary_score(self, ctx: PipelineContext, y_pred, scoring_name: str) -> float:
        if scoring_name == "f1_weighted":
            return float(f1_score(ctx.y_test, y_pred, average="weighted", zero_division=0))
        if scoring_name == "precision_weighted":
            return float(
                precision_score(ctx.y_test, y_pred, average="weighted", zero_division=0)
            )
        if scoring_name == "recall_weighted":
            return float(
                recall_score(ctx.y_test, y_pred, average="weighted", zero_division=0)
            )
        if scoring_name == "roc_auc_ovr_weighted":
            try:
                probs = ctx.final_model.predict_proba(ctx.X_test)
                if probs.shape[1] == 2:
                    return float(roc_auc_score(ctx.y_test, probs[:, 1]))
                return float(
                    roc_auc_score(
                        ctx.y_test,
                        probs,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
            except Exception:
                return float(accuracy_score(ctx.y_test, y_pred))
        return float(accuracy_score(ctx.y_test, y_pred))

    def _regression_primary_score(self, y_true, y_pred, scoring_name: str) -> float:
        if scoring_name == "neg_mean_absolute_error":
            return -float(mean_absolute_error(y_true, y_pred))
        if scoring_name == "neg_mean_squared_error":
            return -float(mean_squared_error(y_true, y_pred))
        if scoring_name == "neg_root_mean_squared_error":
            return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return float(r2_score(y_true, y_pred))

    def _classification_metric_bundle(
        self, ctx: PipelineContext, preds
    ) -> Dict[str, Any]:
        probs = None
        try:
            if hasattr(ctx.final_model, "predict_proba"):
                probs = ctx.final_model.predict_proba(ctx.X_test)
        except Exception:
            probs = None

        bundle = _classification_metrics(ctx.y_test, preds, y_proba=probs)
        bundle["confusion_matrix"] = confusion_matrix(ctx.y_test, preds).tolist()
        return bundle

    def _regression_metric_bundle(self, ctx: PipelineContext, preds) -> Dict[str, Any]:
        bundle = _regression_metrics(ctx.y_test, preds)
        mape = None
        try:
            denom = np.where(
                np.abs(np.asarray(ctx.y_test)) < 1e-9,
                np.nan,
                np.asarray(ctx.y_test),
            )
            mape = float(
                np.nanmean(np.abs((np.asarray(ctx.y_test) - np.asarray(preds)) / denom))
                * 100
            )
        except Exception:
            mape = None

        bundle["mape"] = round(mape, 4) if mape is not None and np.isfinite(mape) else None
        return bundle

    def _performance_metrics_payload(
        self, ctx: PipelineContext, scoring_name: str, metric_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        requested_metric = str(
            ctx.config.get("eval_metric")
            or self._default_metric_label(ctx.is_classification)
        )
        score_meta = self._score_metadata(scoring_name, ctx.is_classification)
        return {
            "task_type": "classification" if ctx.is_classification else "regression",
            "optimized_metric": {
                "requested": requested_metric,
                "scoring_name": scoring_name,
                "resolved_display_label": score_meta["label"],
                "score_direction": score_meta["direction"],
                "goal": ctx.goal,
                "mode": ctx.mode,
            },
            "all_metrics": metric_details,
        }

    def _build_validation_summary(
        self,
        scoring_name: str,
        score_meta: Dict[str, str],
        cv_mean_raw: float,
        holdout_raw: float,
        cv_display: float,
        holdout_display: float,
    ) -> Dict[str, Any]:
        direction = score_meta["direction"]
        raw_gap = float(cv_mean_raw - holdout_raw)
        display_gap = float(holdout_display - cv_display)
        abs_display_gap = abs(display_gap)
        baseline_display = max(abs(float(cv_display)), 1.0)
        relative_gap = abs_display_gap / baseline_display

        if direction == "higher_is_better":
            degrade_ratio = max(0.0, float(cv_display - holdout_display)) / max(
                abs(float(cv_display)), 1e-9
            )
            if holdout_display + 1e-12 < cv_display:
                if degrade_ratio >= 0.16 or abs_display_gap >= 6.0 or abs(raw_gap) >= 0.08:
                    status = "possible_overfit"
                elif degrade_ratio >= 0.08 or abs_display_gap >= 3.0 or abs(raw_gap) >= 0.04:
                    status = "watch"
                else:
                    status = "stable"
            elif holdout_display > cv_display + 1e-12:
                status = "holdout_outperformed_cv"
            else:
                status = "stable"
        else:
            degrade_ratio = max(0.0, float(holdout_display - cv_display)) / max(
                abs(float(cv_display)), 1e-9
            )
            if holdout_display > cv_display + 1e-12:
                if degrade_ratio >= 0.2 or abs_display_gap >= max(0.75, baseline_display * 0.2):
                    status = "possible_overfit"
                elif degrade_ratio >= 0.1 or abs_display_gap >= max(0.35, baseline_display * 0.1):
                    status = "watch"
                else:
                    status = "stable"
            elif holdout_display + 1e-12 < cv_display:
                status = "holdout_outperformed_cv"
            else:
                status = "stable"

        if status == "possible_overfit":
            message = (
                f"Holdout performance trails CV materially ({cv_display:.4f} vs {holdout_display:.4f})."
            )
        elif status == "watch":
            message = (
                f"Holdout performance is somewhat weaker than CV ({cv_display:.4f} vs {holdout_display:.4f})."
            )
        elif status == "holdout_outperformed_cv":
            message = (
                f"Holdout performance is stronger than CV ({cv_display:.4f} vs {holdout_display:.4f})."
            )
        else:
            message = (
                f"CV and holdout are closely aligned ({cv_display:.4f} vs {holdout_display:.4f})."
            )

        return {
            "status": status,
            "message": message,
            "score_label": score_meta["label"],
            "holdout_score_label": score_meta["label"].replace("CV ", "Holdout ", 1),
            "score_direction": direction,
            "cv_score_raw": round(float(cv_mean_raw), 6),
            "holdout_score_raw": round(float(holdout_raw), 6),
            "cv_score": round(float(cv_display), 4),
            "holdout_score": round(float(holdout_display), 4),
            "generalization_gap_raw": round(raw_gap, 6),
            "generalization_gap_display": round(display_gap, 4),
            "generalization_gap_ratio": round(float(degrade_ratio), 6),
            "absolute_gap_display": round(abs_display_gap, 4),
            "absolute_gap_ratio": round(float(relative_gap), 6),
        }

    def _build_warning_payload(
        self,
        ctx: PipelineContext,
        cv_std: float,
        validation_summary: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        warnings: list[Dict[str, Any]] = []
        sanitizer = getattr(ctx, "sanitizer_report", {}) or {}
        leakage_frame = ctx.X_train.copy()
        leakage_frame[ctx.target_column] = ctx.y_train.values
        leakage = run_leakage_report(leakage_frame, ctx.target_column)

        if validation_summary.get("status") in {"possible_overfit", "watch"}:
            warnings.append(
                {
                    "type": (
                        "overfitting"
                        if validation_summary.get("status") == "possible_overfit"
                        else "validation_gap"
                    ),
                    "severity": (
                        "high"
                        if validation_summary.get("status") == "possible_overfit"
                        else "medium"
                    ),
                    "message": validation_summary.get("message"),
                    "details": {
                        "score_label": validation_summary.get("score_label"),
                        "holdout_score_label": validation_summary.get(
                            "holdout_score_label"
                        ),
                        "cv_score": validation_summary.get("cv_score"),
                        "holdout_score": validation_summary.get("holdout_score"),
                        "generalization_gap_raw": validation_summary.get(
                            "generalization_gap_raw"
                        ),
                        "generalization_gap_display": validation_summary.get(
                            "generalization_gap_display"
                        ),
                        "generalization_gap_ratio": validation_summary.get(
                            "generalization_gap_ratio"
                        ),
                        "absolute_gap_display": validation_summary.get(
                            "absolute_gap_display"
                        ),
                        "absolute_gap_ratio": validation_summary.get(
                            "absolute_gap_ratio"
                        ),
                    },
                }
            )

        if ctx.is_classification:
            class_counts = pd.Series(ctx.y).value_counts(normalize=True)
            if not class_counts.empty and float(class_counts.max()) >= 0.75:
                warnings.append(
                    {
                        "type": "class_imbalance",
                        "severity": "medium",
                        "message": f"One class represents {round(float(class_counts.max()) * 100, 1)}% of the data.",
                    }
                )
                if _resolve_scoring(
                    ctx.config.get("eval_metric", ""), ctx.is_classification
                ) == "accuracy":
                    warnings.append(
                        {
                            "type": "metric_mismatch",
                            "severity": "medium",
                            "message": "Accuracy can overstate quality on imbalanced classification data; F1 or ROC AUC may be more informative.",
                        }
                    )

        if (sanitizer.get("rows_after") or len(ctx.df)) < 120:
            warnings.append(
                {
                    "type": "too_small_dataset",
                    "severity": "high",
                    "message": "Dataset is small after sanitization; CV variance and overfitting risk are elevated.",
                }
            )

        missing_pct = (
            float(ctx.df.isna().mean().mean() * 100) if len(ctx.df.columns) else 0.0
        )
        if missing_pct >= 20:
            warnings.append(
                {
                    "type": "too_many_missing_values",
                    "severity": "medium",
                    "message": f"Average feature missingness is {round(missing_pct, 1)}%.",
                }
            )

        if cv_std >= 0.04:
            warnings.append(
                {
                    "type": "unstable_cv_scores",
                    "severity": "high" if cv_std >= 0.08 else "medium",
                    "message": f"Cross-validation standard deviation is {round(cv_std * 100, 2)} points.",
                }
            )

        if leakage.get("target_correlated") or leakage.get("future_leakage"):
            warnings.append(
                {
                    "type": "target_leakage",
                    "severity": "high",
                    "message": "Leakage detector found suspiciously predictive or future-looking columns.",
                    "details": {
                        "target_correlated": leakage.get("target_correlated", [])[:10],
                        "future_leakage": leakage.get("future_leakage", [])[:10],
                    },
                }
            )
        return warnings

    def execute(self, ctx: PipelineContext):
        preds = ctx.final_model.predict(ctx.X_test)
        execution_profile = getattr(
            ctx, "execution_profile", None
        ) or TrainingComponent()._execution_profile(ctx)
        sweep_size = execution_profile["sweep_size"]
        cv = TrainingComponent()._build_cv_splitter(ctx, ctx.y_train)
        scoring_name = _resolve_scoring(
            ctx.config.get("eval_metric", ""), ctx.is_classification
        )
        cv_scores = []
        if cv is not None:
            try:
                cv_scores = cross_val_score(
                    ctx.final_model,
                    ctx.X_train,
                    ctx.y_train,
                    cv=cv,
                    scoring=scoring_name,
                )
            except Exception as e:
                ctx.reasoning.append(
                    f"CrossValidation: CV scoring fallback triggered ({e})."
                )
                cv_scores = []

        holdout_score = (
            self._classification_primary_score(ctx, preds, scoring_name)
            if ctx.is_classification
            else self._regression_primary_score(ctx.y_test, preds, scoring_name)
        )
        cv_mean = float(np.mean(cv_scores)) if len(cv_scores) else float(holdout_score)
        cv_std = float(np.std(cv_scores)) if len(cv_scores) else 0.0
        score = cv_mean
        ctx.final_score = score

        # Explainability
        shap_summary = {}
        try:
            X_test_proc = ctx.final_model.preprocess(ctx.X_test)
            underlying_model = ctx.final_model.model_
            explainer = shap.Explainer(underlying_model, X_test_proc)
            shap_vals = explainer(X_test_proc[:50])
            importances = np.abs(shap_vals.values).mean(axis=0)
            if len(importances.shape) > 1:
                importances = importances.mean(axis=1)

            f_names = ctx.final_model.get_feature_names_out()
            for i, f in enumerate(f_names[:8]):
                shap_summary[f.split("__")[-1]] = float(importances[i])
        except Exception as e:
            ctx.reasoning.append(f"Explainability SHAP skipped ({e})")

        ctx.shap_summary = shap_summary
        permutation_summary = self._compute_permutation_importance(ctx)
        metric_details = (
            self._classification_metric_bundle(ctx, preds)
            if ctx.is_classification
            else self._regression_metric_bundle(ctx, preds)
        )
        score_meta = self._score_metadata(scoring_name, ctx.is_classification)
        final_display_score = round(
            self._score_display_value(scoring_name, cv_mean, metric_details), 4
        )
        holdout_display_score = round(
            self._score_display_value(scoring_name, holdout_score, metric_details), 4
        )
        validation_summary = self._build_validation_summary(
            scoring_name=scoring_name,
            score_meta=score_meta,
            cv_mean_raw=cv_mean,
            holdout_raw=float(holdout_score),
            cv_display=final_display_score,
            holdout_display=holdout_display_score,
        )

        ctx.record_history(
            "Final",
            final_display_score,
            phase="cross_validation",
            status="ok",
            metric_name=score_meta["label"],
        )
        lb = [
            {
                "model": ctx.winner_pool_name,
                "score": final_display_score,
                "score_label": score_meta["label"],
                "score_direction": score_meta["direction"],
                "phase": "cross_validation",
                "cv_mean": round(cv_mean * 100, 2),
                "cv_std": round(cv_std * 100, 2),
                "holdout_score": holdout_display_score,
                "holdout_score_label": score_meta["label"].replace("CV ", "Holdout ", 1),
                "validation_status": validation_summary["status"],
                "generalization_gap_display": validation_summary[
                    "generalization_gap_display"
                ],
                "generalization_gap_ratio": validation_summary[
                    "generalization_gap_ratio"
                ],
                "absolute_gap_display": validation_summary["absolute_gap_display"],
            }
        ]

        for r in ctx.sweep_results:
            if r["name"] == ctx.winner_pool_name:
                continue
            row_metric_details = {
                key: r[key]
                for key in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "r2",
                    "mse",
                    "rmse",
                    "mae",
                    "mape",
                )
                if key in r
            }
            row = {
                "model": r["name"],
                "score": round(
                    self._score_display_value(
                        scoring_name, float(r.get("cv_mean_raw", r["score"])), row_metric_details
                    ),
                    4,
                ),
                "score_label": score_meta["label"],
                "score_direction": score_meta["direction"],
                "phase": "cross_validation",
            }
            for k, value in row_metric_details.items():
                row[k] = value
            lb.append(row)
        lb[0].update(metric_details)

        winner_debug = next(
            (r for r in ctx.tested_models if r["model"] == ctx.winner_pool_name), None
        )
        if winner_debug is not None:
            winner_debug["phase"] = "cross_validation"
            winner_debug["score"] = final_display_score
            winner_debug["score_label"] = score_meta["label"]
            winner_debug["score_direction"] = score_meta["direction"]
            winner_debug["holdout_score"] = holdout_display_score
            winner_debug["holdout_score_label"] = score_meta["label"].replace(
                "CV ", "Holdout ", 1
            )
            winner_debug["validation_status"] = validation_summary["status"]
            winner_debug["generalization_gap_display"] = validation_summary[
                "generalization_gap_display"
            ]
            winner_debug["generalization_gap_ratio"] = validation_summary[
                "generalization_gap_ratio"
            ]
            winner_debug["absolute_gap_display"] = validation_summary[
                "absolute_gap_display"
            ]
            winner_debug["winner"] = True
            for key, value in metric_details.items():
                winner_debug[key] = value

        ctx.leaderboard = lb
        warnings_payload = self._build_warning_payload(
            ctx,
            cv_std=cv_std,
            validation_summary=validation_summary,
        )
        leakage_frame = ctx.X_train.copy()
        leakage_frame[ctx.target_column] = ctx.y_train.values

        schema_hash = None
        schema_snapshot = {}
        try:
            schema_path = get_schema_path(ctx.job_id)
            with open(schema_path, "r", encoding="utf-8") as handle:
                schema_snapshot = json.load(handle)
            schema_hash = schema_snapshot.get("hash")
        except Exception:
            schema_snapshot = {}

        with db_session() as db:
            dataset_row = (
                db.query(DatasetModel).filter(DatasetModel.id == ctx.dataset_id).first()
            )
            dataset_snapshot = {
                "parent_dataset_id": dataset_row.parent_dataset_id if dataset_row else None,
                "source_type": dataset_row.source_type if dataset_row else None,
            }

        reproducibility = {
            "job_id": ctx.job_id,
            "dataset_id": ctx.dataset_id,
            "parent_dataset_id": dataset_snapshot["parent_dataset_id"],
            "dataset_source_type": dataset_snapshot["source_type"],
            "schema_hash": schema_hash,
            "selected_features": list(ctx.config.get("selected_features") or []),
            "selected_feature_count": len(ctx.config.get("selected_features") or [])
            or len(ctx.num_cols + ctx.cat_cols),
            "auto_clean": bool(ctx.config.get("auto_clean", True)),
            "handle_imbalance": bool(ctx.config.get("handle_imbalance", False)),
            "cv_folds_used": int(ctx.config.get("cv_folds", 0) or 5),
            "pca_mode": ctx.config.get("pca_mode", "auto"),
            "pca_components_requested": int(ctx.config.get("pca_components", 0) or 0),
            "eval_metric_requested": ctx.config.get("eval_metric")
            or self._default_metric_label(ctx.is_classification),
            "train_test_split_random_state": 42,
            "temporal_validation": bool(ctx.config.get("temporal_validation")),
            "temporal_order_column": ctx.config.get("temporal_order_column") or None,
            "stability_seeds": [42, 123, 999],
            "schema_column_count": len((schema_snapshot.get("schema") or {}).keys()),
            "package_versions": self._package_versions(),
        }

        metadata = {
            "task_type": "classification" if ctx.is_classification else "regression",
            "eval_metric_requested": ctx.config.get("eval_metric")
            or self._default_metric_label(ctx.is_classification),
            "pca_mode": ctx.config.get("pca_mode", "auto"),
            "pca_components_requested": int(ctx.config.get("pca_components", 0) or 0),
            "cv_folds_used": int(ctx.config.get("cv_folds", 0) or 5),
            "temporal_validation": bool(ctx.config.get("temporal_validation")),
            "temporal_order_column": ctx.config.get("temporal_order_column") or None,
            "preprocessor": (
                "full_column_transformer"
                if execution_profile["use_full_preprocessor"]
                else "lite_column_transformer"
            ),
            "feature_names": getattr(ctx, "raw_feature_names", []),
            "derived_feature_names": list(ctx.final_model.get_feature_names_out()),
            "pca_applied": False,
            "reproducibility": reproducibility,
            "task_detection": getattr(ctx, "task_decision", {}),
            "class_labels": list(getattr(ctx, "class_labels", [])),
        }

        pca_components_used = None
        try:
            num_pipeline = ctx.final_model.preprocessor_.named_transformers_.get("num")
            if num_pipeline and "pca" in getattr(num_pipeline, "named_steps", {}):
                pca_components_used = int(num_pipeline.named_steps["pca"].n_components_)
        except Exception:
            pca_components_used = None

        ctx.eda_summary["pca_applied"] = bool(pca_components_used)
        ctx.eda_summary["pca_components_used"] = pca_components_used
        metadata["pca_applied"] = bool(pca_components_used)
        metadata["pca_components_used"] = pca_components_used

        inference_artifact = InferenceArtifact(
            model=ctx.final_model,
            label_encoder=getattr(ctx, "label_encoder", None),
            task_type="classification" if ctx.is_classification else "regression",
            target_name=ctx.target_column,
            raw_feature_names=getattr(ctx, "raw_feature_names", []),
        )

        ModelRegistry.save_model(ctx.job_id, inference_artifact, metadata)

        ctx.metrics = {
            "best_model": ctx.winner_pool_name,
            "metric_name": self._primary_metric_label(ctx),
            "score": final_display_score,
            "cv_mean": round(cv_mean * 100, 2),
            "cv_std": round(cv_std * 100, 2),
            "cv_scores": [round(float(s) * 100, 2) for s in list(cv_scores)],
            "holdout_score": holdout_display_score,
            "holdout_score_label": score_meta["label"].replace("CV ", "Holdout ", 1),
            "leaderboard": lb,
            "is_classification": ctx.is_classification,
            "performance_metrics": self._performance_metrics_payload(
                ctx, scoring_name, metric_details
            ),
            "validation_summary": validation_summary,
            "shap_summary": shap_summary,
            "permutation_importance": permutation_summary,
            "model_path": get_model_path(ctx.job_id),
            "feature_names": getattr(ctx, "raw_feature_names", []),
            "derived_feature_names": list(ctx.final_model.get_feature_names_out()),
            "target": ctx.target_column,
            "eda_summary": ctx.eda_summary,
            "model_metadata": metadata,
            "reasoning": ctx.reasoning,
            "goal": ctx.goal,
            "mode": ctx.mode,
            "execution_profile": execution_profile,
            "tested_models": ctx.tested_models,
            "warnings": warnings_payload,
            "sanitizer_report": getattr(ctx, "sanitizer_report", {}),
            "leakage_report": run_leakage_report(leakage_frame, ctx.target_column),
            "reproducibility_snapshot": reproducibility,
            "task_detection": getattr(ctx, "task_decision", {}),
            "class_labels": list(getattr(ctx, "class_labels", [])),
            "dataset_lineage_snapshot": {
                "dataset_id": ctx.dataset_id,
                "parent_dataset_id": dataset_snapshot["parent_dataset_id"],
                "source_type": dataset_snapshot["source_type"],
            },
        }
        ctx.metrics["summary_text"] = summarize_experiment(ctx.metrics)
        save_metrics(ctx.job_id, ctx.metrics)

        MLTracking.log_run(
            job_id=ctx.job_id,
            params={
                "best_model": ctx.winner_pool_name,
                "mode": ctx.mode,
                "goal": ctx.goal,
                "sweep_size": sweep_size,
                "effective_sweep_rows": execution_profile.get("effective_sweep_rows"),
                "top_k": execution_profile["top_k"],
                "optuna_trials": execution_profile["n_trials"],
                "cv_folds": ctx.config.get("cv_folds", 0),
                "metric_name": ctx.metrics["metric_name"],
            },
            metrics=ctx.metrics,
            model=ctx.final_model,
            artifact_path=get_model_path(ctx.job_id),
        )

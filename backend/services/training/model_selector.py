from typing import Dict, Any, List, Tuple
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

from core.meta_learning import zero_shot_recommend

class ModelSelector:
    """Manages model selection and hyperparameter bounding."""

    @staticmethod
    def _selection_stage(model_name: str) -> int:
        name = str(model_name or "")
        if name in {"Logistic Regression", "Linear Regression"}:
            return 0
        if name in {"Ridge", "ElasticNet"}:
            return 1
        if name == "Random Forest":
            return 2
        if name == "Hist Gradient Boosting":
            return 3
        if name in {"XGBoost", "LightGBM"}:
            return 4
        return 5

    @staticmethod
    def _apply_meta_ranking(
        ordered_names: List[str],
        recommendation: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, Any]]:
        if not ordered_names:
            return ordered_names, {
                "applied": False,
                "confidence": float(recommendation.get("confidence") or 0.0),
                "reordered_models": [],
            }

        confidence = float(recommendation.get("confidence") or 0.0)
        rankings = recommendation.get("rankings") or []
        score_lookup = {}
        for row in rankings:
            try:
                score_lookup[str(row.get("model"))] = float(row.get("pred_score", 0.0))
            except Exception:
                continue

        if confidence < 35.0 or not score_lookup:
            return ordered_names, {
                "applied": False,
                "confidence": confidence,
                "reordered_models": [],
            }

        original_index = {name: idx for idx, name in enumerate(ordered_names)}
        reordered = sorted(
            ordered_names,
            key=lambda name: (
                ModelSelector._selection_stage(name),
                -score_lookup.get(name, float("-inf")),
                original_index[name],
            ),
        )
        changed = [
            name for idx, name in enumerate(reordered)
            if idx < len(ordered_names) and ordered_names[idx] != name
        ]
        return reordered, {
            "applied": reordered != ordered_names,
            "confidence": confidence,
            "reordered_models": changed,
        }

    @staticmethod
    def _selection_goal(goal: str, mode: str = "") -> str:
        normalized_goal = (goal or "Performance").strip()
        normalized_mode = (mode or "").strip()
        if normalized_mode == "Full":
            return "Performance"
        return normalized_goal

    @staticmethod
    def get_goal_profile(goal: str, is_clf: bool, rows: int, mode: str = "") -> Dict[str, Any]:
        goal = ModelSelector._selection_goal(goal, mode)

        if goal == "Speed":
            return {
                "goal": "Speed",
                "max_models": 3,
                "allow_svm": False,
                "preferred_order": None,
            }

        if goal == "Balanced":
            return {
                "goal": "Balanced",
                "max_models": 5,
                "allow_svm": False,
                "preferred_order": None,
            }

        return {
            "goal": "Performance",
            "max_models": 8,
            "allow_svm": bool(is_clf and rows < 15000),
            "preferred_order": None,
        }
    
    @staticmethod
    def get_cheap_config(model_name: str, is_clf: bool) -> Dict[str, Any]:
        """Returns 'Stage 1' parameters for rapid family evaluation."""
        if "Forest" in model_name or "Extra Trees" in model_name:
            return {"n_estimators": 20, "max_depth": 5}
        if "KNN" in model_name:
            return {"n_neighbors": 7, "weights": "distance"}
        if "Hist Gradient Boosting" in model_name:
            return {"max_iter": 50, "learning_rate": 0.08, "max_leaf_nodes": 15}
        if "Boosting" in model_name or "XGB" in model_name or "LGBM" in model_name or "LightGBM" in model_name:
            return {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1}
        if "ElasticNet" in model_name:
            return {"alpha": 0.1, "l1_ratio": 0.5}
        if "MLP" in model_name:
            return {
                "hidden_layer_sizes": (32,),
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "max_iter": 180,
            }
        if "SVM" in model_name:
            return {"C": 0.1, "max_iter": 500}
        return {}

    @staticmethod
    def get_tuning_budget(
        model_name: str,
        base_trials: int,
        base_timeout: int,
        traits: Dict[str, Any] | None = None,
    ) -> Dict[str, int]:
        traits = traits or {}
        name = str(model_name or "")
        budget_map = {
            "Logistic Regression": 0.35,
            "Linear Regression": 0.2,
            "Ridge": 0.3,
            "ElasticNet": 0.45,
            "Random Forest": 0.5,
            "Extra Trees": 0.45,
            "Hist Gradient Boosting": 0.7,
            "XGBoost": 1.0,
            "LightGBM": 0.95,
            "SVM": 0.55,
            "KNN": 0.35,
            "MLP": 0.8,
        }
        multiplier = budget_map.get(name, 0.5)
        if traits.get("low_complexity"):
            multiplier *= 0.8
        if name in {"XGBoost", "LightGBM", "MLP"} and traits.get("high_dimensional"):
            multiplier *= 1.1
        trials = max(4, int(round(base_trials * multiplier))) if base_trials else 0
        timeout = max(30, int(round(base_timeout * multiplier))) if base_timeout else 0
        return {"trials": trials, "timeout": timeout}

    @staticmethod
    def get_bayesian_space(trial, model_name: str) -> Dict[str, Any]:
        """Returns Optuna suggestion space based on model string."""
        if "Forest" in model_name or "Extra Trees" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "max_depth": trial.suggest_int("max_depth", 5, 12)
            }
        elif "KNN" in model_name:
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            }
        elif "XGB" in model_name or "LGBM" in model_name or "LightGBM" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8)
            }
        elif "Hist Gradient Boosting" in model_name:
            return {
                "max_iter": trial.suggest_int("max_iter", 60, 220),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True),
            }
        elif "ElasticNet" in model_name:
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
            }
        elif "MLP" in model_name:
            hidden_units = trial.suggest_int("hidden_units", 16, 128, step=16)
            return {
                "hidden_layer_sizes": (hidden_units,),
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True),
            }
        elif "SVM" in model_name:
            return {
                "C": trial.suggest_float("C", 0.1, 10.0, log=True)
            }
        return {}

    @staticmethod
    def _dataset_traits(rows: int, profile: Dict[str, Any]) -> Dict[str, Any]:
        num_cols = list(profile.get("num_cols") or [])
        cat_cols = list(profile.get("cat_cols") or [])
        cols = int(
            profile.get("cols")
            or len(num_cols)
            + len(cat_cols)
            or 0
        )
        has_mixed_types = bool(num_cols and cat_cols)
        high_dimensional = bool(
            cols >= 100 or (rows > 0 and cols >= 40 and (cols / max(rows, 1)) >= 0.15)
        )
        target_entropy = float(profile.get("target_entropy") or 0.0)
        numeric_max_corr = float(profile.get("numeric_max_corr") or 0.0)
        row_factor = min(max(rows, 0) / 50000.0, 1.0)
        feature_factor = min(max(cols, 0) / 120.0, 1.0)
        entropy_factor = min(max(target_entropy, 0.0), 1.0)
        corr_factor = min(max(numeric_max_corr, 0.0), 1.0)
        complexity_score = round(
            (0.3 * row_factor)
            + (0.25 * feature_factor)
            + (0.25 * entropy_factor)
            + (0.2 * corr_factor),
            3,
        )
        return {
            "rows": rows,
            "cols": cols,
            "small_dataset": rows < 5000,
            "large_dataset": rows >= 20000,
            "very_large_dataset": rows >= 100000,
            "high_dimensional": high_dimensional,
            "mixed_types": has_mixed_types,
            "complex_patterns": bool(has_mixed_types or cols >= 12 or rows >= 2000),
            "simple_dataset": bool(rows < 5000 and cols <= 12 and not has_mixed_types),
            "knn_allowed": bool(cols <= 50),
            "mlp_allowed": bool(rows >= 2000 and rows < 100000),
            "target_entropy": target_entropy,
            "numeric_max_corr": numeric_max_corr,
            "complexity_score": complexity_score,
            "low_complexity": bool(complexity_score < 0.32),
        }

    @staticmethod
    def _booster_candidates() -> List[str]:
        candidates = ["Hist Gradient Boosting"]
        if LGBMClassifier is not None and LGBMRegressor is not None:
            candidates.append("LightGBM")
        candidates.append("XGBoost")
        return candidates

    @staticmethod
    def _limit_boosters(candidates: List[str], traits: Dict[str, Any], goal: str) -> List[str]:
        if not candidates:
            return []

        if goal == "Balanced":
            preferred = [name for name in ["Hist Gradient Boosting"] if name in candidates]
            return preferred[:1]

        if goal == "Speed":
            preferred = [name for name in ["Hist Gradient Boosting"] if name in candidates]
            return preferred[:1]

        if traits.get("low_complexity"):
            preferred = [name for name in ["Hist Gradient Boosting"] if name in candidates]
            return preferred[:1]

        preferred = [name for name in ["Hist Gradient Boosting", "XGBoost", "LightGBM"] if name in candidates]
        max_boosters = 3
        return preferred[:max_boosters]

    @staticmethod
    def _build_preferred_order(goal: str, is_clf: bool, traits: Dict[str, Any]) -> List[str]:
        boosters = ModelSelector._limit_boosters(
            ModelSelector._booster_candidates(), traits, goal
        )

        if goal == "Speed":
            if is_clf:
                return [
                    "Logistic Regression",
                    "Random Forest",
                    "Hist Gradient Boosting",
                ]
            return [
                "Linear Regression",
                "Ridge",
                "Random Forest",
                "Hist Gradient Boosting",
            ]

        if goal == "Balanced":
            if is_clf:
                order = ["Logistic Regression", "Random Forest", "Hist Gradient Boosting", "KNN"]
                if not traits["high_dimensional"] and not traits["large_dataset"]:
                    order.append("Extra Trees")
                return order[:5]

            return ["Linear Regression", "Ridge", "ElasticNet", "Random Forest", "Hist Gradient Boosting"]

        if is_clf:
            order = ["Logistic Regression", "Random Forest", *boosters]
            if traits["complex_patterns"] and traits["mlp_allowed"]:
                order.append("MLP")
            if traits["small_dataset"] and not traits["high_dimensional"] and traits["knn_allowed"]:
                order.append("KNN")
            if traits["small_dataset"] and not traits["very_large_dataset"]:
                order.append("SVM")
            return order

        order = ["Linear Regression", "Ridge", "ElasticNet", "Random Forest", *boosters]
        if traits["complex_patterns"] and traits["mlp_allowed"]:
            order.append("MLP")
        return order
    
    @staticmethod
    def select_pool(
        rows: int,
        is_clf: bool,
        goal: str,
        profile: Dict[str, Any],
        mode: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uses meta-learning to rank and select the model pool."""
        goal_profile = ModelSelector.get_goal_profile(goal, is_clf, rows, mode=mode)
        traits = ModelSelector._dataset_traits(rows, profile)

        pool = {
            "Logistic Regression" if is_clf else "Linear Regression": 
                LogisticRegression(max_iter=1000) if is_clf else LinearRegression(),
            "Random Forest": RandomForestClassifier() if is_clf else RandomForestRegressor(),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42) if is_clf else HistGradientBoostingRegressor(random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss') if is_clf else XGBRegressor(),
        }
        if LGBMClassifier is not None and LGBMRegressor is not None:
            pool["LightGBM"] = LGBMClassifier(verbose=-1) if is_clf else LGBMRegressor(verbose=-1)
        if is_clf and goal_profile["goal"] == "Balanced" and not traits["high_dimensional"] and not traits["large_dataset"]:
            pool["Extra Trees"] = ExtraTreesClassifier(random_state=42)
        
        # Regression specific
        if not is_clf:
            pool["Ridge"] = Ridge()
            pool["ElasticNet"] = ElasticNet(random_state=42)
            if goal_profile["goal"] == "Performance" and traits["complex_patterns"] and traits["mlp_allowed"] and not traits["very_large_dataset"]:
                pool["MLP"] = MLPRegressor(
                    hidden_layer_sizes=(64, 32) if rows >= 8000 else (48,),
                    max_iter=250,
                    early_stopping=bool(rows >= 500),
                    random_state=42,
                )
        
        # Classification specific (speed/sample check)
        if is_clf:
            if (goal_profile["goal"] == "Balanced" and not traits["high_dimensional"] and not traits["large_dataset"]) or (
                goal_profile["goal"] == "Performance" and traits["small_dataset"] and not traits["high_dimensional"]
            ):
                if traits["knn_allowed"]:
                    pool["KNN"] = KNeighborsClassifier()
            if goal_profile["goal"] == "Performance" and traits["complex_patterns"] and traits["mlp_allowed"]:
                pool["MLP"] = MLPClassifier(
                    hidden_layer_sizes=(64, 32) if rows >= 8000 else (48,),
                    max_iter=250,
                    early_stopping=bool(rows >= 500),
                    random_state=42,
                )
        if goal_profile["allow_svm"] and not traits["very_large_dataset"]:
            pool["SVM"] = SVC(probability=True)

        pool = {k: v for k, v in pool.items() if v is not None}
        preferred_order = ModelSelector._build_preferred_order(
            goal_profile["goal"], is_clf, traits
        )
        if preferred_order:
            pool = {name: pool[name] for name in preferred_order if name in pool}
        max_models = int(goal_profile.get("max_models") or len(pool))
        if max_models > 0:
            pool = dict(list(pool.items())[:max_models])

        # Meta-Learning Ranking
        candidate_names = list(pool.keys())
        recommendation = zero_shot_recommend(profile, candidate_names)
        reordered_names, memory_signal = ModelSelector._apply_meta_ranking(
            candidate_names,
            recommendation,
        )
        if reordered_names:
            pool = {name: pool[name] for name in reordered_names if name in pool}

        ordered_pool = dict(pool)

        recommendation = dict(recommendation)
        recommendation["goal_profile"] = {
            "requested_goal": (goal or "Performance").strip(),
            "goal": goal_profile["goal"],
            "mode": (mode or "").strip() or "Balanced",
            "models_selected": list(ordered_pool.keys()),
            "dataset_traits": {
                **traits,
                "memory_signal": memory_signal,
            },
        }
        recommendation["memory_signal"] = memory_signal
        recommendation["advisory_only"] = not memory_signal.get("applied", False)

        return ordered_pool, recommendation

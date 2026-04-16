from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
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
    def get_goal_profile(goal: str, is_clf: bool, rows: int) -> Dict[str, Any]:
        goal = (goal or "Performance").strip()

        if goal == "Speed":
            return {
                "goal": "Speed",
                "max_models": 3,
                "allow_svm": False,
                "preferred_order": (
                    [name for name in ["Logistic Regression", "Random Forest", "LightGBM"] if name != "LightGBM" or LGBMClassifier is not None]
                    if is_clf
                    else [name for name in ["Linear Regression", "Ridge", "LightGBM"] if name != "LightGBM" or LGBMRegressor is not None]
                ),
            }

        if goal == "Balanced":
            return {
                "goal": "Balanced",
                "max_models": 4 if is_clf else 5,
                "allow_svm": bool(is_clf and rows < 5000),
                "preferred_order": (
                    [name for name in ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"] if name != "LightGBM" or LGBMClassifier is not None]
                    if is_clf
                    else [name for name in ["Linear Regression", "Ridge", "Random Forest", "LightGBM", "XGBoost"] if name != "LightGBM" or LGBMRegressor is not None]
                ),
            }

        return {
            "goal": "Performance",
            "max_models": 6 if is_clf else 6,
            "allow_svm": bool(is_clf and rows < 15000),
            "preferred_order": None,
        }
    
    @staticmethod
    def get_cheap_config(model_name: str, is_clf: bool) -> Dict[str, Any]:
        """Returns 'Stage 1' parameters for rapid family evaluation."""
        if "Forest" in model_name:
            return {"n_estimators": 20, "max_depth": 5}
        if "Boosting" in model_name or "XGB" in model_name or "LGBM" in model_name or "LightGBM" in model_name:
            return {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1}
        if "SVM" in model_name:
            return {"C": 0.1, "max_iter": 500}
        return {}

    @staticmethod
    def get_bayesian_space(trial, model_name: str) -> Dict[str, Any]:
        """Returns Optuna suggestion space based on model string."""
        if "Forest" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "max_depth": trial.suggest_int("max_depth", 5, 12)
            }
        elif "XGB" in model_name or "LGBM" in model_name or "LightGBM" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8)
            }
        elif "SVM" in model_name:
            return {
                "C": trial.suggest_float("C", 0.1, 10.0, log=True)
            }
        return {}
    
    @staticmethod
    def select_pool(rows: int, is_clf: bool, goal: str, profile: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uses meta-learning to rank and select the model pool."""
        goal_profile = ModelSelector.get_goal_profile(goal, is_clf, rows)

        pool = {
            "Logistic Regression" if is_clf else "Linear Regression": 
                LogisticRegression(max_iter=1000) if is_clf else LinearRegression(),
            "Random Forest": RandomForestClassifier() if is_clf else RandomForestRegressor(),
            "XGBoost": XGBClassifier(eval_metric='logloss') if is_clf else XGBRegressor(),
        }
        if LGBMClassifier is not None and LGBMRegressor is not None:
            pool["LightGBM"] = LGBMClassifier(verbose=-1) if is_clf else LGBMRegressor(verbose=-1)
        
        # Regression specific
        if not is_clf:
            pool["Ridge"] = Ridge()
            pool["Lasso"] = Lasso()
        
        # Classification specific (speed/sample check)
        if goal_profile["allow_svm"]:
            pool["SVM"] = SVC(probability=True)

        pool = {k: v for k, v in pool.items() if v is not None}

        # Meta-Learning Ranking
        candidate_names = list(pool.keys())
        recommendation = zero_shot_recommend(profile, candidate_names)

        ranked_names = [
            entry["model"]
            for entry in recommendation.get("rankings", [])
            if entry.get("model") in pool
        ]

        preferred_order = goal_profile.get("preferred_order") or []
        if preferred_order:
            preferred_names = [name for name in preferred_order if name in pool]
            remaining_ranked = [name for name in ranked_names if name not in preferred_names]
            ranked_names = preferred_names + remaining_ranked

        if not ranked_names:
            ranked_names = list(pool.keys())

        deduped_names = []
        for name in ranked_names:
            if name not in deduped_names:
                deduped_names.append(name)

        selected_names = deduped_names[:goal_profile["max_models"]]
        ordered_pool = {name: pool[name] for name in selected_names}

        recommendation = dict(recommendation)
        recommendation["goal_profile"] = {
            "goal": goal_profile["goal"],
            "models_selected": list(ordered_pool.keys()),
        }

        return ordered_pool, recommendation

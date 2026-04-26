"""
core/config.py — Centralized application configuration.

Reads from .env via pydantic-settings. Import `settings` anywhere.
Legacy CONFIG dict is preserved for backward compatibility.
"""
import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore
    except ImportError:
        BaseSettings = object  # type: ignore


# ── Legacy dict (backward compat) ────────────────────────────────────────────
CONFIG = {
    "small_data_rows": 500,
    "high_missing": 10.0,
    "overfit_diff": 0.10,
    "max_confidence": 0.95,
}


class Settings:
    """Simple settings class that reads from environment / .env."""

    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv(
                dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"),
                override=False
            )
        except Exception:
            pass

        # Server
        self.host: str = os.getenv("HOST", "0.0.0.0")

        try:
            self.port: int = int(os.getenv("PORT", "8000"))
        except Exception:
            self.port = 8000

        self.debug: bool = str(os.getenv("DEBUG", "false")).lower() == "true"

        # Storage
        self.runs_dir: str = os.getenv("RUNS_DIR", "runs")
        self.tmp_dir: str = os.getenv("TMP_DIR", "tmp")

        try:
            self.max_run_age_days: int = int(os.getenv("MAX_RUN_AGE_DAYS", "7"))
        except Exception:
            self.max_run_age_days = 7

        try:
            self.max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "500"))
        except Exception:
            self.max_upload_mb = 500
        self.allow_pickle_uploads: bool = str(
            os.getenv("ALLOW_PICKLE_UPLOADS", "false")
        ).lower() == "true"

        # Redis / Celery
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.celery_broker: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        self.celery_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

        # Database
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./automl_studio.db")

        # AI / LLM
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

        # Training
        try:
            self.default_cv_folds: int = int(os.getenv("DEFAULT_CV_FOLDS", "3"))
        except Exception:
            self.default_cv_folds = 3

        try:
            self.default_test_size: float = float(os.getenv("DEFAULT_TEST_SIZE", "0.2"))
        except Exception:
            self.default_test_size = 0.2

        try:
            self.max_optuna_trials: int = int(os.getenv("MAX_OPTUNA_TRIALS", "30"))
        except Exception:
            self.max_optuna_trials = 30

        try:
            self.max_features_budget: int = int(os.getenv("MAX_FEATURES_BUDGET", "100"))
        except Exception:
            self.max_features_budget = 100

        # Leakage thresholds
        try:
            self.leakage_corr_threshold: float = float(os.getenv("LEAKAGE_CORR_THRESHOLD", "0.98"))
        except Exception:
            self.leakage_corr_threshold = 0.98

        try:
            self.near_constant_threshold: float = float(os.getenv("NEAR_CONSTANT_THRESHOLD", "0.99"))
        except Exception:
            self.near_constant_threshold = 0.99

        # Drift thresholds
        try:
            self.psi_warning_threshold: float = float(os.getenv("PSI_WARNING_THRESHOLD", "0.1"))
        except Exception:
            self.psi_warning_threshold = 0.1

        try:
            self.psi_critical_threshold: float = float(os.getenv("PSI_CRITICAL_THRESHOLD", "0.2"))
        except Exception:
            self.psi_critical_threshold = 0.2

        try:
            self.ks_p_threshold: float = float(os.getenv("KS_P_THRESHOLD", "0.05"))
        except Exception:
            self.ks_p_threshold = 0.05


# Singleton
settings = Settings()

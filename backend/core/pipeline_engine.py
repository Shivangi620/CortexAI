import time
import traceback
import json
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd

from infra.database import get_db, JobModel
from infra.logger import get_logger
from infra.result_contract import sanitize_for_json

log = get_logger(__name__)


class SyncedList(list):
    """List wrapper that pushes updates whenever it changes."""

    def __init__(self, values=None, on_change=None):
        super().__init__(values or [])
        self._on_change = on_change

    def _notify(self):
        if callable(self._on_change):
            self._on_change()

    def append(self, item):
        super().append(item)
        self._notify()

    def extend(self, items):
        super().extend(items)
        self._notify()

    def clear(self):
        super().clear()
        self._notify()

class PipelineStep(str, Enum):
    INITIALIZE = "Initializing"
    VALIDATE = "Validating Data"
    PROFILE = "Profiling & Cleaning"
    FEATURE_ENG = "Feature Engineering"
    TRAIN = "Model Training"
    EVALUATE = "Evaluation & Extraction"
    COMPLETE = "Completed"
    FAILED = "Failed"


class PipelineContext:
    """Holds intermediate state and configuration for the entire pipeline run."""
    def __init__(self, job_id: str, dataset_id: str, file_path: str, target_column: str, goal: str, mode: str, config: Dict[str, Any]):
        self.job_id = job_id
        self.dataset_id = dataset_id
        self.file_path = file_path
        self.target_column = target_column
        self.goal = goal
        self.mode = mode
        self.config = config
        
        # Intermediate Data State
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        
        # Meta State
        self.is_classification: bool = True
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.preprocessor: Any = None
        self.health_metadata: Dict[str, Any] = {}
        
        # Execution Results
        self.model_pool: Dict[str, Any] = {}
        self.sweep_results: List[Dict[str, Any]] = []
        self.final_model: Any = None
        self.winner_pool_name: str = ""
        self.final_score: float = 0.0
        self.metrics: Dict[str, Any] = {}
        self.leaderboard: List[Dict[str, Any]] = []
        self.shap_summary: Dict[str, Any] = {}
        self.reasoning: List[str] = []
        self.eda_summary: Dict[str, Any] = {}
        self.tested_models: List[Dict[str, Any]] = []
        self._history_callback = None

    def record_history(self, label: str, metric: Any, **extra):
        if callable(self._history_callback):
            payload = {"time": label, "metric": metric}
            payload.update(extra or {})
            self._history_callback(payload)


class PipelineComponent:
    """Abstract base class for a pipeline step."""
    def get_name(self) -> str:
        return self.__class__.__name__

    def get_step_type(self) -> PipelineStep:
        raise NotImplementedError()

    def execute(self, ctx: PipelineContext):
        """Perform operations and mutate `ctx`."""
        raise NotImplementedError()


class PipelineEngine:
    """Orchestrates pipeline components with proper checkpointing and tracking."""

    def __init__(self, context: PipelineContext, components: List[PipelineComponent]):
        self.ctx = context
        self.components = components
        self.ctx.reasoning = SyncedList(self.ctx.reasoning, self._sync_reasoning)
        self.ctx._history_callback = self._append_history

    def log(self, message: str):
        log.info(f"[Pipeline][{self.ctx.job_id}] {message}")

        if not isinstance(self.ctx.reasoning, list):
            self.ctx.reasoning = []

        self.ctx.reasoning.append(message)

        try:
            self._sync_reasoning()
        except Exception as e:
            log.warning(f"Reasoning sync failed: {e}")

    def _sync_reasoning(self):
        try:
          with get_db() as db:
            job = db.query(JobModel).filter(JobModel.id == self.ctx.job_id).first()
            if job:
                reasoning = self.ctx.reasoning if isinstance(self.ctx.reasoning, list) else []

                try:
                    job.reasoning_json = json.dumps(reasoning)
                except Exception:
                    job.reasoning_json = json.dumps([str(r) for r in reasoning])

                db.commit()
        except Exception as e:
          log.warning(f"Reasoning sync failed: {e}")

    def _append_history(self, entry: Dict[str, Any]):
        try:
            with get_db() as db:
                job = db.query(JobModel).filter(JobModel.id == self.ctx.job_id).first()
                if not job:
                    return

                try:
                    history = json.loads(job.history_json) if job.history_json else []
                except Exception:
                    history = []

                history.append(sanitize_for_json(entry))
                job.history_json = json.dumps(sanitize_for_json(history), allow_nan=False)
                db.commit()
        except Exception as e:
            log.warning(f"History sync failed: {e}")

    def update_status(self, step: PipelineStep):
        self.log(f"Status update: {step.value}")

        try:
            with get_db() as db:
                job = db.query(JobModel).filter(JobModel.id == self.ctx.job_id).first()
                if job:
                    job.status = "training"
                    db.commit()

            self._append_history({"time": "Pipeline", "metric": step.value, "kind": "stage"})

        except Exception as e:
            log.warning(f"Status update failed: {e}")

    def mark_failed(self, error_msg: str):
        try:
          with get_db() as db:
            job = db.query(JobModel).filter(JobModel.id == self.ctx.job_id).first()
            if job:
                job.status = "failed"
                job.error = error_msg

                reasoning = self.ctx.reasoning if isinstance(self.ctx.reasoning, list) else []

                try:
                    job.reasoning_json = json.dumps(reasoning)
                except Exception:
                    job.reasoning_json = json.dumps([str(r) for r in reasoning])

                db.commit()
        except Exception as e:
          log.warning(f"Failed to mark job as failed: {e}")
          
    def run(self):
        start_time = time.time()
        comp_name = "Unknown"

        self.log("Pipeline Engine Intialized. Starting execution.")

        try:
            for component in self.components:
                step_type = component.get_step_type()
                comp_name = component.get_name()

                self.update_status(step_type)
                self.log(f"Running Component: {comp_name}")

                try:
                    component.execute(self.ctx)
                except Exception as e:
                    raise Exception(f"{comp_name} failed: {str(e)}") from e

            self.update_status(PipelineStep.COMPLETE)
            duration = time.time() - start_time
            self.log(f"Pipeline executed successfully in {duration:.2f}s.")

            return self.ctx

        except Exception as e:
            err_msg = f"Pipeline Failed in {comp_name}: {str(e)}\n{traceback.format_exc()}"
            self.log(err_msg)
            self.mark_failed(str(e))
            raise e

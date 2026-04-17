import os
import json
from datetime import datetime

# Optional imports
try:
    import mlflow
except Exception:
    mlflow = None

try:
    import wandb
except Exception:
    wandb = None


class MLTracking:
    """Handles integration with MLflow and Weights & Biases."""

    @staticmethod
    def init_mlflow(experiment_name="AutoML_Studio"):
        # ✅ FIX 1: guard if mlflow not installed
        if not mlflow:
            return

        try:
            if os.getenv("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"MLflow init failed: {e}")  # ✅ FIX 2


    @staticmethod
    def log_run(job_id, params, metrics, model=None, artifact_path=None):
        """Logs a single run to both MLflow and W&B if configured."""

        # ✅ FIX 3: ensure safe defaults
        params = params or {}
        metrics = metrics or {}
        flat_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        # MLflow
        if mlflow and os.getenv("MLFLOW_ENABLED", "false").lower() == "true":
            try:
                with mlflow.start_run(run_name=f"Job_{job_id}"):
                    if params:
                        mlflow.log_params(params)
                    if flat_metrics:
                        mlflow.log_metrics(flat_metrics)

                    # ✅ FIX 4: guard sklearn submodule
                    if model and hasattr(mlflow, "sklearn"):
                        try:
                            mlflow.sklearn.log_model(model, "model")
                        except Exception:
                            pass  # don't break run

                    if artifact_path and os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path)

            except Exception as e:
                print(f"MLflow logging failed: {e}")

        # W&B
        if wandb and os.getenv("WANDB_API_KEY"):
            try:
                wandb.init(
                    project="AutoML_Studio",
                    name=f"Job_{job_id}",
                    config=params,
                    reinit=True
                )

                if metrics:
                    wandb.log(flat_metrics)

                # ✅ FIX 5: safe artifact handling
                if artifact_path and os.path.exists(artifact_path):
                    try:
                        wandb.save(artifact_path)
                    except Exception:
                        pass

                wandb.finish()

            except Exception as e:
                print(f"W&B logging failed: {e}")

        # Local experiment tracking DB
        try:
            from infra.database import get_db, JobModel, ExperimentRun, DatasetModel

            with get_db() as db:
                job = db.query(JobModel).filter(JobModel.id == job_id).first()
                if job:
                    dataset = (
                        db.query(DatasetModel)
                        .filter(DatasetModel.id == job.dataset_id)
                        .first()
                    )
                    try:
                        job_params = json.loads(job.params_json) if job.params_json else {}
                    except Exception:
                        job_params = {}

                    score = (
                        metrics.get("score")
                        if metrics.get("score") is not None
                        else flat_metrics.get("test_score")
                    )
                    metric_name = (
                        metrics.get("metric_name")
                        or params.get("metric_name")
                        or job_params.get("eval_metric")
                        or "Score"
                    )
                    best_model = (
                        metrics.get("best_model")
                        or params.get("best_model")
                        or ""
                    )

                    run = ExperimentRun(
                        job_id=job_id,
                        dataset_id=job.dataset_id,
                        dataset_name=(dataset.display_name if dataset else None),
                        workspace_id=str(job_params.get("workspace_id") or "") or None,
                        workspace_name=str(job_params.get("workspace_name") or "") or None,
                        model_name=str(best_model) if best_model else None,
                        metric_name=str(metric_name),
                        score=str(score) if score is not None else None,
                        hyperparams_json=json.dumps(params),
                        metrics_json=json.dumps(metrics),
                        leaderboard_json=json.dumps(metrics.get("leaderboard", [])),
                        feature_count=str(len(metrics.get("feature_names") or [])),
                        row_count=str((metrics.get("eda_summary") or {}).get("rows_after_target_cleaning", "")),
                        task_type="classification" if metrics.get("is_classification") else "regression",
                        mode=str(params.get("mode") or job_params.get("mode") or ""),
                        goal=str(params.get("goal") or job_params.get("goal") or ""),
                        preset_name=str(job_params.get("preset_name") or "") or None,
                        summary_text=str(metrics.get("summary_text") or "") or None,
                    )
                    db.add(run)
        except Exception as e:
            print(f"Experiment tracking DB write failed: {e}")


class StructuredLogger:
    @staticmethod
    def log(event, **kwargs):
        # ✅ FIX 6: safe timestamp + formatting
        try:
            timestamp = datetime.utcnow().isoformat()
        except Exception:
            timestamp = "unknown_time"

        try:
            data = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        except Exception:
            data = ""

        print(f"[{timestamp}] EVENT={event} | {data}")

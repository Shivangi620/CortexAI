import os
import json
import csv
from celery import Celery

from infra.database import db_session, JobModel, DatasetModel, NotificationModel, WorkspaceModel, ExperimentRun
from infra.logger import get_logger
from infra.result_contract import normalize_results

log = get_logger(__name__)
from core.insights import generate_insights, generate_story
from core.pipeline_engine import PipelineEngine, PipelineContext
from services.training.components import (
    DataValidationComponent,
    FeatureEngineeringComponent,
    ModelSelectionComponent,
    TrainingComponent,
    EvaluationComponent
)

# ── CSV Field Size Limit ──────────────────────────────────────────────────────
csv.field_size_limit(int(1e9))

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "automl_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)
celery_app.conf.broker_connection_retry_on_startup = True

@celery_app.task(bind=True, max_retries=0)
def run_training_job(
    self, job_id, dataset_id, file_path, target_column, goal, mode,
    task_type="",
    eval_metric="Performance",
    selected_features=None,
    handle_imbalance=False,
    auto_clean=True,
    cv_folds=0,
    pca_mode="auto",
    pca_components=0,
):
    """
    Celery task: runs training in background using the Modular Component Pipeline Engine.
    """
    # Fetch health metadata / profile to prep context
    profile_data = {}
    health_metadata = {}
    try:
        with db_session() as db:
            ds = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
            if ds and ds.profile_json:
                try:
                    profile_data = json.loads(ds.profile_json)
                    health_metadata = profile_data.get("health", {})
                except Exception:
                    profile_data = {}
                    health_metadata = {}
    except Exception:
        profile_data = {}
        health_metadata = {}

    config = {
        "task_type": task_type,
        "eval_metric": eval_metric,
        "selected_features": selected_features,
        "handle_imbalance": handle_imbalance,
        "auto_clean": auto_clean,
        "cv_folds": cv_folds or 5,
        "pca_mode": pca_mode,
        "pca_components": pca_components,
    }

    ctx = PipelineContext(
        job_id=job_id,
        dataset_id=dataset_id,
        file_path=file_path,
        target_column=target_column,
        goal=goal,
        mode=mode,
        config=config
    )
    ctx.health_metadata = health_metadata

    components = [
        DataValidationComponent(),
        FeatureEngineeringComponent(),
        ModelSelectionComponent(),
        TrainingComponent(),
        EvaluationComponent()
    ]

    engine = PipelineEngine(context=ctx, components=components)

    try:
        final_ctx = engine.run()
        results = normalize_results(final_ctx.metrics or {})

        try:
            insights = generate_insights(profile_data or {}, results)
            story = generate_story(profile_data or {}, results)
        except Exception as e:
            log.warning(f"Insights/story generation failed: {e}", exc_info=True)
            insights = {}
            story = None

        try:
            from core.meta_learning import save_meta_record

            save_meta_record(profile_data or {}, results or {})
        except Exception as e:
            log.warning(f"Meta-learning save skipped: {e}")

        try:
            with db_session() as db:
                job = db.query(JobModel).filter(JobModel.id == job_id).first()
                if job:
                    job.status = "completed"

                    try:
                        job.results_json = json.dumps(results)
                    except Exception:
                        job.results_json = json.dumps({})

                    try:
                        job.insights_json = json.dumps(insights)
                    except Exception:
                        job.insights_json = json.dumps({})

                    job.story = story
                    job.model_path = results.get("model_path") if isinstance(results, dict) else None

                    reasoning = final_ctx.reasoning if isinstance(final_ctx.reasoning, list) else []
                    try:
                        job.reasoning_json = json.dumps(reasoning)
                    except Exception:
                        job.reasoning_json = json.dumps([str(r) for r in reasoning])

                    db.commit()
                    try:
                        params = json.loads(job.params_json) if job.params_json else {}
                    except Exception:
                        params = {}

                    workspace_id = params.get("workspace_id")
                    workspace_name = params.get("workspace_name")
                    if workspace_id or workspace_name:
                        workspace = None
                        if workspace_id:
                            workspace = db.query(WorkspaceModel).filter(
                                WorkspaceModel.id == workspace_id,
                            ).first()
                        if not workspace and workspace_name:
                            workspace = db.query(WorkspaceModel).filter(
                                WorkspaceModel.name == workspace_name,
                            ).first()
                        if workspace:
                            workspace.dataset_id = dataset_id
                            workspace.last_job_id = job_id
                            workspace.settings_json = json.dumps(params)

                    latest_run = (
                        db.query(ExperimentRun)
                        .filter(ExperimentRun.job_id == job_id)
                        .order_by(ExperimentRun.created_at.desc())
                        .first()
                    )
                    if workspace_id and latest_run:
                        workspace = db.query(WorkspaceModel).filter(
                            WorkspaceModel.id == workspace_id,
                        ).first()
                        if workspace:
                            workspace.last_run_id = latest_run.id

                    db.add(
                        NotificationModel(
                            entity_type="job",
                            entity_id=job_id,
                            title="Training Completed",
                            message=(results.get("summary_text") if isinstance(results, dict) else None) or f"Run {job_id[:8]} completed successfully.",
                            level="success",
                        )
                    )
                    db.commit()
        except Exception as e:
            log.warning(f"Final DB write failed: {e}", exc_info=True)

    except Exception:
        raise

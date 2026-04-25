"""
Compatibility wrapper around the production pipeline engine.

The legacy `core.pipeline.PipelineEngine` used to run a separate training path.
It now delegates to the same component pipeline used by `worker.py` so local
callers and tests exercise the production flow.
"""

from core.pipeline_engine import PipelineContext, PipelineEngine, PipelineStep
from services.training.components import (
    DataValidationComponent,
    EvaluationComponent,
    FeatureEngineeringComponent,
    ModelSelectionComponent,
    TrainingComponent,
)


def build_production_pipeline(
    job_id: str,
    dataset_id: str,
    file_path: str,
    target: str,
    goal: str,
    mode: str,
    config: dict | None = None,
) -> PipelineEngine:
    ctx = PipelineContext(
        job_id=job_id,
        dataset_id=dataset_id,
        file_path=file_path,
        target_column=target,
        goal=goal,
        mode=mode,
        config=config or {},
    )
    return PipelineEngine(
        context=ctx,
        components=[
            DataValidationComponent(),
            FeatureEngineeringComponent(),
            ModelSelectionComponent(),
            TrainingComponent(),
            EvaluationComponent(),
        ],
    )


__all__ = ["PipelineContext", "PipelineEngine", "PipelineStep", "build_production_pipeline"]

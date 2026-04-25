import json
import os

import joblib
import pandas as pd
import pytest

from core.meta_learning import MetaLearner
from core.pipeline_engine import PipelineContext
from core.worker import run_training_job
from infra.database import DatasetModel, JobModel
from infra.storage import get_model_path
from services.training.components import DataValidationComponent
from services.training.inference import TabularModelPipeline
from .conftest import TestingSessionLocal


def _run_worker_training(tmp_path, suffix: str, eval_metric: str = "Accuracy"):
    csv_path = tmp_path / f"worker_train_{suffix}.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 25, 31, 38, 44, 52, 27, 36, 48, 55, 29, 41],
            "income": [31, 35, 48, 61, 74, 89, 42, 57, 81, 96, 45, 68],
            "signup_date": [
                "2024-01-01",
                "2024-01-08",
                "2024-01-12",
                "2024-02-01",
                "2024-02-10",
                "2024-03-04",
                "2024-01-20",
                "2024-02-18",
                "2024-03-12",
                "2024-04-01",
                "2024-01-25",
                "2024-02-26",
            ],
            "segment": ["basic", "basic", "plus", "plus", "pro", "pro", "basic", "plus", "pro", "pro", "basic", "plus"],
            "outcome": ["stay", "stay", "stay", "buy", "buy", "buy", "stay", "buy", "buy", "buy", "stay", "buy"],
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id=f"worker-ds-{suffix}",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "classification",
                        "suggested_target": "outcome",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id=f"worker-job-{suffix}",
                dataset_id=f"worker-ds-{suffix}",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        f"worker-job-{suffix}",
        f"worker-ds-{suffix}",
        os.fspath(csv_path),
        "outcome",
        "Speed",
        "Fast",
        task_type="classification",
        eval_metric=eval_metric,
        cv_folds=3,
    )

    return f"worker-job-{suffix}"


def test_worker_pipeline_builds_end_to_end_artifact(tmp_path):
    job_id = _run_worker_training(tmp_path, "artifact")

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()

    assert job is not None
    assert job.status == "completed"

    results = json.loads(job.results_json)
    assert results["feature_names"] == ["age", "income", "signup_date", "segment"]
    assert results["model_metadata"]["task_type"] == "classification"
    assert results["task_detection"]["task_type"] == "classification"
    assert "derived_feature_names" in results


def test_worker_pipeline_balanced_mode_completes(tmp_path):
    csv_path = tmp_path / "worker_train_balanced.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 25, 31, 38, 44, 52, 27, 36, 48, 55, 29, 41, 24, 33, 46, 58],
            "income": [31, 35, 48, 61, 74, 89, 42, 57, 81, 96, 45, 68, 39, 54, 77, 92],
            "signup_date": [
                "2024-01-01",
                "2024-01-08",
                "2024-01-12",
                "2024-02-01",
                "2024-02-10",
                "2024-03-04",
                "2024-01-20",
                "2024-02-18",
                "2024-03-12",
                "2024-04-01",
                "2024-01-25",
                "2024-02-26",
                "2024-03-06",
                "2024-03-16",
                "2024-04-11",
                "2024-04-20",
            ],
            "segment": [
                "basic",
                "basic",
                "plus",
                "plus",
                "pro",
                "pro",
                "basic",
                "plus",
                "pro",
                "pro",
                "basic",
                "plus",
                "basic",
                "plus",
                "pro",
                "pro",
            ],
            "outcome": [
                "stay",
                "stay",
                "stay",
                "buy",
                "buy",
                "buy",
                "stay",
                "buy",
                "buy",
                "buy",
                "stay",
                "buy",
                "stay",
                "stay",
                "buy",
                "buy",
            ],
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="worker-ds-balanced",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "classification",
                        "suggested_target": "outcome",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="worker-job-balanced",
                dataset_id="worker-ds-balanced",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        "worker-job-balanced",
        "worker-ds-balanced",
        os.fspath(csv_path),
        "outcome",
        "Balanced",
        "Balanced",
        task_type="classification",
        eval_metric="Accuracy",
        cv_folds=2,
    )

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == "worker-job-balanced").first()

    assert job is not None
    assert job.status == "completed"

    results = json.loads(job.results_json)
    assert results["best_model"]
    assert results["model_metadata"]["task_type"] == "classification"


def test_worker_pipeline_uses_shared_loader_for_json_datasets(tmp_path):
    json_path = tmp_path / "worker_train.json"
    frame = pd.DataFrame(
        {
            "age": [21, 25, 31, 38, 44, 52, 27, 36, 48, 55, 29, 41],
            "income": [31, 35, 48, 61, 74, 89, 42, 57, 81, 96, 45, 68],
            "signup_date": [
                "2024-01-01",
                "2024-01-08",
                "2024-01-12",
                "2024-02-01",
                "2024-02-10",
                "2024-03-04",
                "2024-01-20",
                "2024-02-18",
                "2024-03-12",
                "2024-04-01",
                "2024-01-25",
                "2024-02-26",
            ],
            "segment": ["basic", "basic", "plus", "plus", "pro", "pro", "basic", "plus", "pro", "pro", "basic", "plus"],
            "outcome": ["stay", "stay", "stay", "buy", "buy", "buy", "stay", "buy", "buy", "buy", "stay", "buy"],
        }
    )
    frame.to_json(json_path, orient="records")

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="worker-ds-json",
                file_path=os.fspath(json_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "classification",
                        "suggested_target": "outcome",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="worker-job-json",
                dataset_id="worker-ds-json",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        "worker-job-json",
        "worker-ds-json",
        os.fspath(json_path),
        "outcome",
        "Speed",
        "Fast",
        task_type="classification",
        eval_metric="Accuracy",
        cv_folds=3,
    )

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == "worker-job-json").first()

    assert job is not None
    assert job.status == "completed"
    results = json.loads(job.results_json)
    assert results["feature_names"] == ["age", "income", "signup_date", "segment"]


def test_worker_pipeline_respects_auto_clean_flag(tmp_path):
    csv_path = tmp_path / "worker_train_no_clean.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 21, 35, 47, 52, 29, 41, 33],
            "income": ["31", "31", "48", "61", "74", "42", "57", "39"],
            "segment": ["basic", "basic", "plus", "pro", "pro", "basic", "plus", "basic"],
            "outcome": ["stay", "stay", "buy", "buy", "buy", "stay", "buy", "stay"],
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="worker-ds-no-clean",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "classification",
                        "suggested_target": "outcome",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="worker-job-no-clean",
                dataset_id="worker-ds-no-clean",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        "worker-job-no-clean",
        "worker-ds-no-clean",
        os.fspath(csv_path),
        "outcome",
        "Speed",
        "Fast",
        task_type="classification",
        eval_metric="Accuracy",
        cv_folds=2,
        auto_clean=False,
    )

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == "worker-job-no-clean").first()

    assert job is not None
    assert job.status == "completed"
    results = json.loads(job.results_json)
    assert results["sanitizer_report"]["duplicate_rows_removed"] == 0
    assert results["sanitizer_report"]["numeric_coercions"] == []
    assert any(
        "Auto-clean disabled" in message for message in results["reasoning"]
    )


def test_worker_regression_scores_use_display_metric_for_rmse(tmp_path):
    csv_path = tmp_path / "worker_train_regression.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 25, 31, 38, 44, 52, 27, 36, 48, 55, 29, 41],
            "income": [31, 35, 48, 61, 74, 89, 42, 57, 81, 96, 45, 68],
            "tenure": [1, 2, 2, 3, 4, 5, 1, 3, 4, 5, 2, 4],
            "salary": [32000, 36000, 47000, 59000, 71000, 88000, 41000, 56000, 79000, 93000, 45000, 66000],
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="worker-ds-reg-rmse",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "regression",
                        "suggested_target": "salary",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="worker-job-reg-rmse",
                dataset_id="worker-ds-reg-rmse",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        "worker-job-reg-rmse",
        "worker-ds-reg-rmse",
        os.fspath(csv_path),
        "salary",
        "Balanced",
        "Fast",
        task_type="regression",
        eval_metric="RMSE",
        cv_folds=2,
    )

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == "worker-job-reg-rmse").first()

    assert job is not None
    assert job.status == "completed"
    results = json.loads(job.results_json)
    assert results["metric_name"] == "CV RMSE"
    assert results["score"] >= 0
    assert results["holdout_score"] >= 0
    assert results["leaderboard"][0]["score_label"] == "CV RMSE"
    assert results["leaderboard"][0]["holdout_score_label"] == "Holdout RMSE"
    assert results["leaderboard"][0]["phase"] == "cross_validation"


def test_worker_leaderboard_rows_include_phase_and_score_label(tmp_path):
    job_id = _run_worker_training(tmp_path, "leaderboard-shape")

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()

    assert job is not None
    results = json.loads(job.results_json)
    assert results["metric_name"].startswith("CV ")
    assert results["leaderboard"]
    for row in results["leaderboard"]:
        assert row["phase"] == "cross_validation"
        assert row["score_label"] == results["metric_name"]
        assert row["score_direction"] in {"higher_is_better", "lower_is_better"}
        if results["is_classification"]:
            assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(row.keys())
        else:
            assert {"r2", "mae", "mse", "rmse"} <= set(row.keys())
    assert results["leaderboard"][0]["validation_status"] in {
        "stable",
        "watch",
        "possible_overfit",
        "holdout_outperformed_cv",
    }
    assert "absolute_gap_display" in results["leaderboard"][0]
    assert "validation_summary" in results
    assert "absolute_gap_display" in results["validation_summary"]


def test_worker_uses_f1_default_when_metric_is_auto(tmp_path):
    job_id = _run_worker_training(tmp_path, "default-metric", eval_metric="")

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()

    assert job is not None
    results = json.loads(job.results_json)
    assert results["performance_metrics"]["optimized_metric"]["requested"] == "F1-score"
    assert results["model_metadata"]["eval_metric_requested"] == "F1-score"
    assert results["validation_summary"]["score_label"] == results["metric_name"]
    assert results["leaderboard"][0]["validation_status"] == results["validation_summary"]["status"]


def test_worker_disables_classification_only_options_for_regression(tmp_path):
    csv_path = tmp_path / "worker_train_regression_controls.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 25, 31, 38, 44, 52, 27, 36, 48, 55, 29, 41],
            "income": [31, 35, 48, 61, 74, 89, 42, 57, 81, 96, 45, 68],
            "tenure": [1, 2, 2, 3, 4, 5, 1, 3, 4, 5, 2, 4],
            "salary": [32000, 36000, 47000, 59000, 71000, 88000, 41000, 56000, 79000, 93000, 45000, 66000],
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="worker-ds-reg-controls",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "regression",
                        "suggested_target": "salary",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="worker-job-reg-controls",
                dataset_id="worker-ds-reg-controls",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        "worker-job-reg-controls",
        "worker-ds-reg-controls",
        os.fspath(csv_path),
        "salary",
        "Speed",
        "Fast",
        task_type="regression",
        eval_metric="Precision",
        handle_imbalance=True,
        cv_folds=2,
    )

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == "worker-job-reg-controls").first()

    assert job is not None
    assert job.status == "completed"
    results = json.loads(job.results_json)
    assert results["model_metadata"]["task_type"] == "regression"
    assert results["model_metadata"]["eval_metric_requested"] == "RMSE"
    assert any("classification-only" in message or "disabling it for regression" in message for message in results["reasoning"])
    assert results["performance_metrics"]["optimized_metric"]["resolved_display_label"] == "CV RMSE"
    assert results["performance_metrics"]["optimized_metric"]["score_direction"] == "lower_is_better"


def test_worker_warns_when_accuracy_is_used_on_imbalanced_classification(tmp_path):
    csv_path = tmp_path / "worker_train_imbalanced.csv"
    frame = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "income": list(range(40, 60)),
            "outcome": ["stay"] * 16 + ["buy"] * 4,
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="worker-ds-imbalanced",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "classification",
                        "suggested_target": "outcome",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="worker-job-imbalanced",
                dataset_id="worker-ds-imbalanced",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        "worker-job-imbalanced",
        "worker-ds-imbalanced",
        os.fspath(csv_path),
        "outcome",
        "Speed",
        "Fast",
        task_type="classification",
        eval_metric="Accuracy",
        cv_folds=2,
    )

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == "worker-job-imbalanced").first()

    results = json.loads(job.results_json)
    warning_types = {warning["type"] for warning in results["warnings"]}
    assert "class_imbalance" in warning_types
    assert "metric_mismatch" in warning_types


def test_worker_goal_changes_execution_budget(tmp_path):
    csv_path = tmp_path / "worker_train_goal_budget.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 25, 31, 38, 44, 52, 27, 36, 48, 55, 29, 41, 24, 33, 46, 58],
            "income": [31, 35, 48, 61, 74, 89, 42, 57, 81, 96, 45, 68, 39, 54, 77, 92],
            "signup_date": [
                "2024-01-01","2024-01-08","2024-01-12","2024-02-01",
                "2024-02-10","2024-03-04","2024-01-20","2024-02-18",
                "2024-03-12","2024-04-01","2024-01-25","2024-02-26",
                "2024-03-06","2024-03-16","2024-04-11","2024-04-20",
            ],
            "segment": ["basic","basic","plus","plus","pro","pro","basic","plus","pro","pro","basic","plus","basic","plus","pro","pro"],
            "outcome": ["stay","stay","stay","buy","buy","buy","stay","buy","buy","buy","stay","buy","stay","stay","buy","buy"],
        }
    )
    frame.to_csv(csv_path, index=False)

    for dataset_id, job_id, goal in [
        ("worker-ds-goal-speed", "worker-job-goal-speed", "Speed"),
        ("worker-ds-goal-performance", "worker-job-goal-performance", "Performance"),
    ]:
        with TestingSessionLocal() as db:
            db.add(
                DatasetModel(
                    id=dataset_id,
                    file_path=os.fspath(csv_path),
                    profile_json=json.dumps(
                        {
                            "rows": len(frame),
                            "cols": len(frame.columns),
                            "columns": list(frame.columns),
                            "task_type": "classification",
                            "suggested_target": "outcome",
                        }
                    ),
                    source_type="upload",
                )
            )
            db.add(
                JobModel(
                    id=job_id,
                    dataset_id=dataset_id,
                    status="training",
                    reasoning_json="[]",
                    params_json=json.dumps({}),
                )
            )
            db.commit()

        run_training_job.run(
            job_id,
            dataset_id,
            os.fspath(csv_path),
            "outcome",
            goal,
            "Balanced",
            task_type="classification",
            eval_metric="Accuracy",
            cv_folds=2,
        )

    with TestingSessionLocal() as db:
        speed_job = db.query(JobModel).filter(JobModel.id == "worker-job-goal-speed").first()
        performance_job = db.query(JobModel).filter(JobModel.id == "worker-job-goal-performance").first()

    speed_results = json.loads(speed_job.results_json)
    performance_results = json.loads(performance_job.results_json)
    assert speed_results["execution_profile"]["goal"] == "Speed"
    assert performance_results["execution_profile"]["goal"] == "Performance"
    assert speed_results["execution_profile"]["top_k"] <= performance_results["execution_profile"]["top_k"]
    assert speed_results["execution_profile"]["n_trials"] <= performance_results["execution_profile"]["n_trials"]
    assert speed_results["execution_profile"]["sweep_size"] <= performance_results["execution_profile"]["sweep_size"]


def test_worker_artifact_decodes_original_class_labels():
    from pathlib import Path

    job_id = _run_worker_training(Path("/tmp"), "predict")
    artifact = joblib.load(get_model_path(job_id))

    payload = pd.DataFrame(
        [
            {
                "age": 33,
                "income": 58,
                "signup_date": "2024-02-15",
                "segment": "plus",
            }
        ]
    )
    prediction = artifact.predict(payload)[0]
    probabilities = artifact.predict_proba(payload)[0]

    assert prediction in {"stay", "buy"}
    assert set(artifact.classes_) == {"buy", "stay"}
    assert len(probabilities) == 2


def test_worker_artifact_preserves_original_target_label_case(tmp_path):
    csv_path = tmp_path / "worker_train_label_case.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 25, 31, 38, 44, 52, 27, 36],
            "income": [31, 35, 48, 61, 74, 89, 42, 57],
            "segment": ["basic", "basic", "plus", "plus", "pro", "pro", "basic", "plus"],
            "outcome": ["Stay", "Stay", "Stay", "Buy", "Buy", "Buy", "Stay", "Buy"],
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="worker-ds-label-case",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "classification",
                        "suggested_target": "outcome",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="worker-job-label-case",
                dataset_id="worker-ds-label-case",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        "worker-job-label-case",
        "worker-ds-label-case",
        os.fspath(csv_path),
        "outcome",
        "Speed",
        "Fast",
        task_type="classification",
        eval_metric="Accuracy",
        cv_folds=2,
    )

    artifact = joblib.load(get_model_path("worker-job-label-case"))
    prediction = artifact.predict(frame.drop(columns=["outcome"]).head(1))[0]

    assert prediction in {"Stay", "Buy"}
    assert set(artifact.classes_) == {"Buy", "Stay"}


def test_worker_uses_temporal_validation_when_datetime_column_is_ordered(tmp_path):
    csv_path = tmp_path / "worker_train_temporal.csv"
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=20, freq="D").astype(str),
            "signal": list(range(20)),
            "outcome": ["cold", "warm"] * 10,
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="worker-ds-temporal",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": len(frame),
                        "cols": len(frame.columns),
                        "columns": list(frame.columns),
                        "task_type": "classification",
                        "suggested_target": "outcome",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="worker-job-temporal",
                dataset_id="worker-ds-temporal",
                status="training",
                reasoning_json="[]",
                params_json=json.dumps({}),
            )
        )
        db.commit()

    run_training_job.run(
        "worker-job-temporal",
        "worker-ds-temporal",
        os.fspath(csv_path),
        "outcome",
        "Speed",
        "Fast",
        task_type="classification",
        eval_metric="Accuracy",
        cv_folds=3,
    )

    with TestingSessionLocal() as db:
        job = db.query(JobModel).filter(JobModel.id == "worker-job-temporal").first()

    results = json.loads(job.results_json)
    assert results["model_metadata"]["temporal_validation"] is True
    assert results["model_metadata"]["temporal_order_column"] == "event_time"
    assert results["reproducibility_snapshot"]["temporal_validation"] is True


def test_data_validation_defers_leakage_drops_until_training_split(tmp_path):
    csv_path = tmp_path / "worker_train_leak_check.csv"
    frame = pd.DataFrame(
        {
            "feature": [10, 11, 12, 13, 14, 15],
            "perfect_leak": [0, 1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    frame.to_csv(csv_path, index=False)

    ctx = PipelineContext(
        job_id="validation-leak-check",
        dataset_id="dataset-leak-check",
        file_path=os.fspath(csv_path),
        target_column="target",
        goal="Speed",
        mode="Fast",
        config={"auto_clean": False, "task_type": "classification"},
    )

    DataValidationComponent().execute(ctx)

    assert "perfect_leak" in ctx.df.columns


def test_meta_learner_encodes_string_metadata_for_lightgbm():
    learner = MetaLearner()
    if learner.model is None:
        pytest.skip("LightGBM is not available in the test environment")

    records = []
    for i in range(12):
        records.append(
            {
                "meta_features_json": json.dumps(
                    {
                        "n_rows": 100 + i * 10,
                        "n_cols": 4,
                        "num_ratio": 0.75,
                        "cat_ratio": 0.25,
                        "binary_ratio": 0.0,
                        "datetime_ratio": 0.0,
                        "continuous_ratio": 0.75,
                        "missing_pct": 0.0,
                        "is_imbalanced": 0,
                    }
                ),
                "leaderboard_json": json.dumps(
                    [
                        {
                            "model": "Random Forest",
                            "score": 70 + i,
                            "phase": "cross_validation",
                        },
                        {
                            "model": "XGBoost",
                            "score": 68 + i,
                            "phase": "cross_validation",
                        },
                    ]
                ),
                "task_type": "classification",
                "metric_name": "Accuracy",
            }
        )

    X, y = learner.prepare_data(records)

    assert not X.empty
    assert y is not None
    assert len(X.select_dtypes(include=["object", "string", "category"]).columns) == 0

    learner.model.fit(X, y)
    learner.is_trained = True
    learner.val_error = 0.1

    recommendation = learner.predict_rankings(
        {
            "rows": 180,
            "cols": 4,
            "num_cols": ["age", "income", "score"],
            "cat_cols": ["segment"],
            "column_stats": {},
            "missing_pct": 0.0,
            "imbalance": "Low",
        },
        ["Random Forest", "XGBoost"],
    )

    assert recommendation["source"] == "LightGBM Meta-Learner"
    assert [entry["model"] for entry in recommendation["rankings"]] == [
        "Random Forest",
        "XGBoost",
    ]


def test_tabular_pipeline_preserves_duplicate_rows_during_fit():
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame(
        {
            "segment": ["basic", "basic", "plus", "plus", "basic", "basic"],
            "signup_date": [
                "2024-01-01",
                "2024-01-01",
                "2024-02-10",
                "2024-02-10",
                "2024-03-15",
                "2024-03-15",
            ],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 0])

    pipeline = TabularModelPipeline(
        base_estimator=LogisticRegression(max_iter=1000),
        feature_columns=list(X.columns),
        preprocessing="lite",
    )

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(X)
    assert list(pipeline.feature_names_in_) == list(X.columns)

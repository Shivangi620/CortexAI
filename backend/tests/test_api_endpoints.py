import asyncio
import json
import math
import os
import zipfile
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from infra.database import DatasetModel, ExperimentRun, JobModel
from infra.storage import get_model_path
from .conftest import TestingSessionLocal
from api.routes.datasets import (
    _dropbox_download_url,
    SourceImportRequest,
    _inspect_local_source,
    _normalize_remote_url,
    _onedrive_download_url,
    get_latest_workspace,
    import_from_source,
    restore_workspace,
    upload_dataset,
)
from api.routes.experiments import list_experiments
from api.routes.drift import _resolve_retrain_launch_config, _resolve_retrain_launch_context
from api.routes.predict import (
    ScenarioPackPayload,
    batch_predict,
    get_scenario_context,
    list_scenario_packs,
    save_scenario_pack,
)
from api.routes.misc import synthetic_expand
from services.studio_service import synthetic_data_judge
from api.routes.reports import _artifact_filename, _model_card_html
from api.routes.training import (
    TrainingRegistryPreviewRequest,
    get_status,
    get_training_model_registry,
    global_leaderboard,
    list_jobs,
)
from main import app, health_check


def test_health_check():
    response = health_check()
    assert response["status"] == "ok"


def test_get_jobs_empty():
    response = list_jobs()
    assert isinstance(response, list)


def test_get_experiments():
    response = list_experiments()
    assert isinstance(response, list)


def test_get_leaderboard():
    response = global_leaderboard()
    assert isinstance(response, list)


def test_404_not_found():
    paths = {route.path for route in app.routes}
    assert "/api/this_does_not_exist" not in paths


def test_status_endpoint_sanitizes_nan_payloads():
    with TestingSessionLocal() as db:
        db.add(
            JobModel(
                id="job-with-nan",
                dataset_id="ds-1",
                status="completed",
                history_json=json.dumps([{"time": "Final", "metric": math.nan}]),
                results_json='{"best_model":"LGBM","score":NaN,"leaderboard":[{"model":"LGBM","score":NaN}]}',
                insights_json='{"confidence":NaN}',
                reasoning_json=json.dumps(["done"]),
            )
        )
        db.commit()

    body = get_status("job-with-nan")
    assert body["history"] == [{"time": "Final", "metric": None}]
    assert body["results"]["score"] == 0.0
    assert body["results"]["leaderboard"] == [{"model": "LGBM", "score": None}]
    assert body["insights"] == {"confidence": None}


def test_status_endpoint_exposes_persisted_model_registry_preview():
    with TestingSessionLocal() as db:
        db.add(
            JobModel(
                id="job-with-registry-preview",
                dataset_id="ds-1",
                status="training",
                params_json=json.dumps(
                    {
                        "goal": "Balanced",
                        "mode": "Full",
                        "model_registry_preview": {
                            "selection_goal": "Performance",
                            "mode": "Full",
                            "selected_models": ["Logistic Regression", "Random Forest", "Hist Gradient Boosting"],
                            "model_groups": {
                                "baseline": ["Logistic Regression"],
                                "boosting": ["Hist Gradient Boosting"],
                                "optional": [],
                            },
                        },
                    }
                ),
            )
        )
        db.commit()

    body = get_status("job-with-registry-preview")
    assert body["id"] == "job-with-registry-preview"
    assert body["config"]["goal"] == "Balanced"
    assert body["model_registry_preview"]["selection_goal"] == "Performance"
    assert body["model_registry_preview"]["selected_models"] == [
        "Logistic Regression",
        "Random Forest",
        "Hist Gradient Boosting",
    ]


def test_jobs_and_experiments_expose_drift_reopen_origin():
    with TestingSessionLocal() as db:
        db.add(
            JobModel(
                id="job-with-launch-origin",
                dataset_id="ds-1",
                status="completed",
                params_json=json.dumps(
                    {
                        "launch_context": {
                            "source": "drift_recommendation",
                            "parent_job_id": "parent-job",
                        }
                    }
                ),
                results_json=json.dumps(
                    {
                        "best_model": "Random Forest",
                        "score": 91.2,
                        "metric_name": "F1-score",
                        "is_classification": True,
                    }
                ),
            )
        )
        db.add(
            ExperimentRun(
                id="exp-with-launch-origin",
                job_id="job-with-launch-origin",
                dataset_id="ds-1",
                model_name="Random Forest",
                metric_name="F1-score",
                score="91.2",
                task_type="classification",
                mode="Balanced",
                goal="Balanced",
            )
        )
        db.commit()

    jobs = list_jobs()
    job_row = next(row for row in jobs if row["id"] == "job-with-launch-origin")
    assert job_row["launch_source"] == "drift_recommendation"
    assert job_row["launch_label"] == "Drift Reopen"

    experiments = list_experiments()
    exp_row = next(row for row in experiments if row["id"] == "exp-with-launch-origin")
    assert exp_row["launch_source"] == "drift_recommendation"
    assert exp_row["launch_label"] == "Drift Reopen"


def test_model_card_html_includes_drift_reopen_origin():
    html = _model_card_html(
        "job-drift-card",
        {"best_model": "Random Forest", "metric_name": "F1-score", "score": 91.2},
        {"rows": 1000, "cols": 12},
        {},
        "",
        {
            "launch_source": "drift_recommendation",
            "launch_label": "Drift Reopen",
            "launch_context": {
                "parent_job_id": "parent-job-123",
                "recommended_goal": "Performance",
                "recommended_mode": "Full",
                "message": "Reopened after critical drift.",
            },
        },
    )

    assert "Operational Origin" in html
    assert "Drift Reopen" in html
    assert "parent-job-123" in html
    assert "Performance" in html
    assert "Full" in html


def test_artifact_filename_uses_launch_origin_prefix():
    drift_name = _artifact_filename(
        "job-drift-card",
        {"launch_source": "drift_recommendation"},
        "pdf",
    )
    manual_name = _artifact_filename(
        "job-manual-card",
        {"launch_source": "manual"},
        "model_card",
    )

    assert drift_name == "drift_reopen_automl_report_job-drif.pdf"
    assert manual_name == "manual_model_card_job-manu.html"


def test_synthetic_expand_returns_dataset_ids_and_actual_row_count(tmp_path):
    dataset_id = "synthetic-route-ds"
    dataset_path = tmp_path / "synthetic_route.csv"
    frame = pd.DataFrame(
        {
            "flag": [True, False, None, True],
            "city": ["Austin", "Boston", "Austin", None],
            "value": [1.2, 2.4, 1.7, 2.1],
        }
    )
    frame.to_csv(dataset_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id=dataset_id,
                file_path=os.fspath(dataset_path),
                profile_json=json.dumps({"rows": len(frame), "cols": len(frame.columns), "columns": list(frame.columns)}),
                source_type="upload",
            )
        )
        db.commit()

    payload = synthetic_expand(dataset_id, n_rows=6)

    assert payload["dataset_id"] == dataset_id
    assert payload["new_dataset_id"]
    assert payload["synthetic_rows_added"] == 6
    assert payload["total_rows"] == len(frame) + 6


def test_synthetic_judge_handles_nullable_boolean_columns(tmp_path):
    parent_id = "synthetic-parent-ds"
    child_id = "synthetic-child-ds"
    parent_path = tmp_path / "parent.csv"
    child_path = tmp_path / "child.csv"

    parent = pd.DataFrame(
        {
            "flag": pd.Series([True, False, None], dtype="boolean"),
            "segment": pd.Series(["A", "B", "A"], dtype="string"),
            "value": [1.0, 2.0, 1.5],
        }
    )
    child = pd.concat(
        [
            parent,
            pd.DataFrame(
                {
                    "flag": pd.Series([True, None], dtype="boolean"),
                    "segment": pd.Series(["A", "B"], dtype="string"),
                    "value": [1.1, 1.9],
                }
            ),
        ],
        ignore_index=True,
    )
    parent.to_csv(parent_path, index=False)
    child.to_csv(child_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id=parent_id,
                file_path=os.fspath(parent_path),
                profile_json=json.dumps({"rows": len(parent), "columns": list(parent.columns)}),
                source_type="upload",
            )
        )
        db.add(
            DatasetModel(
                id=child_id,
                file_path=os.fspath(child_path),
                profile_json=json.dumps({"rows": len(child), "columns": list(child.columns)}),
                source_type="synthetic",
                parent_dataset_id=parent_id,
            )
        )
        db.commit()

    payload = synthetic_data_judge(child_id)

    assert payload["dataset_id"] == child_id
    assert payload["rows_evaluated"] == 2
    assert "realism_score" in payload


def test_scenario_pack_persists_approval_policy_and_approved_scenarios():
    payload = ScenarioPackPayload(
        name="Guardrail Pack",
        description="Policy-aware simulator pack",
        base_mode="cohort",
        sweep_feature="income",
        sweep_values=[10, 20, 30],
        approval_policy={
            "max_numeric_delta_ratio": 0.2,
            "hard_bounds": True,
            "blocked_features": ["salary", "price"],
        },
        approved_scenarios=["scenario-1", "scenario-2"],
    )

    response = save_scenario_pack("scenario-pack-job", payload)
    listed = list_scenario_packs("scenario-pack-job")

    assert response["saved"] is True
    assert listed["packs"]
    pack = listed["packs"][0]
    assert pack["approval_policy"]["max_numeric_delta_ratio"] == 0.2
    assert pack["approval_policy"]["hard_bounds"] is True
    assert pack["approval_policy"]["blocked_features"] == ["salary", "price"]
    assert pack["approved_scenarios"] == ["scenario-1", "scenario-2"]


def test_experiments_endpoint_sanitizes_nan_payloads():
    with TestingSessionLocal() as db:
        db.add(
            ExperimentRun(
                id="exp-with-nan",
                job_id="job-exp-nan",
                dataset_id="ds-1",
                model_name="LGBM",
                metric_name="Accuracy",
                score=math.nan,
                task_type="classification",
                mode="Balanced",
                goal="Performance",
                feature_count=5,
                row_count=100,
                hyperparams_json='{"depth": 8, "lr": NaN}',
                metrics_json='{"precision": NaN, "recall": 88.1}',
                leaderboard_json='[{"model":"LGBM","score":NaN}]',
            )
        )
        db.commit()

    rows = list_experiments()
    target = next(row for row in rows if row["id"] == "exp-with-nan")
    assert target["score"] is None
    assert target["hyperparams"] == {"depth": 8, "lr": None}
    assert target["metrics"] == {"precision": None, "recall": 88.1}
    assert target["leaderboard"] == [{"model": "LGBM", "score": None}]


def test_workspace_latest_restores_dataset_and_job(tmp_path):
    csv_path = tmp_path / "workspace.csv"
    csv_path.write_text("feature,target\n1,yes\n2,no\n", encoding="utf-8")

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="workspace-ds",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": 2,
                        "cols": 2,
                        "columns": ["feature", "target"],
                        "suggested_target": "target",
                        "task_type": "classification",
                    }
                ),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="workspace-job",
                dataset_id="workspace-ds",
                status="completed",
                history_json=json.dumps([{"time": "Final", "metric": 93.2}]),
                results_json=json.dumps({"best_model": "LiteGBM", "score": 93.2}),
            )
        )
        db.commit()

    body = get_latest_workspace()
    assert body["dataset"]["id"] == "workspace-ds"
    assert body["dataset"]["profile"]["suggested_target"] == "target"
    assert body["dataset"]["preview_records"][0]["feature"] == 1
    assert body["job"]["id"] == "workspace-job"


def test_workspace_restore_honors_explicit_job_id():
    with TestingSessionLocal() as db:
        if not db.query(DatasetModel).filter(DatasetModel.id == "workspace-ds").first():
            db.add(
                DatasetModel(
                    id="workspace-ds",
                    file_path="tests/data/dummy_data.csv",
                    profile_json=json.dumps(
                        {
                            "rows": 2,
                            "cols": 2,
                            "columns": ["feature", "target"],
                            "suggested_target": "target",
                            "task_type": "classification",
                        }
                    ),
                    source_type="upload",
                )
            )
        if not db.query(JobModel).filter(JobModel.id == "workspace-job").first():
            db.add(
                JobModel(
                    id="workspace-job",
                    dataset_id="workspace-ds",
                    status="completed",
                    history_json=json.dumps([{"time": "Final", "metric": 93.2}]),
                    results_json=json.dumps({"best_model": "LiteGBM", "score": 93.2}),
                )
            )
        db.commit()

    body = restore_workspace(job_id="workspace-job")
    assert body["job"]["id"] == "workspace-job"
    assert body["dataset"]["id"] == "workspace-ds"


def test_training_model_registry_preview_exposes_selected_models_and_full_mode_upgrade(tmp_path):
    csv_path = tmp_path / "registry_preview.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 29, 37, 46, 52, 61],
            "income": [31000, 45000, 62000, 83000, 91000, 102000],
            "city": ["A", "B", "A", "B", "C", "A"],
            "target": [0, 0, 1, 1, 1, 0],
        }
    )
    frame.to_csv(csv_path, index=False)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="registry-preview-ds",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps(
                    {
                        "rows": 6,
                        "cols": 4,
                        "columns": ["age", "income", "city", "target"],
                        "suggested_target": "target",
                    }
                ),
                source_type="upload",
            )
        )
        db.commit()

    body = get_training_model_registry(
        TrainingRegistryPreviewRequest(
            dataset_id="registry-preview-ds",
            target_column="target",
            goal="Balanced",
            mode="Full",
        )
    )
    assert body["task_type"] == "classification"
    assert body["requested_goal"] == "Balanced"
    assert body["selection_goal"] == "Performance"
    assert "Logistic Regression" in body["selected_models"]
    assert "Random Forest" in body["selected_models"]
    assert "Hist Gradient Boosting" in body["selected_models"]


def test_drift_retrain_launch_config_prefers_recommendation_override():
    payload = _resolve_retrain_launch_config(
        {"goal": "Balanced", "mode": "Balanced"},
        goal_override="Performance",
        mode_override="Full",
    )

    assert payload == {"goal": "Performance", "mode": "Full"}


def test_drift_retrain_launch_config_falls_back_to_job_defaults():
    payload = _resolve_retrain_launch_config(
        {"goal": "Balanced", "mode": "Balanced"},
        goal_override="",
        mode_override="",
    )

    assert payload == {"goal": "Balanced", "mode": "Balanced"}


def test_drift_retrain_launch_context_carries_origin_story():
    payload = _resolve_retrain_launch_context(
        json.dumps(
            {
                "source_job_id": "source-job-1234",
                "message": "Re-open the model search beyond Logistic Regression.",
                "candidate_models": ["Random Forest", "Hist Gradient Boosting"],
            }
        ),
        parent_job_id="parent-job-9876",
        launch_config={"goal": "Performance", "mode": "Full"},
    )

    assert payload["source"] == "drift_recommendation"
    assert payload["parent_job_id"] == "parent-job-9876"
    assert payload["recommended_goal"] == "Performance"
    assert payload["recommended_mode"] == "Full"
    assert payload["source_job_id"] == "source-job-1234"
    assert payload["candidate_models"] == ["Random Forest", "Hist Gradient Boosting"]


def test_scenario_context_endpoint_returns_ranges_and_rows(tmp_path):
    csv_path = tmp_path / "scenario.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 29, 37, 46],
            "income": [31000, 45000, 62000, 83000],
            "target": [0, 0, 1, 1],
        }
    )
    frame.to_csv(csv_path, index=False)

    model = LogisticRegression(max_iter=1000)
    model.fit(frame[["age", "income"]], frame["target"])
    model_path = get_model_path("scenario-job")
    joblib.dump(model, model_path)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="scenario-ds",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps({"rows": 4, "cols": 3, "columns": ["age", "income", "target"]}),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="scenario-job",
                dataset_id="scenario-ds",
                status="completed",
                model_path=model_path,
                results_json=json.dumps(
                    {
                        "best_model": "Logistic Regression",
                        "score": 91.2,
                        "feature_names": ["age", "income"],
                        "target": "target",
                        "is_classification": True,
                        "model_path": model_path,
                    }
                ),
            )
        )
        db.commit()

    body = get_scenario_context("scenario-job")
    assert body["feature_names"] == ["age", "income"]
    assert len(body["feature_ranges"]) == 2
    assert body["sample_rows"]


def test_scenario_context_endpoint_sanitizes_non_finite_values(tmp_path):
    csv_path = tmp_path / "scenario_dirty.csv"
    frame = pd.DataFrame(
        {
            "age": [21, 29, 37, 46],
            "income": [31000, np.nan, np.inf, 83000],
            "target": [0, 0, 1, 1],
        }
    )
    frame.to_csv(csv_path, index=False)

    fit_frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    model = LogisticRegression(max_iter=1000)
    model.fit(fit_frame[["age", "income"]], fit_frame["target"])
    model_path = get_model_path("scenario-job-dirty")
    joblib.dump(model, model_path)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="scenario-ds-dirty",
                file_path=os.fspath(csv_path),
                profile_json=json.dumps({"rows": 4, "cols": 3, "columns": ["age", "income", "target"]}),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="scenario-job-dirty",
                dataset_id="scenario-ds-dirty",
                status="completed",
                model_path=model_path,
                results_json=json.dumps(
                    {
                        "best_model": "Logistic Regression",
                        "score": 91.2,
                        "feature_names": ["age", "income"],
                        "target": "target",
                        "is_classification": True,
                        "model_path": model_path,
                    }
                ),
            )
        )
        db.commit()

    body = get_scenario_context("scenario-job-dirty")
    assert json.dumps(body)
    income_range = next(item for item in body["feature_ranges"] if item["feature"] == "income")
    assert income_range["min"] == 31000.0
    assert income_range["max"] == 83000.0
    assert body["sample_rows"][0]["values"]["income"] in {None, 57000.0}


def test_batch_predict_preserves_duplicate_rows(tmp_path):
    train_csv = tmp_path / "batch_train.csv"
    train_frame = pd.DataFrame(
        {
            "age": [21, 29, 37, 46],
            "income": [31000, 45000, 62000, 83000],
            "target": [0, 0, 1, 1],
        }
    )
    train_frame.to_csv(train_csv, index=False)

    model = LogisticRegression(max_iter=1000)
    model.fit(train_frame[["age", "income"]], train_frame["target"])
    model_path = get_model_path("batch-job")
    joblib.dump(model, model_path)

    with TestingSessionLocal() as db:
        db.add(
            DatasetModel(
                id="batch-ds",
                file_path=os.fspath(train_csv),
                profile_json=json.dumps({"rows": 4, "cols": 3, "columns": ["age", "income", "target"]}),
                source_type="upload",
            )
        )
        db.add(
            JobModel(
                id="batch-job",
                dataset_id="batch-ds",
                status="completed",
                model_path=model_path,
                results_json=json.dumps(
                    {
                        "best_model": "Logistic Regression",
                        "score": 91.2,
                        "feature_names": ["age", "income"],
                        "target": "target",
                        "is_classification": True,
                        "model_path": model_path,
                    }
                ),
            )
        )
        db.commit()

    batch_csv = "\n".join(
        [
            "age,income",
            "21,31000",
            "21,31000",
            "37,62000",
            "37,62000",
        ]
    )

    class FakeUpload:
        def __init__(self, filename: str, content: bytes):
            self.filename = filename
            self._content = content
            self._offset = 0

        async def read(self, size: int = -1):
            if self._offset >= len(self._content):
                return b""
            if size is None or size < 0:
                size = len(self._content) - self._offset
            chunk = self._content[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

        async def close(self):
            return None

    upload = FakeUpload("batch.csv", batch_csv.encode("utf-8"))
    body = asyncio.run(batch_predict("batch-job", upload))
    assert body["row_count"] == 4
    assert len(body["preview"]) == 4


def test_model_card_uses_metric_labels_and_non_percent_regression_scores():
    html = _model_card_html(
        "job-rmse-card",
        {
            "best_model": "Ridge",
            "metric_name": "CV RMSE",
            "score": 51971.013,
            "tested_models": [
                {
                    "model": "Ridge",
                    "phase": "cross_validation",
                    "score_label": "CV RMSE",
                    "score": 51971.013,
                }
            ],
        },
        {"rows": 1200, "cols": 8},
        {},
        "",
    )

    assert "CV RMSE" in html
    assert "51,971.0130" in html
    assert "Cross Validation" in html
    assert "Holdout Score" not in html


def test_upload_accepts_generic_zip_dataset(tmp_path):
    import zipfile

    class FakeUpload:
        def __init__(self, filename: str, content: bytes):
            self.filename = filename
            self._content = content
            self._offset = 0

        async def read(self, size: int = -1):
            if self._offset >= len(self._content):
                return b""
            if size is None or size < 0:
                size = len(self._content) - self._offset
            chunk = self._content[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

        async def close(self):
            return None

    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("folder/training_data.csv", "feature,target\n1,0\n2,1\n")

    payload = asyncio.run(
        upload_dataset(
            file=FakeUpload("dataset.zip", archive_path.read_bytes()),
            archive_member="folder/training_data.csv",
        )
    )

    assert payload["ingest_summary"]["source_type"] == "zip_upload"
    assert payload["ingest_summary"]["rows"] == 2


def test_upload_dataset_sanitizes_non_finite_profile_stats():
    class FakeUpload:
        def __init__(self, filename: str, content: bytes):
            self.filename = filename
            self._content = content
            self._offset = 0

        async def read(self, size: int = -1):
            if self._offset >= len(self._content):
                return b""
            if size is None or size < 0:
                size = len(self._content) - self._offset
            chunk = self._content[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

        async def close(self):
            return None

    payload = asyncio.run(
        upload_dataset(
            file=FakeUpload("single-row.csv", b"value,target\n42,1\n"),
        )
    )

    stats = payload["profile"]["column_stats"]["value"]
    assert payload["dataset_id"]
    assert stats["mean"] == 42.0
    assert stats["std"] is None
    assert stats["skew"] is None


def test_import_source_supports_remote_json_api(monkeypatch):
    from api.routes import datasets as dataset_routes

    class FakeResponse:
        headers = {"content-type": "application/json"}
        request = type("Request", (), {"url": "https://api.example.com/data.json"})()

        def json(self):
            return {"data": [{"value": 1, "label": "a"}, {"value": 2, "label": "b"}]}

    monkeypatch.setattr(dataset_routes, "_download_remote_payload", lambda *args, **kwargs: FakeResponse())

    payload = import_from_source(
        SourceImportRequest(
            source_type="api",
            connection_uri="https://api.example.com/data",
            http_method="GET",
            headers_json="{}",
            body_json="{}",
        )
    )

    assert payload["ingest_summary"]["rows"] == 2
    assert payload["preview_records"][0]["label"] == "a"


def test_inspect_local_source_reports_zip_members(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/data.csv", "value\n1\n")

    payload = _inspect_local_source(os.fspath(archive_path), display_name="dataset.zip")

    assert "nested/data.csv" in payload["zip_members"]
    assert payload["recommended"]["archive_member"] == "nested/data.csv"


def test_inspect_local_source_reports_excel_sheets(tmp_path):
    excel_path = tmp_path / "inspect.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame({"value": [1]}).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame({"value": [2]}).to_excel(writer, sheet_name="detail", index=False)

    payload = _inspect_local_source(os.fspath(excel_path), display_name="inspect.xlsx")

    assert payload["excel_sheets"] == ["summary", "detail"]
    assert payload["recommended"]["sheet_name"] == "summary"


def test_dropbox_download_url_forces_direct_download():
    direct = _dropbox_download_url("https://www.dropbox.com/s/abc123/data.csv?dl=0")
    assert "dl=1" in direct


def test_onedrive_download_url_adds_download_flag():
    direct = _onedrive_download_url("https://onedrive.live.com/?cid=abc&resid=def")
    assert "download=1" in direct


def test_normalize_remote_url_routes_cloud_links():
    assert _normalize_remote_url("https://www.dropbox.com/s/abc123/data.csv?dl=0", "dropbox").startswith(
        "https://www.dropbox.com/"
    )
    assert "download=1" in _normalize_remote_url(
        "https://onedrive.live.com/?cid=abc&resid=def",
        "onedrive",
    )

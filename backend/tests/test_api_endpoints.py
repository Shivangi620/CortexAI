import json
import math

from infra.database import ExperimentRun, JobModel
from tests.conftest import TestingSessionLocal


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_jobs_empty(client):
    # Depending on DB setup, it might be empty
    response = client.get("/api/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_experiments(client):
    response = client.get("/api/experiments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_leaderboard(client):
    response = client.get("/api/leaderboard")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_404_not_found(client):
    response = client.get("/api/this_does_not_exist")
    assert response.status_code == 404


def test_status_endpoint_sanitizes_nan_payloads(client):
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

    response = client.get("/api/status/job-with-nan")

    assert response.status_code == 200
    body = response.json()
    assert body["history"] == [{"time": "Final", "metric": None}]
    assert body["results"]["score"] == 0.0
    assert body["results"]["leaderboard"] == [{"model": "LGBM", "score": None}]
    assert body["insights"] == {"confidence": None}


def test_experiments_endpoint_sanitizes_nan_payloads(client):
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

    response = client.get("/api/experiments")

    assert response.status_code == 200
    rows = response.json()
    target = next(row for row in rows if row["id"] == "exp-with-nan")
    assert target["score"] is None
    assert target["hyperparams"] == {"depth": 8, "lr": None}
    assert target["metrics"] == {"precision": None, "recall": 88.1}
    assert target["leaderboard"] == [{"model": "LGBM", "score": None}]

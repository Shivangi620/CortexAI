import json
import math

from infra.database import JobModel
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

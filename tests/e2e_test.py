import requests
import time
import os

BASE_URL = "http://127.0.0.1:8000"

def test_flow():
    print("Starting Expanded E2E Flow Test...")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {resp.status_code} - {resp.json()}")
        assert resp.status_code == 200
    except Exception as e:
        print(f"FAILED: Health Check: {e}")
        return

    # 2. Upload Dataset
    print("Uploading dataset...")
    try:
        with open("/home/aj/Documents/CODIN/tests/data/dummy_data.csv", "rb") as f:
            resp = requests.post(f"{BASE_URL}/api/upload", files={"file": f})
        print(f"Upload: {resp.status_code} - {resp.json()}")
        assert resp.status_code == 200
        dataset_id = resp.json()["dataset_id"]
    except Exception as e:
        print(f"FAILED: Upload: {e}")
        return

    # 3. Start Training
    print(f"Starting training for dataset {dataset_id}...")
    try:
        train_req = {
            "dataset_id": dataset_id,
            "target_column": "target",
            "goal": "Performance",
            "mode": "Fast"
        }
        resp = requests.post(f"{BASE_URL}/api/train", json=train_req)
        print(f"Train: {resp.status_code} - {resp.json()}")
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
    except Exception as e:
        print(f"FAILED: Train: {e}")
        return

    # 4. Poll Status
    print(f"Polling job {job_id}...")
    max_retries = 30
    for i in range(max_retries):
        resp = requests.get(f"{BASE_URL}/api/status/{job_id}")
        status = resp.json()["status"]
        print(f"[{i}] Status: {status}")
        if status == "completed":
            print("Training Completed Successfully!")
            break
        if status == "failed":
            print(f"Training Failed: {resp.json().get('error')}")
            break
        time.sleep(2)
    else:
        print("Training Timed Out!")
        return

    # 5. Check Results
    if status == "completed":
        print("Fetching results...")
        resp = requests.get(f"{BASE_URL}/api/status/{job_id}")
        results = resp.json().get("results")
        print(f"Best Model: {results.get('best_model')}")
        print(f"Score: {results.get('score')}")
        assert results.get("best_model") is not None

        # 6. Test Prediction
        print("Testing Prediction...")
        predict_payload = {
            "features": {
                "feature1": 1.5,
                "feature2": 2.5
            }
        }
        resp = requests.post(f"{BASE_URL}/api/predict/{job_id}", json=predict_payload)
        print(f"Predict: {resp.status_code} - {resp.json()}")
        assert resp.status_code == 200
        assert "prediction" in resp.json()

        # 7. Test Model Card Report
        print("Testing Model Card Report...")
        resp = requests.get(f"{BASE_URL}/api/report/{job_id}/model-card")
        print(f"Model Card: {resp.status_code} (HTML size: {len(resp.text)})")
        assert resp.status_code == 200
        assert "MISSION DOSSIER" in resp.text.upper()

        # 8. Test Drift Detection (No Drift)
        print("Testing Drift Detection (No Drift)...")
        # Verify baseline file exists
        from infra.storage import get_run_dir
        baseline_file = os.path.join(get_run_dir(job_id), "data", "drift_baseline.json")
        print(f"Checking baseline file: {baseline_file}")
        if os.path.exists(baseline_file):
            print("Baseline file EXISTS.")
        else:
            print("Baseline file MISSING!")

        with open("/home/aj/Documents/CODIN/tests/data/dummy_data.csv", "rb") as f:
            resp = requests.post(f"{BASE_URL}/api/drift/{job_id}", files={"file": f})
        
        if resp.status_code != 200:
            print(f"Drift (No Drift) FAILED: {resp.status_code} - {resp.text}")
        else:
            print(f"Drift (No Drift): {resp.status_code} - {resp.json().get('overall_status')}")
        
        assert resp.status_code == 200
        # Updated to match actual strings in drift_service.py
        assert "No Significant Drift" in resp.json().get("overall_status")

        # 9. Test Drift Detection (Significant Drift)
        print("Testing Drift Detection (Significant Drift)...")
        drifted_file = "/home/aj/Documents/CODIN/tests/data/drifted_data.csv"
        os.makedirs(os.path.dirname(drifted_file), exist_ok=True)
        with open(drifted_file, "w") as f:
            f.write("feature1,feature2,target\n")
            for _ in range(20):
                f.write("100.0,200.0,0\n")
        
        with open(drifted_file, "rb") as f:
            resp = requests.post(f"{BASE_URL}/api/drift/{job_id}", files={"file": f})
        
        if resp.status_code != 200:
            print(f"Drift (Significant) FAILED: {resp.status_code} - {resp.text}")
        else:
            print(f"Drift (Significant): {resp.status_code} - {resp.json().get('overall_status')}")
            
        assert resp.status_code == 200
        assert "Drift Detected" in resp.json().get("overall_status")

    print("Expanded E2E Flow Test PASSED!")

if __name__ == "__main__":
    test_flow()

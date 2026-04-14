import os
import shutil
import sqlite3
import json

# Setup paths
BASE_DIR = "/home/aj/Downloads/CODIN"
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
TMP_DIR = os.path.join(BACKEND_DIR, "tmp")
RUNS_DIR = os.path.join(BACKEND_DIR, "runs")
DB_PATH = os.path.join(BACKEND_DIR, "automl_studio.db")


def migrate_models():
    print(f"🚀 Starting model migration...")
    print(f"📁 TMP_DIR: {TMP_DIR}")
    print(f"📁 RUNS_DIR: {RUNS_DIR}")

    if not os.path.exists(TMP_DIR):
        print("❌ TMP_DIR not found. Nothing to migrate.")
        return

    os.makedirs(RUNS_DIR, exist_ok=True)

    moved_count = 0

    for filename in os.listdir(TMP_DIR):
        if not filename.endswith("_model.pkl"):
            continue

        job_id = filename.replace("_model.pkl", "")
        src_path = os.path.join(TMP_DIR, filename)

        if not os.path.isfile(src_path):
            continue

        dest_dir = os.path.join(RUNS_DIR, job_id, "artifacts")
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            print(f"❌ Failed to create directory for {job_id}: {e}")
            continue

        dest_path = os.path.join(dest_dir, "model.pkl")

        print(f"📦 Moving {filename} -> runs/{job_id}/artifacts/model.pkl")

        try:
            shutil.copy2(src_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"❌ Failed to move {filename}: {e}")

    print(f"✅ Migration complete. Moved {moved_count} models.")


if __name__ == "__main__":
    try:
        migrate_models()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
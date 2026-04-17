#!/bin/bash

# 🛡️ ROBUST START SCRIPT FOR HUGGING FACE SPACES
# This script launches Redis, FastAPI, Celery, and Streamlit in one container.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
export HOME="/home/user"

# 1. Start Redis (Internal Broker)
# We run it in the background using & and point it to /tmp for write access.
echo "Starting Redis..."
redis-server --port 6379 --dir /tmp --dbfilename dump.rdb --daemonize no &
REDIS_PID=$!

# 2. Start FastAPI Backend (Public Gateway)
echo "Starting FastAPI Backend on port ${PORT:-7860}..."
cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}" --timeout-keep-alive 600 &
BACKEND_PID=$!

# 3. Start Celery Worker (Task Processor)
echo "Starting Celery Worker..."
cd "$SCRIPT_DIR/backend"
python -m celery -A core.worker.celery_app worker --loglevel=info --concurrency=1 &
WORKER_PID=$!

# 4. Start Streamlit Frontend (Internal Service)
# This port (8001) will be proxied by FastAPI.
echo "Starting Streamlit Frontend on internal port 8001..."
cd "$SCRIPT_DIR/frontend"
python -m streamlit run app.py --server.port 8001 --server.headless true --server.address 127.0.0.1 &
FRONTEND_PID=$!

echo "🚀 All services launched!"

# Keep the script alive. If the Frontend or Backend fails, the container should stop.
wait -n $FRONTEND_PID $BACKEND_PID

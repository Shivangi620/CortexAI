#!/bin/bash

# 🛡️ ROBUST START SCRIPT FOR HUGGING FACE SPACES
# This script launches Redis, FastAPI, Celery, and Streamlit in one container.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
export HOME="/home/user"

# 1. Start Redis (Internal Broker)
echo "Starting Redis..."
redis-server --port 6379 --dir /tmp --dbfilename dump.rdb --daemonize no &
REDIS_PID=$!

# 2. Start FastAPI Backend (Internal API)
echo "Starting FastAPI Backend on port 8000..."
cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 600 &
BACKEND_PID=$!

# 3. Start Celery Worker (Task Processor)
echo "Starting Celery Worker..."
cd "$SCRIPT_DIR/backend"
python -m celery -A core.worker.celery_app worker --loglevel=info --concurrency=1 &
WORKER_PID=$!

# 4. Start Streamlit Frontend (Internal UI)
echo "Starting Streamlit Frontend on port 8501..."
cd "$SCRIPT_DIR/frontend"
python -m streamlit run app.py --server.port 8501 --server.headless true --server.address 127.0.0.1 &
FRONTEND_PID=$!

# 5. Start Nginx (Public Gateway)
echo "Starting Nginx on port ${PORT:-7860}..."
# We use -c to point to our custom config
nginx -c "$SCRIPT_DIR/nginx.conf" &
NGINX_PID=$!

echo "🚀 All services launched!"

# Keep the script alive.
wait -n $NGINX_PID $FRONTEND_PID $BACKEND_PID

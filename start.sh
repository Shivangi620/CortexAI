#!/bin/bash
set -euo pipefail

# This script launches Redis, FastAPI, Celery, Streamlit, and Nginx in one HF Space container.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HOME="${HOME:-/home/user}"
export PYTHONPATH="$SCRIPT_DIR/backend:${PYTHONPATH:-}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-$REDIS_URL}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-$REDIS_URL}"
export AUTOML_API_URL="${AUTOML_API_URL:-http://127.0.0.1:8000/api}"
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-500}"
export PORT="${PORT:-7860}"
export STREAMLIT_ENABLE_CORS="${STREAMLIT_ENABLE_CORS:-false}"
export STREAMLIT_ENABLE_XSRF_PROTECTION="${STREAMLIT_ENABLE_XSRF_PROTECTION:-false}"

mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi

cleanup() {
    local exit_code=$?
    echo "Shutting down services..."
    kill "${NGINX_PID:-}" "${FRONTEND_PID:-}" "${WORKER_PID:-}" "${BACKEND_PID:-}" "${REDIS_PID:-}" 2>/dev/null || true
    wait "${NGINX_PID:-}" "${FRONTEND_PID:-}" "${WORKER_PID:-}" "${BACKEND_PID:-}" "${REDIS_PID:-}" 2>/dev/null || true
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting Redis..."
redis-server --port 6379 --dir /tmp --dbfilename dump.rdb --daemonize no &
REDIS_PID=$!

echo "Starting FastAPI backend on port 8000..."
cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 600 &
BACKEND_PID=$!

echo "Starting Celery worker..."
cd "$SCRIPT_DIR/backend"
python -m celery -A core.worker.celery_app worker --loglevel=info --concurrency=1 &
WORKER_PID=$!

echo "Starting Streamlit frontend on port 8501..."
cd "$SCRIPT_DIR/frontend"
python -m streamlit run app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS "${STREAMLIT_ENABLE_CORS}" \
    --server.enableXsrfProtection "${STREAMLIT_ENABLE_XSRF_PROTECTION}" \
    --server.maxUploadSize "${MAX_UPLOAD_MB}" \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

echo "Starting Nginx on port ${PORT:-7860}..."
cd "$SCRIPT_DIR"
sed "s/__PORT__/${PORT}/g" "$SCRIPT_DIR/nginx.conf" > /tmp/nginx.generated.conf
nginx -c /tmp/nginx.generated.conf &
NGINX_PID=$!

echo "All services launched."
wait -n "$NGINX_PID" "$FRONTEND_PID" "$WORKER_PID" "$BACKEND_PID" "$REDIS_PID"

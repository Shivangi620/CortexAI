#!/bin/bash
set -euo pipefail

# This script launches Redis, FastAPI, Celery, and Nginx in one HF Space container.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HOME="${HOME:-/home/user}"
export PYTHONPATH="$SCRIPT_DIR/backend:${PYTHONPATH:-}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-$REDIS_URL}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-$REDIS_URL}"
export PORT="${PORT:-7860}"
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-500}"

mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi

cleanup() {
    local exit_code=$?
    echo "Shutting down services..."
    kill "${NGINX_PID:-}" "${WORKER_PID:-}" "${BACKEND_PID:-}" "${REDIS_PID:-}" "${TAIL_PID:-}" 2>/dev/null || true
    wait "${NGINX_PID:-}" "${WORKER_PID:-}" "${BACKEND_PID:-}" "${REDIS_PID:-}" "${TAIL_PID:-}" 2>/dev/null || true
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting Redis..."
redis-server --port 6379 --dir /tmp --dbfilename dump.rdb --daemonize no > /tmp/redis.log 2>&1 &
REDIS_PID=$!

echo "Starting FastAPI backend on port 8000..."
cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!

echo "Starting Celery worker..."
cd "$SCRIPT_DIR/backend"
python -m celery -A core.worker.celery_app worker --loglevel=info --concurrency=1 > /tmp/worker.log 2>&1 &
WORKER_PID=$!

# Tail logs in background to ensure they appear in HF Logs tab
tail -f /tmp/backend.log /tmp/worker.log /tmp/redis.log &
TAIL_PID=$!

echo "Waiting for FastAPI to be ready on port 8000..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null; then
        echo "FastAPI is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "Starting Nginx on port ${PORT:-7860}..."
cd "$SCRIPT_DIR"
sed "s/__PORT__/${PORT}/g" "$SCRIPT_DIR/nginx.conf" > /tmp/nginx.generated.conf
nginx -c /tmp/nginx.generated.conf &
NGINX_PID=$!

echo "All services launched."
wait -n "$NGINX_PID" "$WORKER_PID" "$BACKEND_PID" "$REDIS_PID"


#!/bin/bash
set -euo pipefail

# Supported deployed/container launcher.
# The Docker image already builds the React bundle during image build.
# Optional:
#   CODIN_BUILD_FRONTEND_ON_START=1 -> rebuild frontend assets at container startup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HOME="${HOME:-/home/user}"
export PYTHONPATH="$SCRIPT_DIR/backend:${PYTHONPATH:-}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-$REDIS_URL}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-$REDIS_URL}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-7860}"
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-500}"
export CODIN_BUILD_FRONTEND_ON_START="${CODIN_BUILD_FRONTEND_ON_START:-0}"

if [[ "$CODIN_BUILD_FRONTEND_ON_START" == "1" ]] && command -v npm >/dev/null 2>&1 && [[ -f "$SCRIPT_DIR/package.json" ]]; then
    if [[ ! -d "$SCRIPT_DIR/node_modules" ]]; then
        echo "Installing frontend dependencies..."
        if [[ -f "$SCRIPT_DIR/package-lock.json" ]]; then
            (cd "$SCRIPT_DIR" && npm ci)
        else
            (cd "$SCRIPT_DIR" && npm install)
        fi
    fi
    echo "Rebuilding React frontend bundle at container startup..."
    (cd "$SCRIPT_DIR" && npm run build:frontend)
fi

cleanup() {
    local exit_code=$?
    echo "Shutting down services..."
    kill "${WORKER_PID:-}" "${BACKEND_PID:-}" "${REDIS_PID:-}" "${TAIL_PID:-}" 2>/dev/null || true
    wait "${WORKER_PID:-}" "${BACKEND_PID:-}" "${REDIS_PID:-}" "${TAIL_PID:-}" 2>/dev/null || true
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting Redis..."
redis-server --port 6379 --dir /tmp --dbfilename dump.rdb --daemonize no > /tmp/redis.log 2>&1 &
REDIS_PID=$!

echo "Starting FastAPI backend on ${HOST}:${PORT}..."
cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host "$HOST" --port "$PORT" --timeout-keep-alive 600 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!

echo "Starting Celery worker..."
cd "$SCRIPT_DIR/backend"
python -m celery -A core.worker.celery_app worker --loglevel=info --concurrency=1 > /tmp/worker.log 2>&1 &
WORKER_PID=$!

# Tail logs in background to ensure they appear in HF Logs tab
tail -f /tmp/backend.log /tmp/worker.log /tmp/redis.log &
TAIL_PID=$!

echo "Waiting for FastAPI to be ready on port ${PORT}..."
for i in {1..30}; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" > /dev/null; then
        echo "FastAPI is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "All services launched."
echo "Studio is expected at http://127.0.0.1:${PORT}/overview"
wait -n "$WORKER_PID" "$BACKEND_PID" "$REDIS_PID"

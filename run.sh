#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting CODIN Neural Studio..."

# ── Trap defined FIRST so Ctrl+C always cleans up ────────────────────────────
CELERY_PID=""
BACKEND_PID=""
REDIS_STARTED_BY_SCRIPT="false"
cleanup() {
    echo ""
    echo "Stopping all services..."
    [ -n "$CELERY_PID" ]   && kill "$CELERY_PID"   2>/dev/null || true
    [ -n "$BACKEND_PID" ]  && kill "$BACKEND_PID"  2>/dev/null || true
    if [[ "$REDIS_STARTED_BY_SCRIPT" == "true" ]]; then
        redis-cli shutdown >/dev/null 2>&1 || true
    fi
    echo "Done."
    exit 0
}

trap cleanup INT TERM

# ── Virtual Environment ───────────────────────────────────────────────────────
if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

export PYTHONPATH="$SCRIPT_DIR/backend:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-$REDIS_URL}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-$REDIS_URL}"
export AUTOML_SKIP_MONGO_MIGRATION="${AUTOML_SKIP_MONGO_MIGRATION:-true}"
mkdir -p "$MPLCONFIGDIR"

# ── React Frontend Build ──────────────────────────────────────────────────────
if command -v npm >/dev/null 2>&1; then
    if [[ ! -d "node_modules" ]]; then
        echo "Installing frontend dependencies..."
        if [[ -f "package-lock.json" ]]; then
            npm ci
        else
            npm install
        fi
    fi
    echo "Building React frontend bundle..."
    npm run build:frontend
else
    echo "❌ npm is required to build the React frontend."
    exit 1
fi

# ── Redis ─────────────────────────────────────────────────────────────────────
echo "Starting Redis..."
if ! redis-cli ping &>/dev/null; then
    LC_ALL=C redis-server --daemonize yes
    sleep 1
    if ! redis-cli ping &>/dev/null; then
        echo "❌ Redis failed to start. Install with: sudo apt install redis-server"
        exit 1
    fi
    REDIS_STARTED_BY_SCRIPT="true"
    echo "✅ Redis is up."
else
    echo "✅ Redis already running."
fi

# ── Celery Worker — must run from backend/ dir ───────────────────────────────
echo "Starting Celery Worker..."
cd "$SCRIPT_DIR/backend"
python -m celery -A core.worker.celery_app worker --loglevel=warning --concurrency=2 &
CELERY_PID=$!
sleep 1
# streamlit
#streamlit run ../frontend/ui/app.py --server.headless=true --server.enableCORS=false --server.runOnSave=true --port 8502
# ── FastAPI Backend ───────────────────────────────────────────────────────────
echo "Starting FastAPI Backend..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600 &
BACKEND_PID=$!

# Wait for FastAPI to be healthy
echo "Waiting for backend to be ready..."
for i in {1..20}; do
    if curl -sf http://localhost:8000/health &>/dev/null; then
        echo "✅ Backend is up."
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "⚠️  Backend did not start in time. Check for errors above."
    fi
    sleep 1
done

echo ""
echo "================================================"
echo "  CODIN Neural Studio is live!"
echo "  Frontend + API:  http://localhost:8000"
echo "  Studio Home:     http://localhost:8000/overview"
echo "  API Docs:        http://localhost:8000/docs"
echo "  Press [CTRL+C] to stop all services."
echo "================================================"
echo " "

wait

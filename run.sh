#!/bin/bash
# AutoML Studio V2 — production launcher
# No set -e: we handle errors ourselves to avoid killing the whole script on any non-zero return

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting AutoML Studio V2..."

# ── Trap defined FIRST so Ctrl+C always cleans up ────────────────────────────
CELERY_PID=""
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "Stopping all services..."
    [ -n "$CELERY_PID" ]   && kill "$CELERY_PID"   2>/dev/null || true
    [ -n "$BACKEND_PID" ]  && kill "$BACKEND_PID"  2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
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

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# ── Redis ─────────────────────────────────────────────────────────────────────
echo "Starting Redis..."
if ! redis-cli ping &>/dev/null; then
    LC_ALL=C redis-server --daemonize yes
    sleep 1
    if ! redis-cli ping &>/dev/null; then
        echo "❌ Redis failed to start. Install with: sudo apt install redis-server"
        exit 1
    fi
    echo "✅ Redis is up."
else
    echo "✅ Redis already running."
fi

# ── Celery Worker — must run from backend/ dir ───────────────────────────────
echo "Starting Celery Worker..."
cd "$SCRIPT_DIR/backend"
celery -A core.worker worker --loglevel=warning --concurrency=2 &
CELERY_PID=$!
sleep 1

# ── FastAPI Backend ───────────────────────────────────────────────────────────
echo "Starting FastAPI Backend..."
# Increase timeout and limit to handle large file uploads/processing
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600 &
BACKEND_PID=$!

# Wait for FastAPI to be healthy
echo "Waiting for backend to be ready..."
for i in {1..20}; do
    if curl -sf http://localhost:8000/docs &>/dev/null; then
        echo "✅ Backend is up."
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "⚠️  Backend did not start in time. Check for errors above."
    fi
    sleep 1
done

echo "Starting Streamlit Frontend..."
cd "$SCRIPT_DIR/frontend"
python -m streamlit run app.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "================================================"
echo "  AutoML Studio is live!"
echo "  Frontend:  http://localhost:8501"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Press [CTRL+C] to stop all services."
echo "================================================"
echo " "

wait

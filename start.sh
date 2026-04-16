#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
  echo "Shutting down services..."
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
  [ -n "$REDIS_PID" ] && kill "$REDIS_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"

# Start Redis locally if available
if command -v redis-server >/dev/null 2>&1; then
  echo "Starting local Redis..."
  redis-server --daemonize yes
  sleep 1
  if ! redis-cli ping >/dev/null 2>&1; then
    echo "Redis failed to start. Continuing without local Redis."
  else
    echo "Redis is running."
  fi
else
  echo "redis-server not installed; skipping local Redis startup."
fi

# Start backend
cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600 &
BACKEND_PID=$!

echo "Waiting for backend to start..."
for i in {1..20}; do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "Backend ready."
    break
  fi
  sleep 1
done

# Start frontend
cd "$SCRIPT_DIR/frontend"
python -m streamlit run app.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!

echo "Frontend starting on port 8501..."

echo "AutoML Studio services are now running."
echo "Backend: http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:8501"

tail --pid="$BACKEND_PID" -f /dev/null
wait -n "$BACKEND_PID" "$FRONTEND_PID"

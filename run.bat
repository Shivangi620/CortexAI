@echo off
setlocal

echo 🚀 Starting AutoML Studio V2 (Windows)...

:: 1. Check for Virtual Environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
    pip install eventlet
) else (
    call venv\Scripts\activate
)

:: 2. Redis Check
redis-cli ping >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Redis is not running! 
    echo Please install Memurai or Redis (WSL) and ensure it is started.
    pause
    exit /b 1
)
echo ✅ Redis is up.

:: 3. Kill potentially hanging processes from previous runs
taskkill /IM "celery.exe" /F >nul 2>&1
taskkill /IM "uvicorn.exe" /F >nul 2>&1
taskkill /IM "streamlit.exe" /F >nul 2>&1

:: 4. Start Celery Worker (Windows requires -P eventlet or solo)
echo Starting Celery Worker...
cd backend
start /B celery -A core.worker worker --loglevel=info -P eventlet --concurrency=2
cd ..

:: 5. Start FastAPI Backend
echo Starting FastAPI Backend...
cd backend
start /B uvicorn main:app --host 0.0.0.0 --port 8000
cd ..

:: 6. Start Streamlit Frontend
echo Starting Streamlit Frontend...
streamlit run frontend/app.py --server.headless true --server.port 8501

pause

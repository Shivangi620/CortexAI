@echo off
setlocal enabledelayedexpansion

REM Supported local launcher for Windows.
REM Optional:
REM   set CODIN_RUN_QUALITY=1

cd /d "%~dp0"
echo Starting Inferyx on Windows...

if not exist venv (
  echo Creating virtual environment...
  py -m venv venv
)

call venv\Scripts\activate.bat
if errorlevel 1 (
  echo Failed to activate the virtual environment.
  exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

where npm >nul 2>nul
if errorlevel 1 (
  echo npm is required to build the frontend.
  exit /b 1
)

if not exist node_modules (
  if exist package-lock.json (
    call npm ci
  ) else (
    call npm install
  )
)

if /I "%CODIN_RUN_QUALITY%"=="1" (
  call npm run quality
) else (
  call npm run build:frontend
)
if errorlevel 1 exit /b 1

echo.
echo Redis is required for background training.
echo Recommended: Docker Desktop + Redis container
echo docker run -d --name codin-redis -p 6379:6379 redis:7-alpine
echo.

set PYTHONPATH=%CD%\backend;%PYTHONPATH%
set MPLCONFIGDIR=%TEMP%\matplotlib
set REDIS_URL=redis://127.0.0.1:6379/0
set CELERY_BROKER_URL=%REDIS_URL%
set CELERY_RESULT_BACKEND=%REDIS_URL%
set AUTOML_SKIP_MONGO_MIGRATION=true

start "CODIN Celery Worker" cmd /k "cd /d %CD%\backend && ..\venv\Scripts\python.exe -m celery -A core.worker.celery_app worker --loglevel=warning --pool=solo"
start "CODIN FastAPI" cmd /k "cd /d %CD%\backend && ..\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600"

echo.
echo Inferyx should be available at:
echo http://localhost:8000
echo http://localhost:8000/overview
echo http://localhost:8000/results
echo http://localhost:8000/docs
echo.
echo Close the spawned terminal windows to stop the backend and worker.

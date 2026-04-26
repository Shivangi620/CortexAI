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
set PYTHONPATH=%CD%\backend;%PYTHONPATH%
set MPLCONFIGDIR=%TEMP%\matplotlib
if "%REDIS_URL%"=="" set REDIS_URL=redis://127.0.0.1:6379/0
if "%CELERY_BROKER_URL%"=="" set CELERY_BROKER_URL=%REDIS_URL%
if "%CELERY_RESULT_BACKEND%"=="" set CELERY_RESULT_BACKEND=%REDIS_URL%
set AUTOML_SKIP_MONGO_MIGRATION=true

python -c "import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', 6379)) == 0 else 1)"
if errorlevel 1 (
  echo Redis was not detected on 127.0.0.1:6379.
  echo Start a local Redis service, or set REDIS_URL / CELERY_BROKER_URL / CELERY_RESULT_BACKEND first.
  echo The API will still start, but background training will be unavailable until Redis is running.
  echo.
) else (
  start "CODIN Celery Worker" cmd /k "cd /d %CD%\backend && ..\venv\Scripts\python.exe -m celery -A core.worker.celery_app worker --loglevel=warning --pool=solo"
)

start "CODIN FastAPI" cmd /k "cd /d %CD%\backend && ..\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600"

echo.
echo Inferyx should be available at:
echo http://localhost:8000
echo http://localhost:8000/overview
echo http://localhost:8000/results
echo http://localhost:8000/docs
echo.
echo Close the spawned terminal windows to stop the backend and worker.

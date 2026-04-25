@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo start_windows.bat is now a compatibility wrapper.
echo Delegating to run_windows.bat, which is the supported Windows launcher.
call "%~dp0run_windows.bat"
endlocal & exit /b %errorlevel%

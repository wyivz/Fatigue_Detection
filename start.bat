@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ROOT=%cd%"
set "PY=%ROOT%\venv\Scripts\python.exe"
set "APP=%ROOT%\fatigue_detection_system"
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

echo ========================================
echo  SleepyDetect - Start
echo  Root: %ROOT%
echo  URL : http://127.0.0.1:%PORT%/
echo ========================================

if not exist "%PY%" (
  echo [ERROR] venv python not found:
  echo   %PY%
  pause
  exit /b 1
)

if not exist "%APP%\manage.py" (
  echo [ERROR] manage.py not found:
  echo   %APP%\manage.py
  pause
  exit /b 1
)

if not exist "%ROOT%\stop.ps1" (
  echo [ERROR] stop.ps1 not found:
  echo   %ROOT%\stop.ps1
  pause
  exit /b 1
)

echo [1/2] Stopping old SleepyDetect processes...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\stop.ps1" -Quiet

echo [2/2] Starting Django on port %PORT% ...
echo Keep this window open. Press Ctrl+C to stop.
echo.
cd /d "%APP%"
"%PY%" manage.py runserver 127.0.0.1:%PORT%
set "EC=%ERRORLEVEL%"
echo.
if not "%EC%"=="0" (
  echo [ERROR] Server exited with code %EC%
  pause
)
endlocal
exit /b %EC%
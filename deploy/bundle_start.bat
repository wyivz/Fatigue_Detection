@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ROOT=%cd%"
set "PY=%ROOT%\venv\Scripts\python.exe"
set "APP=%ROOT%\app"
if not exist "%APP%\manage.py" set "APP=%ROOT%\fatigue_detection_system"
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

echo ========================================
echo  SleepyDetect - Start
echo  Root: %ROOT%
echo  URL : http://127.0.0.1:%PORT%/
echo ========================================

if not exist "%ROOT%\.install_ok" (
  echo [ERROR] Not installed yet. Run install.bat first.
  pause
  exit /b 1
)

if not exist "%PY%" (
  echo [ERROR] venv python not found: %PY%
  echo Run install.bat first.
  pause
  exit /b 1
)

if not exist "%APP%\manage.py" (
  echo [ERROR] manage.py not found under app\
  pause
  exit /b 1
)

REM Prefer portable Python home if present
if exist "%ROOT%\runtime\Python38\python.exe" (
  echo home = %ROOT%\runtime\Python38> "%ROOT%\venv\pyvenv.cfg"
  echo include-system-site-packages = false>> "%ROOT%\venv\pyvenv.cfg"
  echo version = 3.8.10>> "%ROOT%\venv\pyvenv.cfg"
)

if exist "%ROOT%\stop.ps1" (
  echo [1/2] Stopping old SleepyDetect processes...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\stop.ps1" -Quiet
)

echo [2/2] Starting Django on port %PORT% ...
echo Keep this window open. Press Ctrl+C to stop.
echo.
cd /d "%APP%"
"%PY%" manage.py migrate --noinput
"%PY%" manage.py runserver 127.0.0.1:%PORT%
set "EC=%ERRORLEVEL%"
echo.
if not "%EC%"=="0" (
  echo [ERROR] Server exited with code %EC%
  pause
)
endlocal
exit /b %EC%

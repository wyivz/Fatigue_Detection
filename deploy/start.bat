@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist "%ROOT%runtime\Python38\python.exe" (
  echo [ERROR] missing runtime\Python38\python.exe
  pause
  exit /b 1
)
if not exist "%ROOT%venv\Scripts\python.exe" (
  echo [ERROR] missing venv\Scripts\python.exe
  pause
  exit /b 1
)
if not exist "%ROOT%app\manage.py" (
  echo [ERROR] missing app\manage.py
  pause
  exit /b 1
)

echo home = %ROOT%runtime\Python38> "%ROOT%venv\pyvenv.cfg"
echo include-system-site-packages = false>> "%ROOT%venv\pyvenv.cfg"
echo version = 3.8.10>> "%ROOT%venv\pyvenv.cfg"

set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

echo ========================================
echo  SleepyDetect portable start
echo  ROOT: %ROOT%
echo  URL:  http://127.0.0.1:%PORT%/
echo  Keep this window open
echo  For GigE: install Hikrobot MVS 4.6.3 Runtime
echo ========================================

cd /d "%ROOT%app"
"%ROOT%venv\Scripts\python.exe" manage.py migrate --noinput
if errorlevel 1 (
  echo [ERROR] migrate failed
  pause
  exit /b 1
)

"%ROOT%venv\Scripts\python.exe" manage.py runserver 127.0.0.1:%PORT%
echo [ERROR] server exited
pause
endlocal

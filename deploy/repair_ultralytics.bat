@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo Repair ultralytics + pyvenv.cfg (offline)...
if not exist "%~dp0venv\Scripts\python.exe" (
  echo [ERROR] venv missing. Run install.bat first.
  pause
  exit /b 1
)
if exist "%~dp0runtime\Python38\python.exe" (
  echo home = %~dp0runtime\Python38> "%~dp0venv\pyvenv.cfg"
  echo include-system-site-packages = false>> "%~dp0venv\pyvenv.cfg"
  echo version = 3.8.10>> "%~dp0venv\pyvenv.cfg"
)
"%~dp0venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps --no-index --find-links "%~dp0wheels" ultralytics
if errorlevel 1 (
  echo [ERROR] reinstall failed
  pause
  exit /b 1
)
"%~dp0venv\Scripts\python.exe" -c "from ultralytics import YOLO; import ultralytics; print('OK', ultralytics.__version__)"
if errorlevel 1 (
  echo [ERROR] still broken. Delete venv folder and run install.bat again.
  pause
  exit /b 1
)
echo Repair done. Run start.bat
pause
endlocal

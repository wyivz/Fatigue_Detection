@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo  SleepyDetect - Train Behavior Model
echo ========================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_behavior.ps1" %*
set "EC=%ERRORLEVEL%"
echo.
if not "%EC%"=="0" (
  echo [ERROR] Training failed. Exit code: %EC%
  pause
  exit /b %EC%
)

echo Training finished. Restart start.bat to load the new weights.
pause
endlocal
exit /b 0

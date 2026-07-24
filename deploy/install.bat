@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo  SleepyDetect - One-click Install
echo ========================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "EC=%ERRORLEVEL%"
echo.
if not "%EC%"=="0" (
  echo [ERROR] Install failed. Exit code: %EC%
  pause
  exit /b %EC%
)

echo.
echo Install finished. You can run start.bat to launch.
pause
endlocal
exit /b 0

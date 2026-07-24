@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%cd%"

echo ========================================
echo  SleepyDetect - Stop
echo  Root: %ROOT%
echo ========================================

if not exist "%ROOT%\stop.ps1" (
  echo [ERROR] stop.ps1 not found:
  echo   %ROOT%\stop.ps1
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\stop.ps1"
echo.
pause
endlocal
exit /b 0
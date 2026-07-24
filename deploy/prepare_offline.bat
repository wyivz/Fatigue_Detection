@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
echo Build OFFLINE install bundle (needs network on THIS PC)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0prepare_install_bundle.ps1" -Offline %*
echo.
pause
endlocal

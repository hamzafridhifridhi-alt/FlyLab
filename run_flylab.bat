@echo off
setlocal enabledelayedexpansion
title FlyLab Web Simulation Server

cd /d "%~dp0"

echo ===================================================
echo     FlyLab Connectome + MuJoCo Web Simulation
echo ===================================================
echo.

if not exist "data\derived\brain_points.bin" (
    echo [WARNING] Connectome data artifacts not found.
    echo [INFO] Running data bootstrap and connectome build first...
    call bootstrap.bat
    if !errorlevel! neq 0 (
        echo [ERROR] Bootstrap failed.
        pause
        exit /b !errorlevel!
    )
)

echo [INFO] Launching FlyLab simulation server...
echo [INFO] Opening http://127.0.0.1:8000 in your browser...
echo.

start "" cmd /c "timeout /t 3 >nul && start http://127.0.0.1:8000"

where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run flylab
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m flylab
) else (
    python -m flylab
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server exited with error code %errorlevel%.
)

echo.
pause

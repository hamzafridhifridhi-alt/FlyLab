@echo off
setlocal enabledelayedexpansion
title MuJoCo Bitcraze Crazyflie 2 Drone 3D Viewer

cd /d "%~dp0"

echo ===================================================
echo   Bitcraze Crazyflie 2 Drone MuJoCo 3D Viewer
echo ===================================================
echo.

where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Running via uv...
    uv run python scripts\launch_drone_mujoco.py
) else if exist ".venv\Scripts\python.exe" (
    echo [INFO] Running via local .venv...
    ".venv\Scripts\python.exe" scripts\launch_drone_mujoco.py
) else (
    echo [INFO] Running via system python...
    python scripts\launch_drone_mujoco.py
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Simulation exited with error code %errorlevel%.
)

echo.
pause

@echo off
setlocal enabledelayedexpansion
title MuJoCo Fly 3D Viewer

cd /d "%~dp0"

echo ===================================================
echo        MuJoCo Fly Simulation 3D Viewer
echo ===================================================
echo.
echo Controls (in console window):
echo   W : Walk forward
echo   A : Turn left
echo   D : Turn right
echo   S : Stop
echo   Q : Quit
echo.

where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Running via uv...
    uv run python scripts\launch_mujoco_viewer.py
) else if exist ".venv\Scripts\python.exe" (
    echo [INFO] Running via local .venv...
    ".venv\Scripts\python.exe" scripts\launch_mujoco_viewer.py
) else (
    echo [INFO] Running via system python...
    python scripts\launch_mujoco_viewer.py
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Simulation exited with error code %errorlevel%.
)

echo.
pause

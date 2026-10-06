@echo off
setlocal enabledelayedexpansion
title Bitcraze Crazyflie 2 Drone MuJoCo Simulation Launcher

cd /d "%~dp0"

:MENU
cls
echo ================================================================
echo    Bitcraze Crazyflie 2 Drone - MuJoCo Simulation Launcher
echo    (Piloted by Drosophila Neural Network & Plant Anomaly E-Nose)
echo ================================================================
echo.
echo Select an option to run:
echo.
echo   [1] Launch Interactive 3D MuJoCo Desktop Viewer (Crazyflie 2)
echo   [2] Launch Full FlyLab Web App ^& Connectome Server (http://127.0.0.1:8000)
echo   [3] Exit
echo.
echo ================================================================
echo.

set /p CHOICE="Enter your choice [1-3]: "

if "%CHOICE%"=="1" (
    echo.
    echo [INFO] Launching 3D MuJoCo Desktop Viewport for Crazyflie 2...
    where uv >nul 2>&1
    if !errorlevel! equ 0 (
        uv run python scripts\launch_drone_mujoco.py
    ) else if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" scripts\launch_drone_mujoco.py
    ) else (
        python scripts\launch_drone_mujoco.py
    )
    echo.
    pause
    goto MENU
)

if "%CHOICE%"=="2" (
    echo.
    echo [INFO] Launching FlyLab Web Simulation Server...
    call run_flylab.bat
    goto MENU
)

if "%CHOICE%"=="3" (
    exit /b 0
)

echo.
echo Invalid selection. Please enter 1, 2, or 3.
timeout /t 2 >nul
goto MENU

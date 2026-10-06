@echo off
setlocal enabledelayedexpansion
title MuJoCo Fly Simulation Launcher

cd /d "%~dp0"

:MENU
cls
echo ================================================================
echo               MuJoCo Fly Lab Simulation Launcher
echo ================================================================
echo.
echo Select an option to run:
echo.
echo   [1] Interactive 3D MuJoCo Desktop Viewer (Keyboard W/A/S/D/Q)
echo   [2] Full FlyLab Web App ^& Connectome Simulation (http://127.0.0.1:8000)
echo   [3] Run Data Bootstrap ^& Connectome Build
echo   [4] Run Stack Verification Tests
echo   [5] Exit
echo.
echo ================================================================
echo.

set /p CHOICE="Enter your choice [1-5]: "

if "%CHOICE%"=="1" (
    echo.
    call run_mujoco_viewer.bat
    goto MENU
)
if "%CHOICE%"=="2" (
    echo.
    call run_flylab.bat
    goto MENU
)
if "%CHOICE%"=="3" (
    echo.
    call bootstrap.bat
    goto MENU
)
if "%CHOICE%"=="4" (
    echo.
    echo [INFO] Running stack verification...
    where uv >nul 2>&1
    if !errorlevel! equ 0 (
        uv run python scripts\verify_stack.py
    ) else (
        python scripts\verify_stack.py
    )
    echo.
    pause
    goto MENU
)
if "%CHOICE%"=="5" (
    exit /b 0
)

echo.
echo Invalid selection. Please enter a number between 1 and 5.
timeout /t 2 >nul
goto MENU

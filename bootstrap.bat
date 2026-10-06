@echo off
setlocal enabledelayedexpansion
title FlyLab Setup & Data Bootstrap

cd /d "%~dp0"

echo ===================================================
echo    FlyLab Data Bootstrap ^& Connectome Artifact Builder
echo ===================================================
echo.

where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [1/3] Syncing python dependencies with uv...
    uv sync
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo.
    echo [2/3] Fetching external data...
    uv run python scripts\bootstrap_data.py
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo.
    echo [3/3] Building connectome artifacts...
    uv run python src\flylab\build_connectome.py
    if !errorlevel! neq 0 exit /b !errorlevel!
) else (
    echo [1/2] Fetching external data...
    python scripts\bootstrap_data.py
    if !errorlevel! neq 0 exit /b !errorlevel!

    echo.
    echo [2/2] Building connectome artifacts...
    python src\flylab\build_connectome.py
    if !errorlevel! neq 0 exit /b !errorlevel!
)

echo.
echo [SUCCESS] FlyLab setup completed successfully!
echo.
pause

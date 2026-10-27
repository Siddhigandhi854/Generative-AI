@echo off
title Generative AI Lab - Interactive Platform
echo =====================================================================
echo                GENERATIVE AI LAB - INTERACTIVE PLATFORM
echo               Guide Teacher: Dr. Sachin Takmare
echo =====================================================================
echo.
echo Activating Python virtual environment and starting FastAPI server...
echo.

cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    start "" http://127.0.0.1:8000
    venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    echo [ERROR] Virtual environment not found. Please run setup first.
    pause
)

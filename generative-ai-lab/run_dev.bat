@echo off
title Generative AI Lab - Development Mode
echo =====================================================================
echo       GENERATIVE AI LAB - DEVELOPMENT MODE (HOT RELOAD)
echo               Guide Teacher: Dr. Sachin Takmare
echo =====================================================================
echo.
cd /d "%~dp0"

echo [1/2] Starting Backend Server on port 8000...
start "GenAI Backend" cmd /k "venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak > nul

echo [2/2] Starting Frontend Vite Server on port 5173...
cd frontend
start "GenAI Frontend" cmd /k "npm run dev"

timeout /t 2 /nobreak > nul
start "" http://localhost:5173
echo Done! Application running at http://localhost:5173

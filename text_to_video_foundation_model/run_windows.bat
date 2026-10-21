@echo off
setlocal
cd /d "%~dp0"

echo =================================================================
echo   Text-to-Video Foundation Studio
echo   Pretrained Diffusion Inference (Zeroscope v2 576w)
echo =================================================================

if not exist .venv (
    echo [1/3] Creating Python virtual environment...
    py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Python 3.10 or 3.11 was not found on PATH.
        echo Please install Python from https://www.python.org/
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

python -c "import diffusers, gradio, torch" 2>nul
if errorlevel 1 (
    echo [2/3] Installing dependencies from requirements.txt...
    python -m pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>nul || pip install torch torchvision
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements.
        pause
        exit /b 1
    )
)

echo [3/3] Starting Web Application...
python app.py

pause

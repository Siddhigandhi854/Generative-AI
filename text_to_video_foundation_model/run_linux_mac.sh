#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "================================================================="
echo "  Text-to-Video Foundation Studio"
echo "  Pretrained Diffusion Inference (Zeroscope v2 576w)"
echo "================================================================="

if [ ! -d ".venv" ]; then
    echo "[1/3] Creating Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import diffusers, gradio, torch" &>/dev/null; then
    echo "[2/3] Installing dependencies..."
    python -m pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "[3/3] Starting Web Application..."
python app.py

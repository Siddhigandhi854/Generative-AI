import os
import sys

# Environment settings for low-memory cloud tiers (Render 512MB free tier)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from routers import (
    exp1_eda,
    exp2_tensors,
    exp3_transformer_qa,
    exp4_diffusion,
    exp5_vae,
    exp6_dcgan,
    exp7_stylegan,
    exp8_musicgen,
    exp9_rag_assistant,
    exp10_video_gen
)

app = FastAPI(
    title="Generative AI Lab Interactive Platform",
    description="Full-stack interactive application powering all 10 Generative AI Experiments",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount generated outputs directory
app.mount("/outputs", StaticFiles(directory=config.output_dir), name="outputs")

# Include Routers for all 10 experiments
app.include_router(exp1_eda.router)
app.include_router(exp2_tensors.router)
app.include_router(exp3_transformer_qa.router)
app.include_router(exp4_diffusion.router)
app.include_router(exp5_vae.router)
app.include_router(exp6_dcgan.router)
app.include_router(exp7_stylegan.router)
app.include_router(exp8_musicgen.router)
app.include_router(exp9_rag_assistant.router)
app.include_router(exp10_video_gen.router)

class SettingsUpdate(BaseModel):
    hf_token: Optional[str] = None
    gemini_key: Optional[str] = None
    openai_key: Optional[str] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Generative AI Lab Backend",
        "guide_teacher": "Dr. Sachin Takmare",
        "active_experiments": 10
    }

@app.get("/api/diagnostics")
def get_diagnostics():
    import platform
    gpu_info = "NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)"
    output_files = list(Path(config.output_dir).glob("*.*"))
    outputs_size_mb = sum(f.stat().st_size for f in output_files) / (1024 * 1024) if output_files else 0.0

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_accelerator": gpu_info,
        "cuda_version": "12.7",
        "active_outputs_count": len(output_files),
        "outputs_size_mb": round(outputs_size_mb, 2),
        "keys_status": config.get_status()
    }

@app.get("/api/settings")
def get_settings():
    return config.get_status()

@app.post("/api/settings")
def update_settings(req: SettingsUpdate):
    config.update_keys(
        hf_token=req.hf_token,
        gemini_key=req.gemini_key,
        openai_key=req.openai_key
    )
    return {
        "message": "Settings updated successfully.",
        "status": config.get_status()
    }

# Mount static React frontend build if available
dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(dist_dir):
    from fastapi.responses import FileResponse
    assets_dir = os.path.join(dist_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("outputs/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        file_path = os.path.join(dist_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


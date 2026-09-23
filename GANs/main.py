import os
import io
import base64
import logging
import zipfile
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from generator import Generator
from pipeline_utils import postprocess_generated_image

# Setup Developer Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gan_face_generator")

# Try loading local .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Environment & Model Configurations
MODEL_PATH = os.getenv("MODEL_PATH", "generator.pth")
MODEL_FILENAME = os.path.basename(MODEL_PATH)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_RAW.split(",") if origin.strip()]

# Global Model & Compute Device Optimization
generator_model: Optional[Generator] = None

def select_optimal_device() -> torch.device:
    """Selects optimal hardware compute device and enables PyTorch inference optimizations."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA GPU detected. Enabled cuDNN benchmarking for 64x64 inference.")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS GPU detected.")
        return torch.device("mps")
    else:
        logger.info("GPU unavailable. Using CPU compute engine.")
        return torch.device("cpu")

device: torch.device = select_optimal_device()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Safely loads the PyTorch Generator model once on server startup.
    """
    global generator_model, device
    logger.info(f"Initializing GAN backend engine on compute device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"[MODEL LOAD ERROR] Model file '{MODEL_FILENAME}' was not found!")
        generator_model = None
    else:
        try:
            netG = Generator(nz=100, ngf=64, nc=3).to(device)
            state_dict = torch.load(MODEL_PATH, map_location=device)
            netG.load_state_dict(state_dict)
            netG.eval()

            # Freeze parameters for safe read-only inference
            for param in netG.parameters():
                param.requires_grad = False

            generator_model = netG
            param_count = sum(p.numel() for p in netG.parameters())
            logger.info(f"[MODEL LOAD SUCCESS] Loaded '{MODEL_FILENAME}' ({param_count:,} parameters).")
        except Exception as e:
            logger.exception(f"[MODEL LOAD FAILURE] Error loading '{MODEL_FILENAME}': {e}")
            generator_model = None
            
    yield
    logger.info("GAN Face Generator API backend shutdown complete.")

# Initialize FastAPI Application
app = FastAPI(
    title="GAN Face Generator API",
    description="FastAPI Backend for Synthetic Human Face Generation using PyTorch DCGAN",
    version="1.6.0",
    lifespan=lifespan
)

# Secure CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)

# Mount Static Files Directories safely
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.exists("outputs"):
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Strict Pydantic Input Validation Schemas
class BatchGenerateRequest(BaseModel):
    count: int = Field(default=4, ge=1, le=16, description="Number of faces to generate (1 to 16)")
    seed: Optional[int] = Field(default=None, ge=0, le=99999999, description="Optional random seed")

class ZipDownloadRequest(BaseModel):
    images: Optional[List[str]] = Field(default=None, max_length=16, description="List of base64 PNG URIs (max 16)")
    count: Optional[int] = Field(default=None, ge=1, le=16, description="Number of faces to generate")
    seed: Optional[int] = Field(default=None, ge=0, le=99999999, description="Random seed")

# STRUCTURED SAFE ERROR HANDLERS (No filesystem path leakage or raw stack traces)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled server exceptions, logs stack trace to server logs, returns safe error response."""
    logger.exception(f"[500 UNHANDLED ERROR] {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred. Please try again later.",
            "status_code": 500
        }
    )

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Formats HTTP exceptions into clean JSON responses."""
    logger.warning(f"[HTTP {exc.status_code}] Path: {request.url.path} | Detail: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "API Request Error",
            "message": str(exc.detail),
            "status_code": exc.status_code
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic input validation failures."""
    logger.warning(f"[422 VALIDATION ERROR] Path: {request.url.path} | Validation Errors: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Invalid request parameters. Batch count must be between 1 and 16.",
            "status_code": 422
        }
    )

# Helper Functions
def extract_latent_stats(latent_tensor: torch.Tensor, used_seed: int):
    """Extracts latent vector statistics for inspection."""
    z_first = latent_tensor[0].squeeze().cpu().numpy()
    return {
        "seed": used_seed,
        "latent_dim": 100,
        "first_5_values": [round(float(v), 4) for v in z_first[:5]],
        "mean": round(float(z_first.mean()), 4),
        "std": round(float(z_first.std()), 4),
        "norm": round(float(np.linalg.norm(z_first)), 4)
    }

# API ENDPOINTS

@app.get("/", response_class=FileResponse, summary="Serve Frontend Dashboard")
async def get_index():
    index_path = os.path.join("templates", "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return JSONResponse({"message": "GAN Face Generator API Server Running."})

@app.get("/health", summary="Health Check")
async def health_check():
    is_loaded = generator_model is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "device": str(device),
        "weights_file": MODEL_FILENAME
    }

@app.get("/model-info", summary="Dynamic Model Specifications")
async def get_model_info():
    is_loaded = generator_model is not None
    return {
        "model_type": "DCGAN (Deep Convolutional GAN)",
        "framework": "PyTorch 2.x",
        "generator_architecture": "5 Transposed Convolutional Layers (ConvTranspose2d + BatchNorm2d + Tanh)",
        "image_dimensions": {
            "resolution": "64 x 64 pixels",
            "width": 64,
            "height": 64,
            "channels": 3,
            "format": "RGB"
        },
        "latent_dimension": 100,
        "training_dataset": "Not available",
        "training_epochs": "Not available",
        "training_batch_size": "Not available",
        "saved_model_file": MODEL_FILENAME,
        "inference_status": f"Active / Model Loaded ({str(device).upper()})" if is_loaded else "Model Not Loaded"
    }

def set_random_seed(seed: int):
    """Sets CPU and CUDA random seeds for reproducible inference."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@app.get("/generate", summary="Generate Single Face (GET)")
@app.post("/generate", summary="Generate Single Face (POST)")
async def generate_single_face(
    seed: Optional[int] = Query(default=None, ge=0, le=99999999, description="Optional random seed"),
    response_format: str = Query(default="json", enum=["json", "png"], description="Response format")
):
    if generator_model is None:
        logger.error("[GENERATE ERROR] Generator model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The trained model '{MODEL_FILENAME}' is unavailable or failed to load."
        )

    used_seed = seed if seed is not None else int(torch.randint(0, 9999999, (1,)).item())
    set_random_seed(used_seed)

    try:
        with torch.inference_mode():
            latent_vector = torch.randn(1, 100, 1, 1, device=device)
            fake_tensor = generator_model(latent_vector)
    except (RuntimeError, MemoryError) as mem_err:
        logger.exception(f"[MEMORY ERROR] {mem_err}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server memory limit exceeded during generation."
        )
    except Exception as inf_err:
        logger.exception(f"[INFERENCE ENGINE FAILURE] {inf_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Neural network inference engine failed during generation."
        )

    try:
        if response_format.lower() == "png":
            png_bytes = postprocess_generated_image(fake_tensor[0], return_format="png_bytes")
            filename = f"gan_face_{used_seed:03d}.png"
            return Response(
                content=png_bytes,
                media_type="image/png",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
        else:
            base64_uri = postprocess_generated_image(fake_tensor[0], return_format="base64")
            stats = extract_latent_stats(latent_vector, used_seed)
            return {
                "success": True,
                "seed": used_seed,
                "latent_dim": 100,
                "latent_stats": stats,
                "image": base64_uri
            }
    except Exception as post_err:
        logger.exception(f"[POSTPROCESS FAILURE] {post_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to convert model tensor output to valid image."
        )

@app.post("/generate-batch", summary="Generate Multiple Faces")
async def generate_batch_faces(request: BatchGenerateRequest):
    if generator_model is None:
        logger.error("[GENERATE BATCH ERROR] Generator model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The trained model '{MODEL_FILENAME}' is unavailable or failed to load."
        )

    count = request.count
    used_seed = request.seed if request.seed is not None else int(torch.randint(0, 9999999, (1,)).item())
    set_random_seed(used_seed)

    try:
        with torch.inference_mode():
            latent_vectors = torch.randn(count, 100, 1, 1, device=device)
            fake_tensors = generator_model(latent_vectors)
    except (RuntimeError, MemoryError) as mem_err:
        logger.exception(f"[BATCH MEMORY ERROR] {mem_err}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server memory limit exceeded. Try requesting fewer images."
        )
    except Exception as inf_err:
        logger.exception(f"[BATCH INFERENCE FAILURE] {inf_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Neural network batch inference failed during generation."
        )

    try:
        base64_images: List[str] = postprocess_generated_image(fake_tensors, return_format="base64")
        stats = extract_latent_stats(latent_vectors, used_seed)
        return {
            "success": True,
            "count": count,
            "seed": used_seed,
            "latent_dim": 100,
            "latent_stats": stats,
            "images": base64_images
        }
    except Exception as post_err:
        logger.exception(f"[BATCH POSTPROCESS FAILURE] {post_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to convert batch model tensors to displayable images."
        )

@app.post("/download-zip", summary="Download Batch Images as ZIP Archive")
async def download_zip(request: ZipDownloadRequest):
    png_bytes_list: List[bytes] = []

    try:
        if request.images and len(request.images) > 0:
            # Enforce max 16 base64 image limit
            if len(request.images) > 16:
                raise HTTPException(status_code=422, detail="ZIP download request exceeds maximum 16 images limit.")
            
            for b64_str in request.images:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                raw_bytes = base64.b64decode(b64_str)
                png_bytes_list.append(raw_bytes)
        else:
            if generator_model is None:
                raise HTTPException(status_code=503, detail="Trained generator model is not loaded.")
            
            count = request.count if request.count else 4
            used_seed = request.seed if request.seed is not None else int(torch.randint(0, 9999999, (1,)).item())
            set_random_seed(used_seed)

            with torch.inference_mode():
                latent_vectors = torch.randn(count, 100, 1, 1, device=device)
                fake_tensors = generator_model(latent_vectors)

            png_bytes_list = postprocess_generated_image(fake_tensors, return_format="png_bytes")

        if len(png_bytes_list) == 0:
            raise HTTPException(status_code=400, detail="No valid images provided or generated for ZIP creation.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, png_bytes in enumerate(png_bytes_list):
                filename = f"gan_face_{idx + 1:03d}.png"
                zip_file.writestr(filename, png_bytes)

        zip_buffer.seek(0)
        archive_name = "gan_faces_batch.zip"
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{archive_name}"'}
        )
    except HTTPException:
        raise
    except Exception as zip_err:
        logger.exception(f"[ZIP DOWNLOAD FAILURE] {zip_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate ZIP download archive."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

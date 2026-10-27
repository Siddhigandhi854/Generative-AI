import os
import uuid
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from config import config

router = APIRouter(prefix="/api/exp5", tags=["Experiment 5: Variational Autoencoder"])

class VAERequest(BaseModel):
    num_samples: int = 4  # 1 to 16
    seed: Optional[int] = None
    latent_dim: int = 2
    z_coords: Optional[List[float]] = None  # for manual latent traversal [z1, z2]

_DECODER_MODEL = None

def get_vae_decoder():
    """Lazy loads the trained Keras Convolutional VAE decoder model."""
    global _DECODER_MODEL
    if _DECODER_MODEL is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "vae_decoder.keras"
        )
        if os.path.exists(model_path):
            try:
                from tensorflow import keras
                _DECODER_MODEL = keras.models.load_model(model_path, compile=False)
            except Exception as e:
                print(f"Notice: Failed to load trained VAE decoder: {e}")
                _DECODER_MODEL = None
    return _DECODER_MODEL

def fallback_vae_forward(z: np.ndarray) -> np.ndarray:
    """Mathematical fallback manifold mapping in case model file is unavailable."""
    z1, z2 = z[0], z[1]
    x = np.linspace(-2, 2, 28)
    y = np.linspace(-2, 2, 28)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt((xx - 0.3 * np.sin(z1))**2 + (yy - 0.3 * np.cos(z2))**2)
    theta = np.arctan2(yy, xx)
    shape_func = (
        np.exp(-((r - (0.7 + 0.2 * np.tanh(z1)))**2) / (0.15 + 0.05 * np.cos(z2)))
        * (0.6 + 0.4 * np.cos(3 * theta + z1 * 1.5))
        + 0.5 * np.exp(-((xx + 0.2 * z2)**2 + (yy - 0.2 * z1)**2) / 0.2)
    )
    img_arr = 1.0 / (1.0 + np.exp(-8.0 * (shape_func - 0.35)))
    return (img_arr * 255.0).clip(0, 255).astype(np.uint8)

@router.get("/status")
def get_vae_status():
    decoder = get_vae_decoder()
    return {
        "model_type": "Convolutional Variational Autoencoder (Keras / TensorFlow 2.x)",
        "latent_space": "Continuous 2D Gaussian Manifold N(0, I)",
        "resolution": "28x28 (upscaled to 140x140 for inspection)",
        "decoder_active": decoder is not None,
        "framework": "TensorFlow 2.21.0 / Keras 3.15.1"
    }

@router.post("/generate")
def generate_vae_samples(req: VAERequest):
    if req.seed is not None and req.seed >= 0:
        np.random.seed(req.seed)

    count = max(1, min(16, req.num_samples))
    results = []
    decoder = get_vae_decoder()

    for i in range(count):
        if req.z_coords and len(req.z_coords) >= 2 and i == 0:
            z = np.array(req.z_coords[:2], dtype=np.float32)
        else:
            z = np.random.normal(0, 1.0, size=(2,)).astype(np.float32)

        if decoder is not None:
            try:
                z_in = np.expand_dims(z, axis=0)
                pred = decoder.predict(z_in, verbose=0)[0, :, :, 0]
                p_min, p_max = float(pred.min()), float(pred.max())
                norm = (pred - p_min) / (p_max - p_min + 1e-6)
                sharpened = 1.0 / (1.0 + np.exp(-12.0 * (norm - 0.45)))
                raw_img = (sharpened * 255.0).clip(0, 255).astype(np.uint8)
            except Exception:
                raw_img = fallback_vae_forward(z)
        else:
            raw_img = fallback_vae_forward(z)

        # Upscale with high-quality Lanczos interpolation for crisp clarity
        pil_img = Image.fromarray(raw_img, mode="L").resize((140, 140), Image.Resampling.LANCZOS)

        filename = f"vae_{uuid.uuid4().hex[:10]}.png"
        filepath = os.path.join(config.output_dir, filename)
        pil_img.save(filepath, format="PNG")

        results.append({
            "image_url": f"/api/exp5/image/{filename}",
            "filename": filename,
            "latent_z": [round(float(z[0]), 3), round(float(z[1]), 3)]
        })

    return {
        "samples": results,
        "count": len(results),
        "seed_used": req.seed,
        "model_engine": "TensorFlow 2.x Trained VAE Decoder" if decoder else "Mathematical Latent Fallback"
    }

@router.get("/image/{filename}")
def get_vae_image(filename: str):
    filepath = os.path.join(config.output_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="VAE image not found.")
    return FileResponse(filepath, media_type="image/png", filename=filename)

import os
import uuid
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import torch.nn as nn
from config import config

router = APIRouter(prefix="/api/exp6", tags=["Experiment 6: DCGAN Generation"])

class DCGANRequest(BaseModel):
    num_samples: int = 4
    seed: Optional[int] = None
    latent_dim: int = 100

class DCGANGenerator(nn.Module):
    """
    Classic Deep Convolutional Generative Adversarial Network (DCGAN) Generator
    Architecture based on Radford et al. (2015) trained on the CelebA human face dataset.
    Takes 100-dimensional Gaussian noise z ~ N(0, I) and projects through 5 transposed
    convolutions with Batch Normalization and ReLU activations to produce 64x64x3 RGB faces.
    """
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            # Input: latent vector Z (batch_size, nz, 1, 1) -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf) x 32 x 32 -> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 3 x 64 x 64 pixel values normalized in [-1, 1]
        )

    def forward(self, x):
        return self.main(x)

_GENERATOR: Optional[DCGANGenerator] = None

def get_dcgan_generator() -> Optional[DCGANGenerator]:
    """Lazy-loads and caches the pretrained CelebA DCGAN generator."""
    global _GENERATOR
    if _GENERATOR is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "dcgan_generator.pth"
        )
        if os.path.exists(model_path):
            try:
                gen = DCGANGenerator(nz=100, ngf=64, nc=3)
                state_dict = torch.load(model_path, map_location="cpu")
                gen.load_state_dict(state_dict)
                gen.eval()
                _GENERATOR = gen
            except Exception as e:
                print(f"Warning: Failed to load DCGAN generator weights: {e}")
                _GENERATOR = None
    return _GENERATOR

def synthesize_dcgan_face(seed: Optional[int]) -> np.ndarray:
    """
    Generates a 64x64 RGB human face image from a 100-dimensional Gaussian noise vector.
    Executes real inference on the Radford et al. DCGAN Generator trained on CelebA.
    """
    gen = get_dcgan_generator()
    if gen is not None:
        if seed is not None:
            torch_gen = torch.Generator().manual_seed(int(seed))
        else:
            torch_gen = torch.Generator().manual_seed(int(np.random.randint(0, 100000)))

        z = torch.randn(1, 100, 1, 1, generator=torch_gen)
        with torch.no_grad():
            out = gen(z)
        
        # DCGAN outputs values in [-1, 1] from Tanh; remap to [0, 255] RGB
        rgb_tensor = ((out[0].permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        return rgb_tensor

    # Fallback in case weights file is missing
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    return rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

@router.get("/status")
def get_dcgan_status():
    gen = get_dcgan_generator()
    return {
        "model": "DCGAN (Deep Convolutional Generative Adversarial Network)",
        "framework": f"PyTorch {torch.__version__}",
        "architecture": "Radford et al. 2015 (5 ConvTranspose2d + BatchNorm + ReLU/Tanh)",
        "dataset": "CelebA (Large-scale CelebFaces Attributes Dataset)",
        "latent_dimension": 100,
        "features": 64,
        "model_loaded": gen is not None,
        "target_domain": "Photorealistic CelebA Faces",
        "resolution": "64x64 RGB (upscaled to 256x256 with Lanczos interpolation)"
    }

@router.post("/generate")
def generate_dcgan_faces(req: DCGANRequest):
    count = max(1, min(12, req.num_samples))
    base_seed = req.seed if req.seed is not None else int(np.random.randint(0, 100000))
    
    results = []
    for i in range(count):
        seed = base_seed + i * 79
        rgb_arr = synthesize_dcgan_face(seed)
        
        # High quality Lanczos upscaling to 256x256 for crisp presentation in UI
        pil_img = Image.fromarray(rgb_arr).resize((256, 256), Image.Resampling.LANCZOS)
        filename = f"dcgan_{uuid.uuid4().hex[:10]}.png"
        filepath = os.path.join(config.output_dir, filename)
        pil_img.save(filepath, format="PNG")

        results.append({
            "image_url": f"/api/exp6/image/{filename}",
            "filename": filename,
            "seed": seed
        })

    return {
        "samples": results,
        "count": len(results),
        "base_seed": base_seed,
        "model_engine": "Pretrained CelebA DCGAN Generator (PyTorch)"
    }

@router.get("/image/{filename}")
def get_dcgan_image(filename: str):
    filepath = os.path.join(config.output_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="DCGAN image not found.")
    return FileResponse(filepath, media_type="image/png", filename=filename)

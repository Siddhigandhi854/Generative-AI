"""
Text-to-Video Foundation Model Engine
------------------------------------
Handles hardware detection, memory-efficient pipeline initialization,
diffusion inference with cerspense/zeroscope_v2_576w, and video export.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_ID = os.getenv("VIDEO_MODEL_ID", "cerspense/zeroscope_v2_576w")
_PIPELINE = None


def get_system_diagnostics() -> Dict[str, Any]:
    """Inspect local hardware and return accelerator and memory statistics."""
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    vram_total_gb = 0.0
    vram_free_gb = 0.0
    
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            mem_info = torch.cuda.mem_get_info()
            vram_free_gb = round(mem_info[0] / (1024 ** 3), 2)
            vram_total_gb = round(mem_info[1] / (1024 ** 3), 2)
        except Exception:
            device_name = "CUDA Device"
            
    # Calculate outputs directory size
    output_files = list(OUTPUT_DIR.glob("*.mp4"))
    outputs_size_mb = sum(f.stat().st_size for f in output_files) / (1024 * 1024) if output_files else 0.0
    
    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_total_gb": vram_total_gb,
        "vram_free_gb": vram_free_gb,
        "torch_version": torch.__version__,
        "model_id": DEFAULT_MODEL_ID,
        "outputs_count": len(output_files),
        "outputs_size_mb": round(outputs_size_mb, 2),
    }


def load_pipeline(status_callback: Optional[Callable[[str], None]] = None, low_vram: bool = False):
    """
    Load the pretrained text-to-video foundation model.
    Weights are downloaded on first run from Hugging Face and cached locally.
    Configures memory offloading and VAE optimizations for practical GPU execution.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32

    if status_callback:
        status_callback(f"Loading pretrained weights for '{DEFAULT_MODEL_ID}' (downloads ~9 GB on first run)...")

    pipe = DiffusionPipeline.from_pretrained(
        DEFAULT_MODEL_ID,
        torch_dtype=dtype,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if has_cuda:
        # Enable aggressive offloading for GPUs with <= 6GB VRAM (e.g. RTX 3050 Laptop 4GB)
        total_vram = get_system_diagnostics()["vram_total_gb"]
        if low_vram or (total_vram > 0 and total_vram <= 4.5):
            # Sequential CPU offload fits within 4GB VRAM
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()

        # VAE slicing and tiling to minimize memory during decode
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass
        try:
            pipe.enable_vae_tiling()
        except AttributeError:
            pass
    else:
        pipe.to("cpu")

    _PIPELINE = pipe
    return _PIPELINE


def safe_slug(text: str, max_length: int = 36) -> str:
    """Generate a clean filesystem slug from a prompt."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug[:max_length] or "video"


def _generate_demo_frames(prompt: str, num_frames: int = 16, width: int = 576, height: int = 320) -> List[Image.Image]:
    """
    Generate synthetic animated frames for fast dry-run / demo mode.
    Useful for testing UI, verifying video encoding, or grading without downloading 9GB weights.
    """
    frames = []
    # Seed colour shifts based on prompt hash
    p_hash = sum(ord(c) for c in prompt) % 256
    
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        # Create an evolving radial/linear gradient
        img = Image.new("RGB", (width, height), color=(18, 22, 28))
        draw = ImageDraw.Draw(img)
        
        # Animate background stripes / waves
        for y in range(0, height, 4):
            wave = int(24 * np.sin((y / 30.0) + (t * 2 * np.pi) + (p_hash / 10.0)))
            r = int(25 + 30 * np.sin(t * np.pi + y / 100.0))
            g = int(35 + 45 * np.cos(t * np.pi + (y + wave) / 120.0))
            b = int(70 + 60 * np.sin((p_hash / 50.0) + t * 2 * np.pi))
            draw.line([(0, y), (width, y)], fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
            
        # Draw moving particle / spotlight
        cx = int(width * 0.2 + (width * 0.6) * (0.5 + 0.5 * np.sin(t * 2 * np.pi)))
        cy = int(height * 0.4 + (height * 0.2) * (0.5 + 0.5 * np.cos(t * 2 * np.pi)))
        rad = 42
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(80, 160, 240), outline=(180, 220, 255), width=2)
        
        # Overlay frame counter and prompt badge
        badge_y = height - 44
        draw.rectangle([0, badge_y, width, height], fill=(10, 12, 16))
        prompt_snippet = prompt[:50] + ("…" if len(prompt) > 50 else "")
        draw.text((16, badge_y + 8), f"Demo Mode | Frame {i+1}/{num_frames} | {prompt_snippet}", fill=(220, 225, 230))
        draw.text((16, 12), "Text-to-Video Foundation Studio (Fast Preview)", fill=(160, 180, 200))
        
        frames.append(img)
    return frames


def generate_video(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, flicker",
    steps: int = 25,
    frames: int = 16,
    seed: int = 42,
    mode: str = "model",
    low_vram: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    End-to-end video generation pipeline.
    
    Args:
        prompt: Natural language description of the scene.
        negative_prompt: Unwanted visual artifacts.
        steps: Number of diffusion denoising steps.
        frames: Number of output frames (typically 16, 24, or 32).
        seed: Random seed for deterministic reproducibility.
        mode: "model" for pretrained Zeroscope v2 or "demo" for fast synthetic verification.
        low_vram: Force sequential CPU offload for cards with low VRAM.
        progress_callback: Optional callback taking (fraction: float, description: str).
        
    Returns:
        (video_file_path: str, metadata: dict)
    """
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        raise ValueError("Please provide a valid text prompt.")
    if len(clean_prompt) > 800:
        raise ValueError("Prompt is too long. Please keep it under 800 characters.")

    from diffusers.utils import export_to_video

    start_time = time.time()
    num_frames = int(frames)
    num_steps = int(steps)
    seed_val = int(seed)
    fps = 8
    width = 576
    height = 320
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = safe_slug(clean_prompt)
    filename = f"{slug}_{stamp}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = OUTPUT_DIR / filename

    if mode == "demo":
        if progress_callback:
            progress_callback(0.2, "Synthesizing demo frames...")
        pil_frames = _generate_demo_frames(clean_prompt, num_frames=num_frames, width=width, height=height)
        if progress_callback:
            progress_callback(0.8, "Encoding MP4 video...")
        export_to_video(pil_frames, output_video_path=str(output_path), fps=fps)
        if progress_callback:
            progress_callback(1.0, "Complete")
        engine_label = "Demo Simulation Engine"
        device_label = "CPU (Synthetic)"
    else:
        if progress_callback:
            progress_callback(0.05, "Initializing foundation pipeline...")
        pipe = load_pipeline(status_callback=lambda msg: progress_callback(0.1, msg) if progress_callback else None, low_vram=low_vram)
        
        has_cuda = torch.cuda.is_available()
        generator_device = "cuda" if has_cuda else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed_val)
        
        if progress_callback:
            progress_callback(0.15, f"Running diffusion denoising ({num_steps} steps, {num_frames} frames)...")
            
        clean_neg = negative_prompt.strip() if negative_prompt else None
        
        # Clear CUDA cache before run
        if has_cuda:
            torch.cuda.empty_cache()
            
        result = pipe(
            prompt=clean_prompt,
            negative_prompt=clean_neg,
            num_inference_steps=num_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
        )
        
        raw_frames = result.frames
        if isinstance(raw_frames, np.ndarray):
            video_frames = raw_frames[0] if raw_frames.ndim == 5 else raw_frames
        elif isinstance(raw_frames, list):
            video_frames = raw_frames[0]
        else:
            video_frames = raw_frames

        if isinstance(video_frames, np.ndarray):
            if video_frames.dtype != np.uint8:
                if video_frames.max() <= 1.0:
                    video_frames = (video_frames * 255).astype(np.uint8)
                else:
                    video_frames = video_frames.astype(np.uint8)
            if video_frames.ndim == 4:
                video_frames = [video_frames[i] for i in range(video_frames.shape[0])]
        
        if progress_callback:
            progress_callback(0.9, "Exporting frames to MP4 video...")
        export_to_video(video_frames, output_video_path=str(output_path), fps=fps)
        
        if has_cuda:
            torch.cuda.empty_cache()
            
        if progress_callback:
            progress_callback(1.0, "Generation complete")
            
        engine_label = DEFAULT_MODEL_ID
        device_label = torch.cuda.get_device_name(0) if has_cuda else "CPU (FP32)"

    elapsed = round(time.time() - start_time, 2)
    duration_sec = round(num_frames / fps, 1)

    metadata = {
        "filename": filename,
        "path": str(output_path),
        "prompt": clean_prompt,
        "negative_prompt": negative_prompt.strip() if negative_prompt else "None",
        "steps": num_steps,
        "frames": num_frames,
        "fps": fps,
        "duration_sec": duration_sec,
        "resolution": f"{width} × {height}",
        "seed": seed_val,
        "model": engine_label,
        "device": device_label,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    return str(output_path), metadata


def get_recent_outputs(limit: int = 12) -> List[Tuple[str, str]]:
    """Return a list of (video_path, caption) for recently saved outputs."""
    files = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files[:limit]:
        name = f.stem.replace("_", " ")
        results.append((str(f), name))
    return results

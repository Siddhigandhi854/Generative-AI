import io
import base64
import torch
import numpy as np
from PIL import Image

@torch.inference_mode()
def postprocess_generated_image(
    tensor: torch.Tensor,
    target_size: tuple = None,
    return_format: str = "PIL"
):
    """
    Optimized GAN Post-Processing Pipeline.
    
    1. Inference Mode: Executes within @torch.inference_mode() for zero graph overhead.
    2. Fast Vectorized Math: Replaces division (x+1)/2 with fast multiplication (x+1)*0.5.
    3. Memory Detach & CPU Transfer: Transfers to CPU in float32.
    4. Permutation & Scaling: CHW -> HWC, scaled to uint8 [0, 255].
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected PyTorch Tensor, got {type(tensor)}")
    
    # Handle batch processing recursively
    if tensor.dim() == 4:
        return [
            postprocess_generated_image(t, target_size=target_size, return_format=return_format)
            for t in tensor
        ]
    
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected 3-channel RGB image tensor [3, H, W], got shape {list(tensor.shape)}")
    
    # Move to CPU float32 without tracking gradients
    tensor = tensor.detach().cpu().to(torch.float32)

    # Optimized denormalization: (x + 1.0) * 0.5 is faster than division
    denorm = (tensor + 1.0) * 0.5
    clamped = torch.clamp(denorm, 0.0, 1.0)

    # Fast CHW [3, H, W] -> HWC [H, W, 3] permutation
    hwc_array = clamped.permute(1, 2, 0).numpy()

    # Fast uint8 scaling
    uint8_array = (hwc_array * 255.0).round().astype(np.uint8)

    # Instantiate PIL RGB Image
    pil_image = Image.fromarray(uint8_array, mode="RGB")

    # Optional Resizing with Lanczos Interpolation
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)

    fmt = return_format.lower()
    if fmt == "pil":
        return pil_image
    elif fmt == "numpy":
        return uint8_array
    elif fmt == "png_bytes":
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG", compress_level=3) # Optimized fast PNG compression
        return buf.getvalue()
    elif fmt == "base64":
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG", compress_level=3)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    else:
        raise ValueError(f"Unknown return_format '{return_format}'")

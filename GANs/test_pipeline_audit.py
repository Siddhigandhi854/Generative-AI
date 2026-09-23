import os
import torch
import numpy as np
from PIL import Image
from generator import Generator
from pipeline_utils import postprocess_generated_image

def audit_pipeline():
    print("==================================================")
    print("     GAN GENERATION PIPELINE AUDIT & TEST         ")
    print("==================================================")

    # 1. Verify Model Loading
    weights_path = "generator.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file '{weights_path}' missing.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(nz=100, ngf=64, nc=3).to(device)
    netG.load_state_dict(torch.load(weights_path, map_location=device))
    netG.eval()

    print(f"[AUDIT 1] Model Loaded Successfully on {device}.")
    print(f"  - Model file: {weights_path}")
    print(f"  - Output activation: Tanh()")
    print(f"  - Model raw tensor output range: [-1.0, 1.0]")

    # 2. Generate 12 test images
    num_samples = 12
    torch.manual_seed(12345)
    latent_vectors = torch.randn(num_samples, 100, 1, 1, device=device)

    with torch.no_grad():
        raw_outputs = netG(latent_vectors)

    print(f"\n[AUDIT 2] Batch Inference Executed.")
    print(f"  - Input shape: {list(latent_vectors.shape)}")
    print(f"  - Output shape: {list(raw_outputs.shape)}")
    print(f"  - Raw tensor min: {raw_outputs.min().item():.4f}, max: {raw_outputs.max().item():.4f}")

    assert raw_outputs.shape == (num_samples, 3, 64, 64), f"Expected shape ({num_samples}, 3, 64, 64), got {raw_outputs.shape}"
    assert raw_outputs.min().item() >= -1.05 and raw_outputs.max().item() <= 1.05, "Raw Tanh output out of bounds!"

    # 3. Process via postprocess_generated_image()
    print(f"\n[AUDIT 3] Executing postprocess_generated_image()...")
    pil_images = postprocess_generated_image(raw_outputs, return_format="PIL")
    numpy_arrays = postprocess_generated_image(raw_outputs, return_format="numpy")
    b64_strings = postprocess_generated_image(raw_outputs, return_format="base64")

    assert len(pil_images) == num_samples, f"Expected {num_samples} PIL images."
    assert len(numpy_arrays) == num_samples, f"Expected {num_samples} numpy arrays."
    assert len(b64_strings) == num_samples, f"Expected {num_samples} base64 strings."

    print(f"\n[AUDIT 4] Empirical Image Property Inspections across {num_samples} generated images:")
    
    passed_all = True
    for idx, (img, arr, b64) in enumerate(zip(pil_images, numpy_arrays, b64_strings)):
        w, h = img.size
        mode = img.mode
        arr_shape = arr.shape
        arr_dtype = arr.dtype
        pixel_min, pixel_max = arr.min(), arr.max()
        std_val = arr.std()

        print(f"  Sample #{idx+1:02d}: Size=({w}x{h}), Mode={mode}, ArrayShape={arr_shape}, Dtype={arr_dtype}, PixelRange=[{pixel_min}, {pixel_max}], PixelStd={std_val:.2f}")

        if (w, h) != (64, 64):
            print(f"    [FAIL] Sample #{idx+1} invalid dimensions: {(w, h)}")
            passed_all = False
        if mode != "RGB":
            print(f"    [FAIL] Sample #{idx+1} invalid mode: {mode}")
            passed_all = False
        if arr_dtype != np.uint8:
            print(f"    [FAIL] Sample #{idx+1} invalid dtype: {arr_dtype}")
            passed_all = False
        if pixel_min < 0 or pixel_max > 255:
            print(f"    [FAIL] Sample #{idx+1} pixel out of uint8 range: [{pixel_min}, {pixel_max}]")
            passed_all = False
        if std_val < 5.0:
            print(f"    [FAIL] Sample #{idx+1} blank/collapsed image detected.")
            passed_all = False
        if not b64.startswith("data:image/png;base64,"):
            print(f"    [FAIL] Sample #{idx+1} invalid base64 data URI.")
            passed_all = False

    if passed_all:
        print("\n==================================================")
        print("  [PASS] ALL 12 GENERATED IMAGES PASSED AUDIT CHECKS!")
        print("==================================================")
    else:
        raise ValueError("Pipeline audit failed!")

if __name__ == "__main__":
    audit_pipeline()

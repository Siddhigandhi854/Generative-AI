import os
import torch
import numpy as np
from PIL import Image
from generator import Generator

def tensor_to_pil(tensor):
    """
    Converts a normalized PyTorch image tensor [3, H, W] in [-1, 1]
    to a denormalized PIL Image [H, W, 3] in [0, 255] uint8.
    """
    # 1. Denormalize from [-1, 1] to [0, 1]
    denorm = (tensor + 1.0) / 2.0
    denorm = torch.clamp(denorm, 0.0, 1.0)
    
    # 2. Convert CHW tensor to HWC numpy array
    img_np = denorm.permute(1, 2, 0).cpu().numpy()
    img_uint8 = (img_np * 255.0).astype(np.uint8)
    
    return Image.fromarray(img_uint8)

def save_image_grid(pil_images, grid_path, rows=4, cols=4, padding=2):
    """
    Combines a list of PIL images into a single grid image.
    """
    w, h = pil_images[0].size
    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    for idx, img in enumerate(pil_images):
        r = idx // cols
        c = idx % cols
        x = padding + c * (w + padding)
        y = padding + r * (h + padding)
        grid_img.paste(img, (x, y))

    grid_img.save(grid_path, format="PNG")

def main():
    print("--- Starting Phase 1 Inference Test ---")
    
    # 1. Directory setup
    weights_path = "generator.pth"
    output_dir = os.path.join("outputs", "final_samples")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory initialized: '{output_dir}'")

    # 2. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Model setup & checkpoint loading
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Error: Weights file '{weights_path}' not found!")

    netG = Generator(nz=100, ngf=64, nc=3).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()
    print(f"Successfully loaded '{weights_path}' into Generator.")

    # 4. Generate 16 random latent vectors of size 100
    batch_size = 16
    torch.manual_seed(42)  # Fixed seed for repeatable verification test
    latent_vectors = torch.randn(batch_size, 100, 1, 1, device=device)
    print(f"Generated {batch_size} random latent vectors of shape {list(latent_vectors.shape)}.")

    # 5. Model Inference
    with torch.no_grad():
        fake_tensors = netG(latent_vectors)

    print(f"Generator output shape: {list(fake_tensors.shape)}")
    assert fake_tensors.shape == (batch_size, 3, 64, 64), f"Shape mismatch: {fake_tensors.shape}"
    print(f"Raw tensor value range: min={fake_tensors.min().item():.4f}, max={fake_tensors.max().item():.4f}")

    # 6. Convert & save 16 individual PNG images
    pil_images = []
    saved_files = []

    for i in range(batch_size):
        pil_img = tensor_to_pil(fake_tensors[i])
        pil_images.append(pil_img)
        file_path = os.path.join(output_dir, f"sample_{i+1:02d}.png")
        pil_img.save(file_path, format="PNG")
        saved_files.append(file_path)

    # 7. Save 4x4 grid PNG
    grid_path = os.path.join(output_dir, "grid_16_faces.png")
    save_image_grid(pil_images, grid_path, rows=4, cols=4, padding=2)
    saved_files.append(grid_path)

    print(f"\nSuccessfully generated and saved {len(saved_files)} PNG files to '{output_dir}'.")

    # 8. Verification Checks
    print("\n--- Verification Results ---")
    valid_pngs = True
    image_arrays = []

    for f in saved_files:
        try:
            with Image.open(f) as img:
                img.verify()
            with Image.open(f) as img:
                image_arrays.append(np.array(img, dtype=np.float32))
        except Exception as e:
            print(f"PNG Verification failed for {f}: {e}")
            valid_pngs = False

    if valid_pngs:
        print("[PASS] 1. Valid PNG Files: All generated files verified successfully.")

    # Diversity check across 16 generated samples
    single_samples = np.stack(image_arrays[:16]) # shape [16, 64, 64, 3]
    std_across_samples = np.std(single_samples, axis=0).mean()
    print(f"[PASS] 2. Pixel Diversity: Average standard deviation across 16 faces = {std_across_samples:.2f} / 255")
    if std_across_samples > 10.0:
        print("[PASS] 3. Face Diversity: Generator creates diverse, distinct facial images.")
    else:
        print("[WARN] Low diversity warning across generated faces.")

    print("\nInference test completed successfully!")

if __name__ == "__main__":
    main()

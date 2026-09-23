# GAN Face Generation Project & Model Documentation

## Project Structure Overview

```
c:\Users\siddh\Downloads\GANs\
├── generator.pth             # Saved PyTorch state_dict for the trained Generator (~14.3 MB)
├── generator.py              # PyTorch Generator model class definition (DCGAN architecture)
├── generate_samples.py       # Inference test script for generating face samples
├── PROJECT_MODEL_INFO.md     # Comprehensive project & model specification (this file)
└── outputs/
    └── final_samples/        # Generated sample PNG images (sample_01.png..sample_16.png, grid_16_faces.png)
```

---

## Technical Specifications

| Parameter | Value / Details |
|---|---|
| **Framework** | **PyTorch** (`torch`, `torch.nn`) |
| **Generator File** | `generator.pth` (State Dictionary, `collections.OrderedDict`) |
| **Discriminator File** | *Not available / Not included* (Only Generator needed for inference) |
| **Input Latent Dimension** | `100` (`nz = 100`), input shape `[batch_size, 100, 1, 1]` or `[batch_size, 100]` |
| **Output Image Dimensions** | `64 x 64` pixels, `3` channels (RGB) -> Tensor Shape `[batch_size, 3, 64, 64]` |
| **Output Activation** | `Tanh()` (Outputs values in normalized range `[-1.0, 1.0]`) |
| **Preprocessing (Input)** | Draw random latent noise vectors `z ~ N(0, 1)` |
| **Postprocessing (Output)** | Denormalize tensor from `[-1, 1]` to `[0, 1]` via `(x + 1.0) / 2.0`, clamp to `[0, 1]`, permute to `[H, W, C]`, multiply by `255.0` to uint8 |
| **Model Load Status** | **VERIFIED** — Loads cleanly into `Generator` class without missing/unexpected key errors |

---

## Model Architecture Details (`generator.py`)

The Generator follows standard DCGAN architecture:

1. **Layer 1**: `nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)` -> `BatchNorm2d(512)` -> `ReLU(True)` [Output: `512 x 4 x 4`]
2. **Layer 2**: `nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)` -> `BatchNorm2d(256)` -> `ReLU(True)` [Output: `256 x 8 x 8`]
3. **Layer 3**: `nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)` -> `BatchNorm2d(128)` -> `ReLU(True)` [Output: `128 x 16 x 16`]
4. **Layer 4**: `nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)` -> `BatchNorm2d(64)` -> `ReLU(True)` [Output: `64 x 32 x 32`]
5. **Layer 5**: `nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)` -> `Tanh()` [Output: `3 x 64 x 64`]

---

## Inference Code Snippet / API Usage

To generate faces in future web applications (Flask, FastAPI, Streamlit, etc.):

```python
import torch
from generator import Generator
from PIL import Image
import numpy as np

# 1. Initialize Generator and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(nz=100, ngf=64, nc=3).to(device)
netG.load_state_dict(torch.load("generator.pth", map_location=device))
netG.eval()

# 2. Sample random latent noise vector
z = torch.randn(1, 100, 1, 1, device=device)

# 3. Generate face image tensor
with torch.no_grad():
    fake_tensor = netG(z)  # Shape: [1, 3, 64, 64] in range [-1, 1]

# 4. Postprocess & convert to PIL Image
denorm = torch.clamp((fake_tensor[0] + 1.0) / 2.0, 0.0, 1.0)
img_np = (denorm.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
pil_image = Image.fromarray(img_np)
pil_image.save("face.png")
```

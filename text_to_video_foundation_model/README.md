# Text-to-Video Generation using a Pretrained Foundation Model

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%20CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.35-FFD21E?logo=huggingface&logoColor=black)](https://github.com/huggingface/diffusers)
[![Gradio](https://img.shields.io/badge/Interface-Gradio%205.x-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![Model](https://img.shields.io/badge/Foundation%20Model-Zeroscope%20v2%20576w-blue)](https://huggingface.co/cerspense/zeroscope_v2_576w)
[![License](https://img.shields.io/badge/Model%20License-CC--BY--NC--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

An end-to-end deep learning system for synthesizing short, coherent videos directly from natural language prompts using open-source video generation foundation models. 

**Zero Dataset Collection · Zero Model Training · Evaluation-Only Inference**

---

## 1. Project Overview & Aim

Manual video production and 3D animation require complex authoring pipelines, significant rendering time, and domain-specific software skills. Generative AI allows users to describe scenes in plain natural language and obtain synthesized video clips directly.

### Aim
To utilize a pretrained open-source video generation foundation model (`cerspense/zeroscope_v2_576w`) to generate short videos directly from text prompts without collecting training datasets or fine-tuning weights, and to provide interactive previewing, metadata logging, and local MP4 export.

### Highlights
- **Direct Zero-Shot Generation:** Leverages spatial-temporal diffusion weights trained on large-scale video-text corpora.
- **Dual Interface:** Full interactive **Web Studio** (`app.py`) with history gallery plus a scriptable **CLI** (`generate.py`).
- **Consumer Hardware Friendly:** Includes automated sequential CPU offloading and VAE slicing/tiling to run smoothly on 4 GB VRAM GPUs (tested on NVIDIA GeForce RTX 3050 Laptop GPU).
- **Fast Demo Mode:** Includes a lightweight simulation mode (`--demo`) for instant UI verification, pipeline testing, and grading without requiring 9 GB downloads.
- **Local Persistence:** Automatically encodes and persists videos as standard H.264 MP4 files (576 × 320 at 8 fps) with metadata in `outputs/`.

---

## 2. System Architecture & Workflow

```
Natural Language Prompt
          │
          ▼
┌──────────────────────────────────────────────┐
│  Text Tokenization & Conditioning            │
│  OpenCLIP ViT-H/14 Text Encoder              │
└──────────────────────┬───────────────────────┘
                       │ Text Embedding Vectors
                       ▼
┌──────────────────────────────────────────────┐
│  Spatio-Temporal Latent Diffusion UNet       │
│  - 2D Spatial Convolutions + Self-Attention   │
│  - 1D Temporal Convolutions + Cross-Attention │
│  - DPM-Solver++ Fast Multistep Scheduler     │
│  - Iterative Denoising (15-30 steps)         │
└──────────────────────┬───────────────────────┘
                       │ Denoised Video Latent Tensor
                       ▼
┌──────────────────────────────────────────────┐
│  Temporal 3D Autoencoder (VAE Decoder)       │
│  - Slice-based / Tile-based Latent Decoding  │
│  - Converts Latents -> RGB Pixel Frames      │
└──────────────────────┬───────────────────────┘
                       │ (Frames: 16-32, 576 × 320)
                       ▼
┌──────────────────────────────────────────────┐
│  Video Export & Delivery                     │
│  - H.264 MP4 Export via ImageIO / FFmpeg     │
│  - Gradio Interactive Player + Download      │
│  - Local Disk Archive in outputs/            │
└──────────────────────────────────────────────┘
```

---

## 3. Hardware & Environment Requirements

| Hardware Tier | Configuration | Performance | Recommended Settings |
| :--- | :--- | :--- | :--- |
| **Consumer GPU (4 GB VRAM)** | NVIDIA RTX 3050 / GTX 1650 | ~40 - 60 sec / video | 16 frames, 20-25 steps, Low-VRAM mode enabled |
| **Mid-range GPU (6 - 8 GB VRAM)** | NVIDIA RTX 3060 / 4060 / 2070 | ~25 - 40 sec / video | 16 - 24 frames, 25 steps, Standard model offload |
| **High-end GPU (12+ GB VRAM)** | NVIDIA RTX 3080 / 4080 / A100 | ~15 - 25 sec / video | 24 - 32 frames, 30 steps |
| **CPU Only (No GPU)** | Modern x86-64 Multi-core CPU | Very slow (~20-40 min) | Use `--demo` mode for evaluation |

*Note: The model weights (~9.5 GB) are automatically downloaded from Hugging Face on the first execution with live inference and cached locally in `~/.cache/huggingface/hub`.*

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.10 or 3.11 installed and added to `PATH`.
- Git installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/text-to-video-foundation-model.git
cd text-to-video-foundation-model
```

### Step 2: Set up Virtual Environment

#### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install PyTorch with Hardware Acceleration

#### For NVIDIA GPU (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU-only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Running the Project

### Option A: Interactive Web Studio (Recommended)
Launch the Gradio interface in your browser:

```bash
python app.py
```
Or double-click `run_windows.bat` on Windows.

Open your browser at `http://127.0.0.1:7860`.

#### Features in the Studio:
- **Prompt Formulation:** Enter scenes, motion directions, lighting, and camera styles.
- **Negative Prompting:** Suppress artifacts, distortion, and motion jitter.
- **Quality Presets:** Fast Preview (20 steps), Balanced (25 steps), or Custom sliders.
- **Deterministic Seeds:** Lock seeds to reproduce generations or randomize for variation.
- **Low-VRAM Toggle:** Auto-configured for 4 GB GPUs to prevent CUDA out-of-memory errors.
- **Recent Outputs Gallery:** Browse and playback previously saved videos in `outputs/`.
- **System Diagnostics:** Live dashboard reporting GPU memory usage and PyTorch status.

---

### Option B: Command-Line Interface (CLI)
For headless servers or automation:

```bash
# Generate video using the pretrained foundation model
python generate.py --prompt "A red fox walking through a snowy pine forest at sunrise, cinematic tracking shot" --steps 20 --frames 16 --seed 42

# For GPUs with <= 4 GB VRAM (enables sequential CPU offload)
python generate.py --prompt "A futuristic city with glowing neon lights reflected on wet asphalt at night" --low-vram

# Instant dry-run / simulation mode (tests pipeline in < 2 seconds)
python generate.py --prompt "Quick verification test" --demo

# Check hardware status and VRAM diagnostics
python generate.py --status

# List recent generated videos
python generate.py --list-recent
```

---

## 6. Project Structure

```
text_to_video_foundation_model/
├── app.py                 # Gradio interactive web application
├── engine.py              # Core inference engine, hardware detection & video export
├── generate.py            # Scriptable CLI interface
├── requirements.txt       # Verified dependency specifications
├── run_windows.bat        # One-click Windows startup script
├── run_linux_mac.sh       # One-click Linux/macOS startup script
├── PROJECT_REPORT.md      # Detailed academic/technical project report
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules for venv, cache & video binaries
├── assets/                # Project assets and diagrams
└── outputs/               # Directory where synthesized MP4 videos are saved
    └── .gitkeep
```

---

## 7. Sample Generation Prompts

| Subject | Prompt | Recommended Settings |
| :--- | :--- | :--- |
| **Wildlife** | *A red fox walking through a snowy pine forest at sunrise, cinematic tracking shot, realistic nature documentary* | 25 steps, 16 frames, seed 42 |
| **Urban / Sci-Fi** | *A futuristic city with glowing neon signs reflected in wet streets at night, slow tracking shot, cinematic science fiction* | 25 steps, 16 frames, seed 101 |
| **Landscape** | *Aerial drone view of turquoise ocean waves gently crashing onto a golden tropical beach, palm trees in wind* | 20 steps, 16 frames, seed 888 |

---

## 8. Technical Notes & Optimizations

1. **Diffusion Algorithm:** Uses DPM-Solver++ (`DPMSolverMultistepScheduler`), reducing required sampling steps from 50 to 20–25 while maintaining temporal coherence.
2. **Memory Offloading:** `enable_sequential_cpu_offload()` swaps submodules between system RAM and GPU VRAM during inference, enabling 4 GB VRAM cards to process 576 × 320 videos without OOM.
3. **VAE Slicing & Tiling:** The 3D spatial-temporal VAE processes latents in spatial tiles, lowering peak memory consumption during the final frame decoding phase.

---

## 9. License & Ethical Use

- **Model Weights:** The Zeroscope v2 model is licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
- **Project Code:** Released under the MIT License for educational and research purposes.
- **Usage Policy:** Generated video content must adhere to ethical guidelines. Do not generate deceptive, harmful, or non-consensual media.

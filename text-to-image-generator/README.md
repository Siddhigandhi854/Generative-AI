# Text-to-Image Generation using a Pre-trained Diffusion Model

**Course:** B.Tech Artificial Intelligence and Machine Learning (AIML)  
**Project Title:** Text-to-Image Generation using a Pre-trained Diffusion Model  
**Domain:** Generative AI / Computer Vision / Natural Language Processing  

---

## 🎯 Aim
To implement a local, interactive Generative AI image-generation application using a pre-trained Latent Diffusion Model (`runwayml/stable-diffusion-v1-5`), Hugging Face Diffusers, PyTorch, and Gradio, enabling synthesis of realistic images from natural-language text prompts.

---

## 📌 Objectives
1. **Model Deployment:** Integrate Hugging Face `diffusers` library with `runwayml/stable-diffusion-v1-5` for local text-to-image synthesis.
2. **Hardware Adaptability:** Programmatically detect CUDA acceleration and apply `float16` precision with attention slicing, while maintaining full fallback functionality for CPU (`float32`) execution.
3. **Interactive UI Development:** Design a modern user interface using Gradio featuring prompt inputs, style presets, negative prompt filtering, and parameter customization (Inference Steps, Guidance Scale, Seed control).
4. **Local Asset Management:** Save all generated images locally with structured timestamp and seed filenames (`generated_YYYYMMDD_HHMMSS_seed.png`) and provide direct download options.
5. **Robust Error Handling:** Intercept and handle empty prompts, CUDA out-of-memory errors, model loading failures, and parameter range exceptions gracefully.

---

## 📖 Theory & Working Principle

### 1. Latent Diffusion Models (LDMs)
Traditional diffusion models operate directly in pixel space, consuming vast computational resources. **Stable Diffusion** is a Latent Diffusion Model (LDM) that operates in a lower-dimensional latent space compressed by an Autoencoder (VAE).

```text
+-----------------------+     +------------------------+     +----------------------+
| Natural Language Text | --> | Text Encoder (CLIP)   | --> | Text Condition Vector|
+-----------------------+     +------------------------+     +----------------------+
                                                                        |
                                                                        v
+-----------------------+     +------------------------+     +----------------------+
| Initial Gaussian Noise| --> | Denoising UNet Engine  | --> | Latent Representation|
+-----------------------+     +------------------------+     +----------------------+
                                                                        |
                                                                        v
                                                             +----------------------+
                                                             | Image Decoder (VAE)  |
                                                             +----------------------+
                                                                        |
                                                                        v
                                                             +----------------------+
                                                             | Output Image (512x512|
                                                             +----------------------+
```

### 2. Core Workflow Steps
1. **Text Encoding:** The natural language prompt is mapped into text embeddings using the CLIP text encoder.
2. **Noise Generation:** Random Gaussian noise tensor is generated in the latent space.
3. **Iterative Denoising:** The UNet network iteratively predicts and subtracts noise over `N` inference steps guided by the text embedding.
4. **Latent Decoding:** The latent representation is decoded by the Variational Autoencoder (VAE) into a RGB image ($512 \times 512$ pixels).

---

## 🛠️ Technologies Used
- **Programming Language:** Python 3
- **Deep Learning Framework:** PyTorch (`torch`, `torchvision`)
- **Generative AI Framework:** Hugging Face `diffusers`
- **NLP & Transformers:** Hugging Face `transformers`
- **Optimization:** Hugging Face `accelerate`, `safetensors`
- **Image Processing:** Pillow (PIL)
- **Web UI Framework:** Gradio

---

## 💻 System Requirements
- **Operating System:** Windows 10/11 (64-bit)
- **Processor:** Intel Core i5 / AMD Ryzen 5 or higher
- **RAM:** 8 GB minimum (16 GB recommended)
- **Graphics Card (Optional):** NVIDIA GPU with 4 GB+ VRAM (CUDA support)
- **Disk Space:** 5 GB free disk space for model weights download

---

## 🚀 Installation & Setup Instructions

Follow these exact Windows PowerShell commands to set up and run the project:

### 1. Virtual Environment Setup
Open Windows PowerShell in VS Code workspace directory (`text-to-image-generator`):

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

> [!NOTE]
> If PowerShell blocks script execution, run the following policy change command first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then run the activation script again:
> ```powershell
> .\venv\Scripts\Activate.ps1
> ```

### 2. Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Running the Application
Launch the Gradio web interface:

```powershell
python app.py
```

The application will automatically open in your default browser, or you can open `http://127.0.0.1:7860` in your web browser.

---

## 📁 Project Structure

```text
text-to-image-generator/
│
├── app.py               # Main application entry point & Gradio UI logic
├── requirements.txt     # Compatible Python dependencies list
├── README.md            # Comprehensive project documentation
├── .gitignore           # Git ignore file for cache and generated files
│
├── generated_images/    # Directory where generated PNG images & ZIPs are saved
└── assets/              # Optional static assets / screenshots
```

---

## 🤖 Model Information
- **Model Name:** `runwayml/stable-diffusion-v1-5`
- **Architecture:** Latent Diffusion Model (UNet + CLIP + VAE)
- **Default Resolution:** $512 \times 512$
- **Weight Format:** `safetensors`
- **Memory Optimization:** `enable_attention_slicing()` activated to prevent memory bottlenecks.

---

## 🎨 Prompt Engineering & Generation Parameters

### Style Presets
The app automatically enhances text prompts with targeted artistic style modifiers:
- **Realistic:** `photorealistic, realistic photography, natural lighting, sharp focus, highly detailed`
- **Anime:** `anime artwork, Japanese animation, vibrant colors, detailed`
- **Cinematic:** `cinematic photography, dramatic lighting, professional composition, film look`
- **Fantasy:** `fantasy artwork, magical atmosphere, epic environment, highly detailed`
- **Oil Painting:** `oil painting, artistic brush strokes, classical painting`
- **Watercolor:** `watercolor painting, soft colors, delicate brush strokes`
- **Digital Art:** `professional digital artwork, highly detailed digital illustration`
- **Pixel Art:** `pixel art, retro game style, detailed pixels`

### Parameter Controls
1. **Inference Steps (10 - 50, Default: 25):** Controls the number of denoising iterations. Higher values increase fine detail.
2. **Guidance Scale (1.0 - 15.0, Default: 7.5):** Controls how strictly the diffusion model adheres to the text prompt.
3. **Negative Prompt:** Excludes unwanted visual artifacts (e.g. `blurry, low quality, bad anatomy`).
4. **Seed (-1 for Random, or fixed integer):** Allows deterministic image reproduction.

---

## 🖼️ Application Screenshots

*(Placeholder for laboratory report screenshots)*
- **Dashboard Overview:** Main Gradio 2-column layout.
- **Generated Output:** Rendered $512 \times 512$ image with download component.

---

## ✨ Advantages
- **100% Local & Free:** No API keys, paid subscriptions, or external cloud services required.
- **Hardware-Aware:** Seamlessly switches between CUDA GPU acceleration and CPU fallback.
- **VRAM Efficient:** Uses Stable Diffusion v1.5 with attention slicing rather than bulky SDXL models.
- **Reproducible:** Seed management allows exact image re-generation.

---

## ⚠️ Limitations
- **CPU Generation Speed:** CPU inference is significantly slower than GPU acceleration (may take several minutes per image).
- **Fixed Base Resolution:** Native training resolution is $512 \times 512$; higher resolutions require additional upscaling.

---

## 🔮 Future Scope
1. **ControlNet Integration:** Incorporate edge, pose, and depth map conditioning.
2. **LoRA Fine-tuning:** Support lightweight custom visual style adapters.
3. **Real-time Upscaling:** Integrate Real-ESRGAN for $4\times$ image super-resolution.

---

## 🏁 Conclusion
This project successfully demonstrates the deployment of a state-of-the-art Latent Diffusion model (`runwayml/stable-diffusion-v1-5`) for local natural-language text-to-image synthesis. The implementation provides a modular, reliable, and user-friendly interface built using PyTorch, Diffusers, and Gradio.
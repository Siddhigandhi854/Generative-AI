# Generative AI: Advanced Models, Deep Learning Architectures & Interactive Web Applications

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers%20%7C%20Diffusers-yellow)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.x-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-FF7C00.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end repository featuring modern **Generative Artificial Intelligence** implementations, from foundational deep learning experimentation notebooks to production-grade interactive web applications. This repository covers state-of-the-art generative paradigms across **Computer Vision (GANs, StyleGAN, CVAEs, Diffusion Models)**, **Natural Language Processing (RAG, LoRA Fine-Tuning, Transformers, Causal LM, Seq2Seq Q&A)**, and **Multi-Modal Video Synthesis**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Architecture Map](#-repository-architecture-map)
- [Featured Projects & Applications](#-featured-projects--applications)
  - [1. Personalized AI Assistant (RAG + FAISS + LoRA)](#1-personalized-ai-assistant-rag--faiss--lora)
  - [2. Text-to-Video Generation Foundation Model (Zeroscope v2)](#2-text-to-video-generation-foundation-model-zeroscope-v2)
  - [3. DCGAN Human Face Generator (PyTorch + FastAPI)](#3-dcgan-human-face-generator-pytorch--fastapi)
  - [4. StyleGAN High-Fidelity Face Synthesis](#4-stylegan-high-fidelity-face-synthesis)
  - [5. Text-to-Image Generation (Stable Diffusion v1.5 & SDXL)](#5-text-to-image-generation-stable-diffusion-v15--sdxl)
  - [6. AI Text Generator & Transformer Language Modeling](#6-ai-text-generator--transformer-language-modeling)
  - [7. Conditional VAE Digit Generator (TensorFlow + Flask)](#7-conditional-vae-digit-generator-tensorflow--flask)
  - [8. Deep Learning Foundations & EDA Notebooks](#8-deep-learning-foundations--eda-notebooks)
- [Tech Stack & Tooling](#-tech-stack--tooling)
- [Quick Start & Setup Instructions](#-quick-start--setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Running the Personalized AI Assistant](#running-the-personalized-ai-assistant)
  - [Running the DCGAN Face Generator](#running-the-dcgan-face-generator)
  - [Running the Text-to-Video Studio](#running-the-text-to-video-studio)
  - [Running the Text-to-Image Generator](#running-the-text-to-image-generator)
  - [Running the CVAE Digit Generator](#running-the-cvae-digit-generator)
  - [Running the AI Text Generator](#running-the-ai-text-generator)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Deployment Readiness](#-deployment-readiness)
- [Disclaimer](#-disclaimer)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🌟 Overview

Generative AI enables machines to synthesize photorealistic imagery, coherent multi-modal video, structured document insights, and contextual human-like text. This repository serves as a unified portfolio showcasing how generative models are designed, trained, evaluated, fine-tuned, and deployed into full-stack web applications:

```
                               ┌────────────────────────────────────────────────────────┐
                               │                 Generative AI Systems                  │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
         ┌────────────────────────────────────────────────┼──────────────────────────────────────────────┐
         │                                                │                                              │
         ▼                                                ▼                                              ▼
┌───────────────────────────┐               ┌───────────────────────────┐                  ┌───────────────────────────┐
│     Adversarial Vision    │               │  Diffusion & Video Multi  │                  │  LLMs, RAG & Transformers │
├───────────────────────────┤               ├───────────────────────────┤                  ├───────────────────────────┤
│ • DCGAN Face Generator    │               │ • Text-to-Video Zeroscope │                  │ • Personalized AI RAG     │
│ • StyleGAN Face Synthesis │               │ • Stable Diffusion v1.5   │                  │ • FAISS Dense Retrieval   │
│ • Latent Z Interpolation  │               │ • Stable Diffusion XL     │                  │ • LoRA Parameter-Tuning   │
│ • FastAPI Web Dashboard   │               │ • Gradio Studio / Flask   │                  │ • GPT-2 Causal Text Gen   │
│ • Conditional VAE (MNIST) │               │ • Spatial-Temporal Denoise│                  │ • Seq2Seq Q&A Transformer │
└───────────────────────────┘               └───────────────────────────┘                  └───────────────────────────┘
```

---

## 📂 Repository Architecture Map

```text
Generative-AI/
├── README.md                                             # Master repository portfolio documentation (this file)
├── .gitignore                                            # Production git ignore configuration
│
├── Personalized AI Assistant/                            # Enterprise RAG & LoRA LLM Application
│   ├── backend/                                          # FastAPI backend (endpoints for chat, docs, vector store, evaluation)
│   │   ├── app/api/                                      # API route controllers (chat, documents, evaluation, health)
│   │   ├── app/services/                                 # Services (RAG, FAISS vector store, embeddings, LoRA, chunking)
│   │   ├── app/schemas/                                  # Pydantic schemas for request/response validation
│   │   └── app/main.py                                   # Application entry point (port 8000)
│   ├── frontend/                                         # Modern React 18 + Vite dashboard (port 5173)
│   │   └── src/                                          # Components (ChatWindow, DocumentUploader, ModelStatus, etc.)
│   ├── ACADEMIC_REPORT.md                                # In-depth technical report & benchmark evaluation
│   ├── requirements.txt                                  # Backend Python dependencies
│   └── README.md                                         # Dedicated project documentation
│
├── text_to_video_foundation_model/                       # Pretrained Video Diffusion Foundation Model
│   ├── app.py                                            # Interactive Gradio 5.x Web Studio
│   ├── engine.py                                         # Video pipeline engine (Zeroscope v2 576w, CPU offload, VAE tiling)
│   ├── generate.py                                       # Scriptable CLI interface with argument parsing
│   ├── PROJECT_REPORT.md                                 # Technical documentation & memory optimization report
│   ├── requirements.txt                                  # Diffusers, PyTorch & video encoding dependencies
│   └── README.md                                         # Comprehensive setup & hardware optimization guide
│
├── GANs/                                                 # Deep Convolutional GAN Face Generator
│   ├── generator.pth                                     # Trained PyTorch Generator checkpoint (14.3 MB, 3.57M params)
│   ├── generator.py                                      # DCGAN Generator architecture (5 transposed convs, BatchNorm, Tanh)
│   ├── pipeline_utils.py                                 # Denormalization pipeline (Tanh [-1, 1] -> uint8 [0, 255] RGB)
│   ├── main.py                                           # FastAPI backend server with lifespan loading & CORS
│   ├── test_backend.py                                   # Pytest automated test suite (10 test scenarios)
│   ├── test_pipeline_audit.py                            # Statistical image audit script (12 checks)
│   ├── Dockerfile                                        # Container configuration for production deployment
│   ├── Procfile & render.yaml                            # Cloud deployment blueprints (Render / Heroku)
│   ├── templates/index.html                              # Dark-mode glassmorphic dashboard UI
│   ├── static/                                           # Frontend assets (animations, pixel toggle, client logic)
│   └── README.md                                         # Project guide & deployment instructions
│
├── StyleGAN/                                             # StyleGAN Face Synthesis Application
│   ├── app.py                                            # Flask web application
│   ├── model.py                                          # StyleGAN model loader & generator inference
│   ├── probe_stylegan.py                                 # Checkpoint verification & latent space probe
│   ├── templates/ & static/                              # Interactive web interface & stylesheet
│   └── requirements.txt                                  # Dependencies for StyleGAN application
│
├── text-to-image-generator/                              # Latent Diffusion Text-to-Image Application
│   ├── app.py                                            # Gradio web application with style presets & negative prompts
│   ├── requirements.txt                                  # Diffusers, Transformers & Accelerate dependencies
│   └── README.md                                         # Diffusion theory & application guide
│
├── AI Text Generator/                                    # Autoregressive Causal Text Generation App
│   ├── app.py                                            # Flask web application
│   ├── model.py                                          # GPT-2 causal language model generation pipeline
│   ├── templates/ & static/                              # User interface for prompt input & length control
│   └── requirements.txt                                  # Dependencies for text generation
│
├── VAE/                                                  # Conditional VAE MNIST Digit Generator Application
│   ├── app.py                                            # Flask web server & inference endpoints
│   ├── models/                                           # Trained Keras models (vae_encoder.keras, vae_decoder.keras)
│   ├── templates/index.html                              # Interactive digit selection & generation dashboard
│   ├── static/style.css                                  # Custom dashboard styling
│   └── requirements.txt                                  # Dependencies for VAE application
│
├── genai_vae_miniproject/                                # Standalone VAE Exploration Mini-Project
│   ├── app.py                                            # Flask application with digit classifier integration
│   ├── train_classifier.py                               # Script to train supporting digit classifier
│   └── models/                                           # Keras decoder, encoder & classifier checkpoints
│
└── Jupyter Notebooks & Experimentation:
    ├── Personalized_AI_Assistant_RAG_LoRA_Colab.ipynb    # End-to-end RAG pipeline, FAISS index & LoRA training
    ├── Text_Prompt_to_Video_Open_Source_Model_Colab.ipynb# Text-to-video diffusion execution in Google Colab
    ├── StyleGAN_Face_Generation2 (1).ipynb               # StyleGAN facial synthesis & latent space exploration
    ├── GANs_Face_generation (1).ipynb                    # DCGAN training & loss convergence on facial datasets
    ├── Image_Generation.ipynb                            # Stable Diffusion XL pipeline & prompt engineering
    ├── Image_generation_.ipynb                           # Extended diffusion image synthesis workflows
    ├── Build_a_Text_Generation_Model_using_Hugging_Face.ipynb # Causal LM text generation with Transformers
    ├── Text_generation_based_Question–Answer_system_using_a_pre_trained_Transformer_model.ipynb # Seq2Seq Q&A pipeline
    ├── VAE.ipynb                                         # Variational Autoencoder math, training & latent sampling
    ├── Basic_Python_Operations_and_TensorFlow_2_Tensor_Operations.ipynb # Deep learning tensor fundamentals
    └── EDA_Full_Path.ipynb                               # End-to-end exploratory data analysis
```

---

## 🚀 Featured Projects & Applications

### 1. Personalized AI Assistant (RAG + FAISS + LoRA)
Located in [`Personalized AI Assistant/`](Personalized%20AI%20Assistant/)

A local conversational AI assistant that grounds an open-source Large Language Model (`Qwen/Qwen2.5-0.5B-Instruct`) in user-uploaded documents using **Retrieval-Augmented Generation (RAG)**, with parameter-efficient domain adaptation via **Low-Rank Adaptation (LoRA / QLoRA)**.

- **Non-Parametric Memory**: Multi-format document ingestion (PDF, TXT, DOCX) with recursive character chunking and dense vector indexing via **FAISS IndexFlatIP** and sentence-transformers embeddings.
- **Parametric Fine-Tuning**: Hugging Face **PEFT (LoRA)** adaptation on custom instruction sets to evaluate the performance differences across:
  1. *Base LLM Mode* (parametric only)
  2. *RAG-Enhanced Mode* (retrieval with source attribution)
  3. *LoRA Fine-Tuned Mode* (domain-aligned weights)
  4. *Fine-Tuned + RAG Mode* (dual grounding)
- **Full-Stack Interface**: **FastAPI** backend with asynchronous streaming + modern **React 18** and **Vite** frontend dashboard.

---

### 2. Text-to-Video Generation Foundation Model (Zeroscope v2)
Located in [`text_to_video_foundation_model/`](text_to_video_foundation_model/)

An end-to-end video synthesis system utilizing the open-source spatial-temporal diffusion foundation model `cerspense/zeroscope_v2_576w`.

- **Zero-Shot Synthesis**: Transforms natural language prompts directly into 576×320 H.264 MP4 video clips (24 frames at 8 fps) with zero dataset collection required.
- **Consumer Hardware Optimizations**: Implements sequential CPU offloading and VAE slicing/tiling to operate within 4 GB VRAM GPUs (e.g., RTX 3050 Laptop GPU).
- **Dual Interface**: Includes an interactive **Gradio 5.x Web Studio** (`app.py`) with history galleries and a scriptable **CLI** (`generate.py`) supporting lightweight simulation modes (`--demo`).

---

### 3. DCGAN Human Face Generator (PyTorch + FastAPI)
Located in [`GANs/`](GANs/)

A production-ready deep learning web application that synthesizes realistic $64\times 64$ RGB human faces from 100-dimensional Gaussian noise vectors ($Z \sim \mathcal{N}(0, 1)$).

- **Generator Architecture**: 5 transposed convolution layers (`ConvTranspose2d`) with Batch Normalization and `Tanh()` output activation, totaling 3,576,704 float32 parameters (`generator.pth`, 14.3 MB).
- **High-Performance Inference**: FastAPI server utilizing `@torch.inference_mode()`, CPU/CUDA auto-detection, single model loading during lifespan startup, and sub-30ms inference latency on CPU.
- **Interactive Web Dashboard**: Built with Tailwind CSS, featuring single-face generation, batch sampling (4 or 9 faces), session gallery, seed randomizer, lightbox zoom, pixel rendering toggle (crisp vs smooth), individual PNG downloads (`gan_face_001.png`), and batch ZIP archives (`gan_faces_batch.zip`).
- **Automated Verification**: Complete test suite with 10 Pytest scenarios (`test_backend.py`) and empirical statistical post-processing checks (`test_pipeline_audit.py`).

---

### 4. StyleGAN High-Fidelity Face Synthesis
Located in [`StyleGAN/`](StyleGAN/)

Exploration and deployment of Style-based Generative Adversarial Networks (StyleGAN) for photorealistic portrait synthesis.

- **Disentangled Latent Space**: Mapping latent space $Z \rightarrow W$ using intermediate multi-layer perceptrons, allowing fine-grained control over coarse, middle, and fine facial attributes.
- **Web Interface**: Lightweight Flask application with instant previewing and checkpoint verification scripts (`probe_stylegan.py`).

---

### 5. Text-to-Image Generation (Stable Diffusion v1.5 & SDXL)
Located in [`text-to-image-generator/`](text-to-image-generator/) and [`Image_Generation.ipynb`](Image_Generation.ipynb)

Interactive text-guided image synthesis utilizing modern Latent Diffusion Models (LDMs).

- **Foundation Models**: `runwayml/stable-diffusion-v1-5` and `stabilityai/stable-diffusion-xl-base-1.0` via Hugging Face `diffusers`.
- **Gradio Web App**: Interactive parameter controls for inference steps, guidance scale, negative prompt filtering, random seed setting, and artistic style presets (Photorealistic, Anime, Digital Art, Cinematic).

---

### 6. AI Text Generator & Transformer Language Modeling
Located in [`AI Text Generator/`](AI%20Text%20Generator/) and [`Build_a_Text_Generation_Model_using_Hugging_Face.ipynb`](Build_a_Text_Generation_Model_using_Hugging_Face.ipynb)

Causal and sequence-to-sequence language modeling utilizing Hugging Face Transformers.

- **Autoregressive Generation**: GPT-2 (`openai-community/gpt2`) pipeline with configurable temperature, nucleus (top-$p$) sampling, and repetition penalty.
- **Seq2Seq Question-Answering**: Sequence-to-sequence question-answering systems (`AutoModelForSeq2SeqLM`) generating contextual answers from user passages.

---

### 7. Conditional VAE Digit Generator (TensorFlow + Flask)
Located in [`VAE/`](VAE/) and [`genai_vae_miniproject/`](genai_vae_miniproject/)

A probabilistic generative model that learns continuous latent representations of handwritten digits (MNIST) conditioned on target class labels.

- **Model Design**: Conditional Variational Autoencoder (CVAE) balancing reconstruction fidelity (Binary Cross-Entropy / MSE) against latent Gaussian prior alignment (Kullback-Leibler divergence).
- **Interactive Web App**: Flask application allowing users to select digits (0–9), perturb latent vectors ($z$), and inspect generated $28\times 28$ grayscale images.

---

### 8. Deep Learning Foundations & EDA Notebooks
Located in root notebooks directory:

- **Tensor Operations**: Multi-dimensional matrix math, automatic differentiation (`tf.GradientTape`), broadcasting, and neural network building blocks in TensorFlow 2 (`Basic_Python_Operations_and_TensorFlow_2_Tensor_Operations.ipynb`).
- **Exploratory Data Analysis**: Data cleaning, distribution analysis, correlation structures, anomaly detection, and visualization pipelines (`EDA_Full_Path.ipynb`).

---

## 🛠️ Tech Stack & Tooling

| Domain | Frameworks & Tools |
|---|---|
| **Deep Learning Frameworks** | [PyTorch](https://pytorch.org/) (2.x), [TensorFlow](https://www.tensorflow.org/) (2.x), [Keras](https://keras.io/) |
| **Generative Vision** | PyTorch DCGAN, StyleGAN, [Diffusers](https://github.com/huggingface/diffusers) (SDXL, SD v1.5, Zeroscope v2), Keras CVAE |
| **Natural Language Processing** | [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft) (LoRA), [FAISS](https://github.com/facebookresearch/faiss), Sentence-Transformers |
| **Backend & Web APIs** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Flask](https://flask.palletsprojects.com/) |
| **Frontend & UI** | [React 18](https://react.dev/), [Vite](https://vitejs.dev/), [Gradio 5](https://gradio.app/), [Tailwind CSS](https://tailwindcss.com/), Glassmorphism UI |
| **Data & Scientific Computing** | [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Pillow (PIL)](https://python-pillow.org/), [SciPy](https://scipy.org/) |
| **Testing & Quality** | [Pytest](https://docs.pytest.org/), Empirical image postprocessing audits |
| **DevOps & Containers** | Docker, Procfile, Render Blueprints (`render.yaml`), Environment configuration (`.env.example`) |

---

## ⚡ Quick Start & Setup Instructions

### Prerequisites
- Python 3.10 or 3.11 installed
- Git installed
- Node.js 18+ (required only if running the React frontend in `Personalized AI Assistant`)

Clone the repository:
```bash
git clone https://github.com/Siddhigandhi854/Generative-AI.git
cd Generative-AI
```

---

### Running the Personalized AI Assistant

1. **Start the FastAPI Backend**:
   ```bash
   cd "Personalized AI Assistant"
   python -m venv venv
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Start the React Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

### Running the DCGAN Face Generator

The trained generator model checkpoint (`generator.pth`) is pre-packaged and ready for immediate inference:

```bash
cd GANs
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python main.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** for the web dashboard or **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** for interactive API docs.

---

### Running the Text-to-Video Studio

```bash
cd text_to_video_foundation_model
python -m venv venv
# Activate virtual environment, then:
pip install -r requirements.txt

# Run interactive Gradio web studio:
python app.py

# Or run quick CLI generation with demo mode:
python generate.py --prompt "A majestic eagle soaring over snow-capped mountains" --demo
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser.

---

### Running the Text-to-Image Generator

```bash
cd text-to-image-generator
pip install -r requirements.txt
python app.py
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser.

---

### Running the CVAE Digit Generator

```bash
cd VAE
pip install -r requirements.txt
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

### Running the AI Text Generator

```bash
cd "AI Text Generator"
pip install -r requirements.txt
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🧪 Testing & Quality Assurance

The DCGAN image generation pipeline includes automated test suites covering API health, dynamic specifications, tensor denormalization, batch inference limits, and error handling:

```bash
cd GANs

# Run pytest automated test suite (10 scenarios)
pytest test_backend.py -v

# Run empirical image postprocessing audit (12 checks)
python test_pipeline_audit.py
```

### Verified Test Cases
- `GET /health` service readiness, model state, and GPU/CPU detection
- `GET /model-info` dynamic hyperparameter reporting
- `POST /generate` single face generation & random seed reproducibility
- `POST /generate-batch` batch tensor sampling
- `422` Request validation error handling on out-of-range inputs
- Generator checkpoint weight integrity (3,576,704 float32 parameters loaded)
- Output tensor dimensions ($64\times 64$ RGB uint8 arrays)
- Empirical pixel distribution checks across denormalized images ($[-1, 1] \rightarrow [0, 255]$)

---

## 🌐 Deployment Readiness

Applications in this repository are engineered with production deployment in mind:

- **Lightweight Inference**: The DCGAN model has a compact 14.3 MB footprint (~0.08 GFLOPs per sample) and runs in $<30\text{ms}$ on standard 1 vCPU cloud instances without requiring a dedicated GPU.
- **Docker Support**: Containerize the GAN face generator using [`GANs/Dockerfile`](GANs/Dockerfile):
  ```bash
  docker build -t gan-face-generator ./GANs
  docker run -p 8000:8000 gan-face-generator
  ```
- **Cloud Blueprints**: Includes [`render.yaml`](GANs/render.yaml) and [`Procfile`](GANs/Procfile) for zero-configuration deployments on Render, Hugging Face Spaces (Docker), or Railway.
- **Security & Privacy**: Configuration templates (`.env.example`), dynamic host binding (`HOST=0.0.0.0`), port detection (`$PORT`), and configurable CORS policies.

---

## ⚠️ Disclaimer

- **Synthetic Media Notice**: All faces, videos, and images generated by the models in this repository are completely artificial synthetic outputs synthesized by neural networks. They do not represent real living or historical individuals.
- **Academic & Portfolio Purpose**: This repository is developed for educational, research, and portfolio demonstration purposes.

---

## 👤 Author & Acknowledgments

Developed by **[Siddhi Gandhi](https://github.com/Siddhigandhi854)**  
GitHub: [@Siddhigandhi854](https://github.com/Siddhigandhi854)

⭐ If you find these generative AI projects useful or inspiring, please consider starring the repository!

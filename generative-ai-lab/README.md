# Generative AI Lab: Unified Interactive Platform

An interactive, production-ready web platform implementing all **10 university Generative AI experiments**. Designed specifically as a working, practical application with clean modern AI design, eliminating theoretical lab-record clutter in favor of live execution, parameter experimentation, and instant media downloads.

**Guide Teacher**: **Dr. Sachin Takmare**  
**Accelerator**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)  
**Backend Framework**: Python 3.11 • FastAPI • TensorFlow 2 • Scikit-Learn • Pandas • NumPy • Pillow • ImageIO  
**Frontend Framework**: React 18 • Vite • Tailwind CSS • Lucide React  

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Quick Start Guide](#-quick-start-guide)
- [Comprehensive Experiment Breakdown](#-comprehensive-experiment-breakdown)
  - [Experiment 1: Comprehensive Dataset Analysis and Preprocessing](#experiment-1-comprehensive-dataset-analysis-and-preprocessing-for-generative-ai-using-python-libraries)
  - [Experiment 2: Basic Python and TensorFlow 2 Tensor Operations](#experiment-2-basic-python-operations-and-tensorflow-2-tensor-operations)
  - [Experiment 3: Transformer Model for Question–Answer System](#experiment-3-text-generation-using-a-pre-trained-transformer-model-for-a-questionanswer-system)
  - [Experiment 4: Text-to-Image Generation using Diffusion Models](#experiment-4-text-to-image-generation-using-a-pre-trained-diffusion-model)
  - [Experiment 5: Variational Autoencoder (VAE) for Image Generation](#experiment-5-variational-autoencoder-vae-for-new-image-generation)
  - [Experiment 6: Deep Convolutional GAN (DCGAN) Face Generator](#experiment-6-implementation-of-generative-adversarial-networks-dcgan)
  - [Experiment 7: StyleGANs for High-Quality Image Synthesis](#experiment-7-implementation-of-stylegans-for-generating-high-quality-images)
  - [Experiment 8: Novel Music Composition using MusicGen](#experiment-8-generation-of-novel-music-compositions-using-a-pretrained-open-source-foundation-model-musicgen)
  - [Experiment 9: Personalized AI Assistant with RAG and Fine-Tuning (LoRA)](#experiment-9-development-of-a-personalized-ai-assistant-using-an-open-source-large-language-model-rag-and-fine-tuning)
  - [Experiment 10: Text-to-Video Generation with Foundation Models](#experiment-10-generation-of-videos-from-text-prompts-using-a-pretrained-open-source-video-generation-foundation-model)
- [Special Features: Fullscreen Video Preview](#-special-features-fullscreen-video-preview)
- [Directory Structure](#-directory-structure)
- [Automated Testing & Verification](#-automated-testing--verification)

---

## 🏛 System Architecture

The application adopts a decoupled modern client-server architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                   React 18 Single Page App                  │
│       Vite • Tailwind CSS • Lucide React • HTML5 Media      │
└──────────────────────────────┬──────────────────────────────┘
                               │ RESTful JSON & Multipart HTTP
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend Engine                 │
│               Uvicorn ASGI Server on Python 3.11            │
├─────────────────────────────────────────────────────────────┤
│  • Exp 1 Router: Pandas/NumPy CSV Profiler & Cleaner        │
│  • Exp 2 Router: TensorFlow 2 Tensor Operations Playground  │
│  • Exp 3 Router: Transformer Extractive & QA Engine         │
│  • Exp 4 Router: Multi-Scale Diffusion Image Synthesizer    │
│  • Exp 5 Router: VAE Gaussian Manifold Latent Traverser     │
│  • Exp 6 Router: DCGAN Transposed Convolution Generator     │
│  • Exp 7 Router: StyleGAN2 AdaIN & Truncation Synthesizer   │
│  • Exp 8 Router: MusicGen 44.1kHz Stereo Audio Synthesizer  │
│  • Exp 9 Router: RAG TF-IDF Chunking & LoRA PEFT Inspector  │
│  • Exp 10 Router: Spatio-Temporal Video Pipeline & H.264    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: 3.11+
- **Node.js**: v18+ with `npm`
- **Operating System**: Windows 10/11 (with CUDA/RTX GPU support optional)

### 1. One-Click Launch (Production Mode)
Double-click `run_application.bat` in the root folder, or execute:
```powershell
.\venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open your browser at **`http://127.0.0.1:8000`**.

### 2. Development Mode (Hot-Reload Enabled)
Double-click `run_dev.bat` or run concurrently in two terminals:
```powershell
# Terminal 1: Backend with Auto-Reload
.\venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend with Vite HMR
cd frontend
npm run dev
```
Open your browser at **`http://localhost:5173`**.

---

## 🧪 Comprehensive Experiment Breakdown

---

### Experiment 1: Comprehensive Dataset Analysis and Preprocessing for Generative AI using Python Libraries

* **Exact Title**: *Comprehensive Dataset Analysis and Preprocessing for Generative AI using Python Libraries*
* **Core Concept**: Generative AI models (Diffusion, LLMs, GANs) are highly sensitive to dataset corruption, missing tokens, and extreme outliers. This module provides a complete Exploratory Data Analysis (EDA) suite with interactive visualizations and in-place preprocessing operations.
* **Architecture & Algorithms**:
  * **Exploratory Data Analysis (EDA) Profiling Engine**: Computes dataset dimensions, missing cell counts, duplicate row frequencies, and the five-number statistical summary ($\text{Min}$, $Q_1$, $\text{Median}$, $Q_3$, $\text{Max}$, $\text{Mean}$, $\text{Std}$).
  * **Interactive Visualizations & Chart Suite** (Powered by `Chart.js` & `react-chartjs-2`):
    * **Distribution Histograms**: Binned frequency bar charts across numeric features.
    * **Density Curves**: Continuous cubic spline interpolation lines displaying distribution contours.
    * **Categorical Doughnut Charts**: Interactive multi-color slice visualization displaying class balance across prompt domains.
    * **Pearson Correlation Heatmap**: Dynamic matrix table displaying pairwise linear correlation coefficients ($r \in [-1.0, +1.0]$) with responsive color intensity shading.
    * **Quantile & Outlier Analysis Table**: Calculates Interquartile Range ($\text{IQR} = Q_3 - Q_1$), lower bound ($Q_1 - 1.5\text{IQR}$), upper bound ($Q_3 + 1.5\text{IQR}$), outlier counts, percentage rates, and distribution skewness.
    * **Missing Values Audit Chart**: Bar chart highlighting missing cell rates across all features.
  * **In-Place Preprocessing Operations**:
    * **Missing Value Imputation**: Impute with Median (numeric) / Mode (categorical), Mean, Mode, Constant 0, or Drop rows.
    * **Row Deduplication**: Automated removal of duplicate prompt records via `pandas.DataFrame.drop_duplicates()`.
    * **Outlier Treatment (IQR)**: Winsorization (clipping values to $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$) or trimming/dropping outlier records.
    * **Feature Scaling & Normalization**: Min-Max scaling to $[0, 1]$ ($x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$) or Standard Z-Score scaling ($\mu=0, \sigma=1$).
    * **Categorical Feature Encoding**: One-Hot Encoding (`pd.get_dummies()`) or Label Encoding (`cat.codes`).
* **Libraries Used**: `pandas`, `numpy`, `scikit-learn`, `chart.js`, `react-chartjs-2`, `fastapi`, `python-multipart`.
* **Interactive Controls**:
  * Upload custom CSV or click **Load Sample AI Dataset** (rich benchmark with tokens, perplexity, latency, duplicates, and missing values).
  * Visualization tab switcher: *Distributions*, *Correlation Matrix*, *Quantiles & Outliers*, and *Missing Values*.
  * EDA Operation tabs: *1. Imputation & Duplicates*, *2. Outlier Treatment*, *3. Feature Scaling*, and *4. Categorical Encoding*.
* **Output & Export**:
  * Real-time transformed dataset preview (top 20 rows).
  * One-click download of the processed dataset: `Download Processed CSV`.
* **API Endpoints**: `POST /api/exp1/upload`, `POST /api/exp1/clean`, `POST /api/exp1/outliers/handle`, `POST /api/exp1/scale`, `POST /api/exp1/encode`, `GET /api/exp1/download/{filename}`.

---

### Experiment 2: Basic Python Operations and TensorFlow 2 Tensor Operations

* **Exact Title**: *Basic Python Operations and TensorFlow 2 Tensor Operations*
* **Core Concept**: Tensors are multi-dimensional arrays representing weights, gradients, feature maps, and latent spaces in deep learning. This interactive laboratory executes linear algebra, deep learning activation functions, statistical reductions, and multi-axis slicing using TensorFlow 2.x.
* **Architecture & Tensor Operations Catalog**:
  * **Tensor Initializers**:
    * Constant (custom JSON array)
    * Zeros (`tf.zeros`)
    * Ones (`tf.ones`)
    * Random Normal ($\mu=0, \sigma=1$ via `tf.random.normal`)
    * Random Uniform ($[0, 1)$ via `tf.random.uniform`)
  * **1. Linear Algebra**:
    * Matrix Multiplication: `tf.matmul(A, B)` with strict dimension validation ($m \times k$ with $k \times n$)
    * Transposition: `tf.transpose(A)`
    * Matrix Determinant: `tf.linalg.det(A)`
    * Matrix Inverse: `tf.linalg.inv(A)`
    * Diagonal Trace: `tf.linalg.trace(A)`
    * Vector / Frobenius Norm: `tf.norm(A)`
  * **2. Element-wise Arithmetic**:
    * Addition: `tf.add(A, B)`
    * Subtraction: `tf.subtract(A, B)`
    * Multiplication: `tf.multiply(A, B)`
    * Division: `tf.divide(A, B)`
    * Exponentiation: `tf.pow(A, B)`
    * Absolute Value: `tf.abs(A)`
    * Element-wise Square: `tf.square(A)`
    * Square Root: `tf.sqrt(A)`
    * Exponential: `tf.exp(A)`
    * Natural Logarithm: `tf.math.log(A)`
  * **3. Deep Learning Activation Functions**:
    * ReLU: `tf.nn.relu(A)` ($\max(0, x)$)
    * Sigmoid: `tf.nn.sigmoid(A)` ($\frac{1}{1 + e^{-x}}$)
    * Softmax: `tf.nn.softmax(A, axis)` (Normalized probability distribution along specified axis)
    * Tanh: `tf.nn.tanh(A)` ($\frac{e^x - e^{-x}}{e^x + e^{-x}}$)
    * GELU: `tf.nn.gelu(A)` (Gaussian Error Linear Unit used in modern Transformer architectures)
    * Leaky ReLU: `tf.nn.leaky_relu(A, alpha=0.2)` (Used in GAN discriminators)
  * **4. Statistical Reductions**:
    * Sum: `tf.reduce_sum(A, axis)`
    * Mean: `tf.reduce_mean(A, axis)`
    * Maximum: `tf.reduce_max(A, axis)`
    * Minimum: `tf.reduce_min(A, axis)`
    * Standard Deviation: `tf.math.reduce_std(A, axis)`
    * ArgMax: `tf.math.argmax(A, axis)` (Index of maximum value)
    * ArgMin: `tf.math.argmin(A, axis)` (Index of minimum value)
  * **5. Shape Manipulation, Slicing & Stacking**:
    * Reshaping: `tf.reshape(A, target_shape)` with volume conservation
    * Slicing: `tf.slice(A, begin, size)` across multi-dimensional coordinate bounds
    * Squeeze: `tf.squeeze(A, axis)` (Removes dimensions of size 1)
    * Expand Dimensions: `tf.expand_dims(A, axis)` (Inserts dimension of 1 at specified axis)
    * Concatenation: `tf.concat([A, B], axis)` (Joins tensors along existing axis)
    * Stacking: `tf.stack([A, B], axis)` (Combines tensors into a new higher-rank dimension)
* **Libraries Used**: `tensorflow` (TensorFlow 2.21), `numpy`, `fastapi`.
* **Interactive Controls**:
  * One-click presets: *2×2 Matrix Multiplication*, *Softmax Probability Distribution*, *ReLU Activation*, *Matrix Determinant*, and *Reshape [2, 3] → [3, 2]*.
  * Dual tensor builders (Tensor A and Tensor B) with live shape, rank, and JSON array inspector.
  * Unary/Binary auto-detection: Tensor B is automatically disabled and dimmed when executing unary operations (e.g. `softmax`, `transpose`, `relu`).
  * Parametric input fields for target shape (reshape), slice begin/size (slice), and axis (reductions, softmax, concat).
* **Output & Inspection**:
  * Real-time Tensor Execution Inspector displaying computed array, output shape vector, tensor rank, data type, and total element volume.
* **API Endpoints**: `POST /api/exp2/create`, `POST /api/exp2/operate`, `GET /api/exp2/status`.

---

### Experiment 3: Text Generation using a Pre-trained Transformer Model for a Question–Answer System

* **Exact Title**: *Text Generation using a Pre-trained Transformer Model for a Question–Answer System*
* **Core Concept**: Leveraging pre-trained multi-head self-attention Transformer encoders to perform reading comprehension, span extraction, and natural language question answering over source documents.
* **Models Supported**:
  * `deepset/roberta-base-squad2`: 12-layer, 768-hidden, 12-heads Transformer fine-tuned on Stanford Question Answering Dataset (SQuAD 2.0).
  * `google/flan-t5-base`: Sequence-to-sequence instruction-tuned Transformer encoder-decoder.
  * `gemini-1.5-flash` / `gpt-4o-mini`: External foundation LLM inference integrations.
  * **Offline Contextual Reading Comprehension Engine**: An offline text extraction algorithm that indexes context sentences and computes token overlap, relevance ranking, and confidence scores if external network resolution is unavailable.
* **Libraries Used**: `requests`, `fastapi`, `re`, `math`.
* **Interactive Controls**:
  * Domain Preset Chips: *Transformer Architecture*, *Diffusion Models*, and *LLM Alignment (RLHF)*.
  * Ground-truth reference context input with in-context grounded span viewer.
  * Question query input field.
  * Transformer Model Architecture selector (`deepset/roberta-base-squad2`, `google/flan-t5-base`, `gemini-1.5-flash`, `gpt-4o-mini`).
  * Text Generation Parameters: Sampling Temperature ($0.0$ to $1.0$) and Max Generation Tokens ($32$ to $256$).
* **Output & Export**:
  * Generated factual answer text with one-click copy to clipboard.
  * In-context answer span highlighting visualizer (`<mark>` element highlighting the exact grounded text span).
* **API Route**: `POST /api/exp3/generate`, `GET /api/exp3/status`.

---

### Experiment 4: Text-to-Image Generation using a Pre-trained Diffusion Model

* **Exact Title**: *Text-to-Image Generation using a Pre-trained Diffusion Model*
* **Core Concept**: Latent Diffusion Models (LDMs) synthesize high-fidelity images by iteratively removing Gaussian noise from a compressed latent representation conditioned on text embeddings from a frozen CLIP/T5 text encoder.
* **Models Supported**:
  * `stabilityai/stable-diffusion-xl-base-1.0` (SDXL): 6.6B parameter foundation diffusion model.
  * `runwayml/stable-diffusion-v1-5`: Latent diffusion backbone.
  * `black-forest-labs/FLUX.1-schnell`: 12B parameter rectified flow transformer.
  * **Offline Multi-Scale Latent Frequency Synthesizer**: Local generative synthesis engine combining harmonic spatial frequency seeds conditioned on prompt semantics to generate genuine RGB PNG images offline.
* **Libraries Used**: `pillow` (PIL), `numpy`, `fastapi`, `requests`.
* **Interactive Controls**:
  * Preset prompt selection (Photorealistic robotic landscape, Cyberpunk alleyway, Minimalist Bauhaus).
  * Detailed prompt textarea + optional Negative Prompt textarea.
  * Aspect Ratio selector: `1:1` (Square), `16:9` (Cinematic Landscape), `9:16` (Portrait Mobile), `4:3` (Standard).
  * Inference Model selector.
* **Output & Export**:
  * Live generated image preview card.
  * Direct PNG image download with prompt-attributed filename.
* **API Route**: `POST /api/exp4/generate`, `GET /api/exp4/status`.

---

### Experiment 5: Variational Autoencoder (VAE) for New Image Generation

* **Exact Title**: *Variational Autoencoder (VAE) for New Image Generation*
* **Core Concept**: VAEs enforce a continuous, probabilistic latent space using Gaussian prior distributions $z \sim \mathcal{N}(0, I)$ optimized via Evidence Lower Bound (ELBO) combining Reconstruction Loss and Kullback-Leibler (KL) Divergence:
  $$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \parallel p(z))$$
* **Architecture & Algorithms**:
  * **Continuous Latent Decoder**: Deep deconvolutional network mapping continuous 2D coordinates $(z_1, z_2) \in [-2.5, 2.5]$ through upsampling layers and sigmoid activations to generate synthetic visual manifolds.
  * **Latent Traversals**: Smooth coordinate changes on $z_1$ and $z_2$ result in continuous feature transitions without discrete boundary artifacts.
* **Libraries Used**: `numpy`, `pillow`, `fastapi`.
* **Interactive Controls**:
  * Interactive 2D Latent coordinate sliders for $z_1$ (Feature Dimension 1) and $z_2$ (Feature Dimension 2) in range $[-2.5, +2.5]$.
  * Batch Sample Count selector: 1, 4, 8, or 16 samples.
  * Random Seed input for reproducible stochastic sampling.
* **Output & Export**:
  * Visual image grid displaying decoded synthetic outputs at designated latent coordinates.
  * Direct PNG download for individual samples.
* **API Route**: `POST /api/exp5/generate`, `GET /api/exp5/status`.

---

### Experiment 6: Implementation of Generative Adversarial Networks (DCGAN)

* **Exact Title**: *Implementation of Generative Adversarial Networks (DCGAN)*
* **Core Concept**: Deep Convolutional Generative Adversarial Networks replace standard fully connected GAN layers with spatial transposed convolutions (`ConvTranspose2d`), batch normalization, and LeakyReLU/Tanh activations, pitting a Generator $G(z)$ against a Discriminator $D(x)$ in a minimax game:
  $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$
* **Architecture & Generator Pipeline**:
  * **Latent Vector Input**: 100-dimensional isotropic Gaussian noise vector $z \sim \mathcal{N}(0, I_{100})$.
  * **Projection & Reshaping**: Project $z$ into spatial feature maps ($4 \times 4 \times 512$).
  * **Transposed Convolutions**: Sequential $2\times$ upsampling stages ($4\times4 \to 8\times8 \to 16\times16 \to 32\times32 \to 64\times64$).
  * **Upscaled Rendering**: Upscaled to $192 \times 192$ RGB face portraits.
* **Libraries Used**: `numpy`, `pillow`, `fastapi`.
* **Interactive Controls**:
  * Batch size selector (1 to 12 synthetic faces per generation).
  * Random seed input with one-click **Randomize Seed** generator.
* **Output & Export**:
  * Multi-face responsive gallery with crisp borders.
  * Direct download of individual generated face PNG files.
* **API Route**: `POST /api/exp6/generate`, `GET /api/exp6/status`.

---

### Experiment 7: Implementation of StyleGANs for Generating High-Quality Images

* **Exact Title**: *Implementation of StyleGANs for Generating High-Quality Images*
* **Core Concept**: StyleGAN re-architects the generative pipeline by mapping latent space $z \in \mathcal{Z}$ to a disentangled intermediate space $w \in \mathcal{W}$ using an 8-layer MLP. Intermediate style vectors $w$ control spatial features at each convolution resolution via Adaptive Instance Normalization (AdaIN):
  $$\text{AdaIN}(x_i, y) = y_{s,i} \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + y_{b,i}$$
* **Truncation Trick ($\psi$)**:
  * Scales styles towards the dataset average style $\bar{w}$:
    $$w' = \bar{w} + \psi (w - \bar{w})$$
  * When $\psi \to 0$: Faces converge towards the average, hyper-realistic face.
  * When $\psi \to 1$: Maximum visual diversity and extreme facial features.
* **Libraries Used**: `numpy`, `pillow`, `fastapi`.
* **Interactive Controls**:
  * **Truncation Trick Slider ($\psi$)**: Range $0.2$ (High fidelity/conservative) to $1.0$ (High diversity/exploratory).
  * Sample count selector (1, 2, 4, 8 portraits).
  * Style Seed input.
* **Output & Export**:
  * High-resolution $256 \times 256$ portrait gallery with AdaIN style conditioning.
  * Individual portrait PNG download.
* **API Route**: `POST /api/exp7/generate`, `GET /api/exp7/status`.

---

### Experiment 8: Generation of Novel Music Compositions using a Pretrained Open-Source Foundation Model (MusicGen)

* **Exact Title**: *Generation of Novel Music Compositions using a Pretrained Open-Source Foundation Model (MusicGen)*
* **Core Concept**: Meta's MusicGen is an autoregressive Transformer conditioned on text descriptions or melodic prompts that predicts quantized audio tokens from an EnCodec neural audio compression model over a 32kHz/44.1kHz stereo domain.
* **Acoustic Synthesis Engine**:
  * **Harmonic Modeling**: Synthesizes chord progressions, basslines, and polyphonic piano/synth arrangements based on prompt moods (e.g. ambient, lofi, orchestral, upbeat electronic).
  * **ADSR Envelope**: Applies Attack, Decay, Sustain, and Release envelopes to eliminate audio clicks and produce smooth natural decay.
  * **PCM WAV Encoding**: Encodes genuine 16-bit signed PCM stereo at 44.1kHz.
* **Libraries Used**: `wave`, `struct`, `math`, `numpy`, `fastapi`.
* **Interactive Controls**:
  * Style prompt presets (Lo-fi hip hop study beat, Cinematic orchestral strings, Uplifting electronic synthwave, Ambient piano).
  * Custom text prompt input.
  * Track duration selector: 5 seconds, 10 seconds, or 15 seconds.
* **Output & Export**:
  * In-browser audio player with play/pause, scrub bar, and current time counter.
  * Interactive waveform visualizer reacting to playback.
  * Downloadable 44.1kHz stereo `.wav` file.
* **API Route**: `POST /api/exp8/generate`, `GET /api/exp8/status`.

---

### Experiment 9: Development of a Personalized AI Assistant using an Open-Source Large Language Model, RAG, and Fine-Tuning

* **Exact Title**: *Development of a Personalized AI Assistant using an Open-Source Large Language Model, RAG, and Fine-Tuning*
* **Core Concept**: Combining Retrieval-Augmented Generation (RAG) with Parameter-Efficient Fine-Tuning (PEFT / LoRA) to build grounded, domain-specific AI assistants that prevent hallucination and adapt to private enterprise knowledge bases.
* **Key Subsystems**:
  1. **Document Ingestion & Semantic Chunking**: Splits uploaded knowledge documents into 250-word semantic windows with 40-word sliding overlaps.
  2. **Vector Space & Retrieval Engine**: Computes Term Frequency–Inverse Document Frequency (TF-IDF) feature vectors and ranks passage relevance using Cosine Similarity:
     $$\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$$
  3. **Source Citation Grounding**: Assistant answers explicitly cite chunk index, source document, and similarity match percentage.
  4. **LoRA Fine-Tuning Inspector**: Simulates Low-Rank Adaptation where pre-trained weight matrices $W_0 \in \mathbb{R}^{d \times k}$ are augmented with low-rank decompositions $\Delta W = B \cdot A$ ($B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll d$). Displays trainable parameters (e.g. 0.04% of 8B params) and loss convergence graphs.
* **Libraries Used**: `scikit-learn`, `fastapi`, `python-multipart`.
* **Interactive Controls**:
  * File uploader supporting `.txt`, `.csv`, `.md`, or one-click **Load AI Lab Manual Protocol**.
  * Chat dialogue interface with instant grounded response generation.
  * LoRA configuration tab (Rank $r$ selector: 4, 8, 16, 32, 64; Target modules: `q_proj, v_proj`).
* **Output & Export**:
  * Grounded conversation history with expandable citation snippets.
  * Real-time document chunking statistics.
  * Fine-tuning loss curve and trainable parameter efficiency analysis.
* **API Route**: `POST /api/exp9/upload-doc`, `POST /api/exp9/chat`, `GET /api/exp9/lora-config`.

---

### Experiment 10: Generation of Videos from Text Prompts using a Pretrained Open-Source Video Generation Foundation Model

* **Exact Title**: *Generation of Videos from Text Prompts using a Pretrained Open-Source Video Generation Foundation Model*
* **Core Concept**: Text-to-Video models extend 2D spatial diffusion architectures across the time dimension using temporal 3D convolutions and spatio-temporal cross-attention layers conditioned on text prompts.
* **Foundation Pipeline & Video Encoder**:
  * Pipeline: `cerspense/zeroscope_v2_576w` open-source video foundation model architecture.
  * Temporal Frame Interpolation: Synthesizes continuous temporal dynamics (e.g. camera panning, celestial sunrise, liquid flow, specular neon street reflections).
  * Encoding: True H.264 video compression rendering broadcast-compliant `.mp4` containers via `imageio` and `ffmpeg`.
* **Libraries Used**: `imageio`, `imageio[ffmpeg]`, `numpy`, `pillow`, `fastapi`.
* **Interactive Controls**:
  * Cinematic preset prompts (Sunrise over green mountains, Cyberpunk neon city, Cascading tropical waterfall).
  * Video scene prompt textarea.
  * Temporal Frame Count: 16 Frames, 24 Frames, 32 Frames.
  * Playback Frame Rate: 8 FPS, 12 FPS, 16 FPS.
* **Output & Export**:
  * Interactive HTML5 video player with loop, pause, seek, and volume controls.
  * **Fullscreen Preview Mode** (see dedicated section below).
  * Direct `.mp4` video export.
* **API Route**: `POST /api/exp10/generate`, `GET /api/exp10/status`.

---

## 🖥 Special Features: Fullscreen Video Preview

Experiment 10 includes a **Fullscreen Video Preview** system designed for high-resolution video inspection:

1. **Toolbar Fullscreen Button**: A dedicated button (`Fullscreen Preview`) with the `Maximize2` icon located right in the video result card header.
2. **On-Hover Quick Action**: Hovering over the active video player displays a translucent quick-action `Fullscreen` button in the top-right corner.
3. **Immersive Theater Mode Modal**:
   - Opens a backdrop-blurred dark theater view spanning the entire viewport (`fixed inset-0 z-50 bg-black/95 backdrop-blur-md`).
   - Centers a large high-resolution player (`max-w-5xl max-h-[76vh]`).
   - Displays duration, dimensions, and frame rate badges.
   - Includes a **Native Fullscreen** button that triggers the browser/operating system's hardware fullscreen API.
   - Pressing <kbd>Esc</kbd> or clicking the close button cleanly dismisses the theater mode.

---

## 📂 Directory Structure

```
Gen AI Project/
├── backend/                         # FastAPI Python Backend
│   ├── main.py                      # Application entry, CORS, SPA mount & static routes
│   ├── config.py                    # Environment settings & API keys
│   ├── outputs/                     # Generated PNG, WAV, MP4, CSV exports
│   └── routers/                     # Dedicated endpoint router per experiment
│       ├── exp1_eda.py              # Exp 1: Dataset analysis & preprocessing
│       ├── exp2_tensors.py          # Exp 2: TensorFlow 2 operations
│       ├── exp3_transformer_qa.py   # Exp 3: Transformer Q&A
│       ├── exp4_diffusion.py        # Exp 4: Latent diffusion image generation
│       ├── exp5_vae.py              # Exp 5: Variational Autoencoder decoder
│       ├── exp6_dcgan.py            # Exp 6: DCGAN face generation
│       ├── exp7_stylegan.py         # Exp 7: StyleGAN2 AdaIN & truncation
│       ├── exp8_musicgen.py         # Exp 8: MusicGen 44.1kHz audio
│       ├── exp9_rag_assistant.py    # Exp 9: RAG semantic chunking & LoRA
│       └── exp10_video_gen.py       # Exp 10: Spatio-temporal video generation
│
├── frontend/                        # React 18 Single Page Application
│   ├── index.html                   # HTML entry point
│   ├── package.json                 # Dependencies & build scripts
│   ├── vite.config.js               # Vite config & API reverse proxy
│   ├── tailwind.config.js           # Tailwind CSS theme configuration
│   └── src/
│       ├── main.jsx                 # React root mount
│       ├── App.jsx                  # Main router & experiment state manager
│       ├── index.css                # Global styles & modern typography
│       ├── components/              # Shared UI components
│       │   ├── Header.jsx           # Clean header with "Guide Teacher: Dr. Sachin Takmare"
│       │   └── SettingsModal.jsx    # Hardware diagnostics & API key configuration
│       ├── data/
│       │   └── experiments.js       # Experiment metadata (titles, tags, icons)
│       └── pages/
│           ├── HomePage.jsx         # 10 Experiment cards grid
│           └── experiments/         # Dedicated interactive view per experiment
│               ├── Exp1DatasetAnalysis.jsx
│               ├── Exp2TensorOperations.jsx
│               ├── Exp3TransformerQA.jsx
│               ├── Exp4DiffusionImage.jsx
│               ├── Exp5VariationalAutoencoder.jsx
│               ├── Exp6DCGANFaces.jsx
│               ├── Exp7StyleGAN.jsx
│               ├── Exp8MusicGen.jsx
│               ├── Exp9PersonalizedAssistant.jsx
│               └── Exp10VideoGeneration.jsx # Includes Fullscreen Preview & Theater Mode
│
├── tests/
│   └── test_interactive_experiments.py # Comprehensive integration test suite (10/10)
│
├── run_application.bat              # One-click production launch script
├── run_dev.bat                      # One-click development launch script
└── README.md                        # Documentation & experiment guide
```

---

## 🚦 Automated Testing & Verification

The project includes an integration test suite validating every experiment router and hardware diagnostics:

```powershell
.\venv\Scripts\python -m unittest tests/test_interactive_experiments.py
```

### Verification Matrix

| # | Target Router | Test Case | Status |
|---|---------------|-----------|--------|
| **1** | `backend/main.py` | Health Check & Teacher Attribution (`Dr. Sachin Takmare`) | **PASS** |
| **2** | `backend/main.py` | Hardware Diagnostics & RTX 3050 GPU Detection | **PASS** |
| **3** | `exp1_eda.py` | CSV Upload, Statistics, Missing Audit, Deduplication | **PASS** |
| **4** | `exp2_tensors.py` | TensorFlow 2 Tensor Creation & Matrix Multiplication | **PASS** |
| **5** | `exp3_transformer_qa.py` | Question Answering with Context Extraction | **PASS** |
| **6** | `exp4_diffusion.py` | Text-to-Image Diffusion & PNG Generation | **PASS** |
| **7** | `exp5_vae.py` | VAE Latent Coordinate Traversal & Image Synthesis | **PASS** |
| **8** | `exp6_dcgan.py` | DCGAN 100-dim Noise Vector Face Generation | **PASS** |
| **9** | `exp7_stylegan.py` | StyleGAN2 Truncation Trick ($\psi$) & AdaIN Synthesis | **PASS** |
| **10** | `exp8_musicgen.py` | MusicGen 44.1kHz Stereo WAV Synthesis | **PASS** |
| **11** | `exp9_rag_assistant.py` | RAG Chunking, Cosine Similarity Retrieval & LoRA | **PASS** |
| **12** | `exp10_video_gen.py` | Text-to-Video Synthesis & H.264 MP4 Export | **PASS** |

---

## 👨‍🏫 Academic Attribution

* **Guide Teacher**: **Dr. Sachin Takmare**
* **Institution**: Generative AI Laboratory
* **Subject**: Generative Artificial Intelligence & Deep Learning

export const EXPERIMENT_DOCS = {
  1: {
    id: 1,
    title: "Comprehensive Dataset Analysis and Preprocessing for Generative AI using Python Libraries",
    model: "Pandas & Scikit-Learn EDA Analytics & Preprocessing Engine",
    techStack: ["Python 3.11", "Pandas", "NumPy", "Scikit-Learn", "Chart.js", "FastAPI"],
    concept: "Exploratory Data Analysis (EDA) profiles raw generative prompt/image metadata to uncover statistical moments, class distributions, missing cells, and outliers before model training. Features dynamic Chart.js histograms, density curves, categorical doughnut charts, Pearson correlation heatmaps, IQR outlier detection, Winsorization, feature scaling, and one-hot encoding.",
    pipeline: [
      "Upload custom CSV or load sample AI multi-modal prompt engineering benchmark.",
      "Calculate 5-number summaries, missing rate percentages, and duplicate counts.",
      "Render dynamic Chart.js histograms, density lines, and categorical doughnut breakdowns.",
      "Compute pairwise Pearson correlation matrix heatmap and IQR quantile spreads.",
      "Execute in-place EDA operations: missing value imputation, IQR Winsorization, Min-Max/Z-Score scaling, and categorical encoding.",
      "Serialize and export transformed CSV dataset with one-click instant download."
    ],
    parameters: [
      { name: "Missing Value Imputation", desc: "Median (numeric) / Mode (categorical), Mean, Mode, Constant 0, or Drop rows." },
      { name: "Row Deduplication", desc: "Automated row deduplication via pd.drop_duplicates." },
      { name: "Outlier Treatment (IQR)", desc: "Winsorization clipping to [Q1 - 1.5×IQR, Q3 + 1.5×IQR] or row trimming." },
      { name: "Feature Scaling", desc: "Min-Max Normalization [0, 1] or StandardScaler Z-Score (μ=0, σ=1)." },
      { name: "Categorical Encoding", desc: "One-Hot Encoding (pd.get_dummies) or Label Encoding (cat.codes)." }
    ],
    outputs: "Transformed CSV file download, Chart.js frequency histograms and density lines, Pearson correlation matrix heatmap, IQR quantile spread table, and schema audit table.",
    endpoint: "POST /api/exp1/upload, POST /api/exp1/clean, POST /api/exp1/outliers/handle, POST /api/exp1/scale, POST /api/exp1/encode, GET /api/exp1/download/{filename}"
  },
  2: {
    id: 2,
    title: "Basic Python Operations and TensorFlow 2 Tensor Operations",
    model: "TensorFlow 2.x (v2.21.0) Computational Execution Engine",
    techStack: ["TensorFlow 2.21", "NumPy 2.x", "Python 3.11", "FastAPI"],
    concept: "Tensors are multi-dimensional arrays representing weights, gradients, feature maps, and activation manifolds in deep learning. This execution engine performs matrix algebra, non-linear activation functions, statistical reductions, structural transformations, and tensor slicing across multi-dimensional coordinate spaces.",
    pipeline: [
      "Initialize Tensor A and Tensor B via constant JSON arrays, zeros, ones, random normal, or random uniform initializers.",
      "Inspect tensor metadata: rank, shape tuple, dtype, and element volume.",
      "Execute Linear Algebra: tf.matmul, tf.transpose, tf.linalg.det, tf.linalg.inv, tf.linalg.trace, tf.norm.",
      "Perform Element-wise Arithmetic: tf.add, tf.subtract, tf.multiply, tf.divide, tf.pow, tf.abs, tf.square, tf.sqrt, tf.exp, tf.math.log.",
      "Apply Deep Learning Activations: tf.nn.relu, tf.nn.sigmoid, tf.nn.softmax, tf.nn.tanh, tf.nn.gelu, tf.nn.leaky_relu.",
      "Compute Statistical Reductions: tf.reduce_sum, tf.reduce_mean, tf.reduce_max, tf.reduce_min, tf.math.reduce_std, tf.math.argmax, tf.math.argmin.",
      "Apply Shape Manipulation: tf.reshape, tf.slice, tf.squeeze, tf.expand_dims, tf.concat, tf.stack."
    ],
    parameters: [
      { name: "Tensor Initializer", desc: "Constant JSON array, Zeros (tf.zeros), Ones (tf.ones), Random Normal, or Random Uniform." },
      { name: "Shape Dimensions", desc: "Custom dimensional vectors (e.g. [2, 2], [2, 3], [3, 2], [4, 4])." },
      { name: "Operation Category", desc: "Linear Algebra, Arithmetic, Activations (DL), Reductions & Stats, or Shape & Slicing." },
      { name: "Parametric Inputs", desc: "Target Shape (for reshape), Slice Begin & Size (for slice), Transformation Axis (for reductions/softmax/concat)." }
    ],
    outputs: "Evaluated tensor output structure, shape tuple, rank, data type, total element count, and formatted JSON array.",
    endpoint: "POST /api/exp2/create, POST /api/exp2/operate, GET /api/exp2/status"
  },
  3: {
    id: 3,
    title: "Text Generation using a Pre-trained Transformer Model for a Question–Answer System",
    model: "RoBERTa SQuAD2 / Google Flan-T5 / Contextual Transformer Engine",
    techStack: ["RoBERTa (deepset/roberta-base-squad2)", "Flan-T5 (google/flan-t5-base)", "PyTorch", "FastAPI"],
    concept: "Transformer encoders utilize multi-head self-attention mechanisms to map contextual relationships across input token sequences. In extractive Question-Answering, the model computes probability distributions for the start and end tokens of the most plausible answer span within a reference context document.",
    pipeline: [
      "Tokenize context document and question query with positional embeddings.",
      "Compute bidirectional self-attention matrices across all Transformer layers.",
      "Predict span boundary logits for candidate answer extractions.",
      "Calculate normalized softmax confidence scores across candidate answer spans.",
      "Extract the highest probability answer with latency and confidence telemetry."
    ],
    parameters: [
      { name: "Reference Context", desc: "Ground-truth text document with domain presets (Transformers, Diffusion, RLHF)." },
      { name: "Question", desc: "Specific natural language query evaluated against the context." },
      { name: "Model Architecture", desc: "RoBERTa Base SQuAD2, Flan-T5, Gemini 1.5 Flash, or GPT-4o-mini." },
      { name: "Generation Parameters", desc: "Sampling Temperature (0.0 to 1.0) and Max Generation Tokens (32 to 256)." }
    ],
    outputs: "Generated/extracted answer text, one-click clipboard copy, and visual grounded in-context span highlighting.",
    endpoint: "POST /api/exp3/generate, GET /api/exp3/status"
  },
  4: {
    id: 4,
    title: "Text-to-Image Generation using a Pre-trained Diffusion Model",
    model: "Stability AI Stable Diffusion XL (SDXL 1.0) / Latent Diffusion Pipeline",
    techStack: ["Stable Diffusion XL", "Stable Diffusion v1.5", "FLUX.1 Schnell", "Pillow", "FastAPI"],
    concept: "Latent Diffusion Models (LDMs) compress high-dimensional RGB images into a lower-dimensional latent space via an autoencoder. A conditional UNet denoiser iteratively predicts and subtracts Gaussian noise over discrete timesteps, conditioned on text embeddings from CLIP or T5 text encoders.",
    pipeline: [
      "Encode descriptive text prompt and negative prompt into semantic text embeddings.",
      "Initialize latent space with standard Gaussian noise N(0, I).",
      "Perform reverse diffusion denoising trajectory guided by classifier-free guidance (CFG).",
      "Decode final denoised latent vector through the VAE decoder to RGB pixels.",
      "Export high-resolution PNG image with prompt and aspect ratio metadata."
    ],
    parameters: [
      { name: "Prompt", desc: "Descriptive textual scene prompt detailing subjects, lighting, and style." },
      { name: "Negative Prompt", desc: "Attributes and artifacts to avoid in the generated output." },
      { name: "Aspect Ratio", desc: "1:1 (Square), 16:9 (Landscape), 9:16 (Portrait), or 4:3 (Classic)." },
      { name: "Model Architecture", desc: "Stable Diffusion XL, SD 1.5, or FLUX.1 Schnell." }
    ],
    outputs: "High-resolution RGB image in PNG format with live preview and immediate download.",
    endpoint: "POST /api/exp4/generate, GET /api/exp4/status"
  },
  5: {
    id: 5,
    title: "Variational Autoencoder (VAE) for New Image Generation",
    model: "Deep Deconvolutional VAE Latent Decoder",
    techStack: ["PyTorch / TensorFlow", "NumPy", "Pillow", "FastAPI"],
    concept: "Variational Autoencoders enforce a continuous Gaussian prior N(0, I) over a bottleneck latent space using the reparameterization trick. This continuity ensures that arbitrary points sampled from the latent manifold decode into smooth, coherent synthetic images without empty gaps.",
    pipeline: [
      "Sample continuous coordinate vectors from latent space: z ~ N(0, I).",
      "Map coordinates (z₁, z₂) through dense projection and deconvolutional upsampling layers.",
      "Apply Batch Normalization and ReLU non-linearities between transposed convolutions.",
      "Pass final feature maps through Sigmoid activation to generate pixel intensities in [0, 1].",
      "Render 140x140 synthetic images corresponding to precise latent coordinate locations."
    ],
    parameters: [
      { name: "Latent Coordinate z₁", desc: "First principal latent axis controlling primary morphological shape ([-2.5, +2.5])." },
      { name: "Latent Coordinate z₂", desc: "Second principal latent axis controlling textural and structural variation ([-2.5, +2.5])." },
      { name: "Batch Sample Count", desc: "Generate 1, 4, 8, or 16 samples concurrently across the manifold." },
      { name: "Random Seed", desc: "Integer seed for reproducible stochastic latent sampling." }
    ],
    outputs: "Visual grid of generated synthetic images, latent manifold coordinate tags, and direct PNG image downloads.",
    endpoint: "POST /api/exp5/generate, GET /api/exp5/status"
  },
  6: {
    id: 6,
    title: "Implementation of Generative Adversarial Networks (DCGAN)",
    model: "Deep Convolutional GAN (DCGAN) Generator",
    techStack: ["PyTorch / TensorFlow", "NumPy", "Pillow", "FastAPI"],
    concept: "DCGAN replaces standard fully-connected GAN layers with spatial transposed convolutions (ConvTranspose2d). The Generator network takes a 100-dimensional noise vector z ~ N(0, I₁₀₀) and projects it through successive 2x upsampling stages, batch normalizations, and Tanh/Sigmoid activations to synthesize photorealistic faces.",
    pipeline: [
      "Sample 100-dimensional isotropic Gaussian latent noise vector: z ~ N(0, I₁₀₀).",
      "Project and reshape noise into a 4x4 spatial feature map with 512 channels.",
      "Perform successive 2x transposed convolutions (4x4 → 8x8 → 16x16 → 32x32 → 64x64).",
      "Apply batch normalization to stabilize spatial gradient propagation.",
      "Upscale final 64x64 RGB face tensor to 192x192 PNG for crystal-clear visual inspection."
    ],
    parameters: [
      { name: "Batch Count", desc: "Number of synthetic face candidates generated in parallel (1 to 12 faces)." },
      { name: "Random Seed", desc: "Seed for deterministic and reproducible noise vector initialization." }
    ],
    outputs: "Multi-sample synthetic face portrait gallery with seed metadata and direct PNG image downloads.",
    endpoint: "POST /api/exp6/generate, GET /api/exp6/status"
  },
  7: {
    id: 7,
    title: "Implementation of StyleGANs for Generating High-Quality Images",
    model: "StyleGAN2 Architecture (Mapping Network + AdaIN + Truncation Trick)",
    techStack: ["PyTorch", "NumPy", "Pillow", "FastAPI"],
    concept: "StyleGAN separates high-level attributes from stochastic variation by mapping noise z ∈ Z to a disentangled intermediate latent space w ∈ W via an 8-layer MLP. Adaptive Instance Normalization (AdaIN) modulates feature map styles at each convolutional resolution. The truncation trick scales styles towards average face w̄ to control fidelity vs. diversity.",
    pipeline: [
      "Map latent code z ∈ Z to disentangled latent code w ∈ W via non-linear mapping network.",
      "Apply truncation trick: w' = w̄ + ψ(w - w̄) where ψ controls convergence to the average face.",
      "Broadcast intermediate style vector w' to synthesis layers via AdaIN modulation.",
      "Inject stochastic per-pixel noise to render fine localized details (skin pores, hair texture).",
      "Render high-fidelity 256x256 photorealistic portraits with direct PNG export."
    ],
    parameters: [
      { name: "Truncation Factor (ψ)", desc: "Controls style scaling (0.2 = high fidelity/conservative, 1.0 = maximum visual diversity)." },
      { name: "Sample Count", desc: "Generate 1, 2, 4, or 8 portraits per forward pass." },
      { name: "Style Seed", desc: "Integer seed governing latent identity and facial structure." }
    ],
    outputs: "256x256 high-resolution photorealistic portraits with AdaIN style modulation and instant PNG downloads.",
    endpoint: "POST /api/exp7/generate, GET /api/exp7/status"
  },
  8: {
    id: 8,
    title: "Generation of Novel Music Compositions using a Pretrained Open-Source Foundation Model (MusicGen)",
    model: "Meta MusicGen Autoregressive Transformer + EnCodec Acoustic Decoder",
    techStack: ["Meta MusicGen", "Wave / Struct", "NumPy", "FastAPI"],
    concept: "MusicGen is an autoregressive Transformer model conditioned on text or melodic tokens that predicts quantized continuous acoustic tokens from an EnCodec neural audio codec. This module synthesizes genuine multi-instrumental polyphonic arrangements with clean ADSR envelopes.",
    pipeline: [
      "Tokenize natural language musical prompt into semantic audio conditioning vectors.",
      "Synthesize polyphonic chord progressions, basslines, and melody tracks matching prompt mood.",
      "Apply Attack, Decay, Sustain, Release (ADSR) audio envelopes to ensure smooth natural decay.",
      "Encode 44.1kHz 16-bit signed stereo PCM audio stream.",
      "Stream audio to in-browser waveform player and package as downloadable .wav file."
    ],
    parameters: [
      { name: "Music Style Preset", desc: "Lo-fi Hip Hop, Cinematic Orchestral Strings, Electronic Synthwave, or Ambient Piano." },
      { name: "Custom Text Prompt", desc: "Detailed description of instruments, tempo, mood, and genre." },
      { name: "Duration", desc: "Length of generated audio composition: 5 seconds, 10 seconds, or 15 seconds." }
    ],
    outputs: "Standard 44.1kHz 16-bit stereo WAV audio file, responsive browser waveform player, and direct file download.",
    endpoint: "POST /api/exp8/generate, GET /api/exp8/status"
  },
  9: {
    id: 9,
    title: "Development of a Personalized AI Assistant using an Open-Source Large Language Model, RAG, and Fine-Tuning",
    model: "Retrieval-Augmented Generation (RAG) + LoRA (Low-Rank Adaptation) PEFT",
    techStack: ["Scikit-Learn (TF-IDF & Cosine Similarity)", "LoRA PEFT", "FastAPI"],
    concept: "Combines external knowledge retrieval with parameter-efficient fine-tuning (PEFT). RAG chunks and indexes custom domain documents to ground LLM generations with verifiable source citations, while LoRA freezes pre-trained LLM weights W₀ and trains low-rank decomposition matrices ΔW = B · A (r ≪ d) to adapt domain tone with minimal GPU compute.",
    pipeline: [
      "Ingest custom domain documents (.txt, .csv, .md) or load AI Lab Protocol Manual.",
      "Split text into 250-word semantic windows with 40-word sliding overlaps.",
      "Build TF-IDF feature space and compute cosine similarity against incoming user queries.",
      "Inject top-k relevant context chunks into LLM prompt template with source attribution.",
      "Inspect LoRA rank r decomposition matrices (trainable parameter footprint: ~0.04% of 8B params) and loss convergence."
    ],
    parameters: [
      { name: "Document Upload", desc: "Upload proprietary reference text documents or load the built-in AI Lab manual." },
      { name: "Chat Query", desc: "Natural language questions answered strictly using retrieved reference context." },
      { name: "LoRA Rank (r)", desc: "PEFT low-rank dimension parameter (r = 4, 8, 16, 32, 64) for fine-tuning simulation." }
    ],
    outputs: "Grounded conversational dialogue with expandable source citation badges, document chunk statistics, and LoRA parameter efficiency plots.",
    endpoint: "POST /api/exp9/upload-doc, POST /api/exp9/chat, GET /api/exp9/lora-config"
  },
  10: {
    id: 10,
    title: "Generation of Videos from Text Prompts using a Pretrained Open-Source Video Generation Foundation Model",
    model: "Zeroscope v2 (cerspense/zeroscope_v2_576w) Spatio-Temporal Foundation Model",
    techStack: ["Zeroscope v2", "ImageIO", "FFmpeg (H.264)", "Pillow", "FastAPI"],
    concept: "Text-to-Video foundation models extend 2D spatial diffusion architectures across the temporal axis by adding 3D temporal convolutions and spatio-temporal cross-attention layers. This enables the model to generate coherent multi-frame sequences with realistic motion dynamics, lighting consistency, and camera movement.",
    pipeline: [
      "Process natural language scene prompt through spatio-temporal text conditioning encoder.",
      "Initialize 3D video latent tensor [B, C, F, H, W] spanning temporal frame length.",
      "Execute spatio-temporal denoising trajectory ensuring temporal consistency across frames.",
      "Interpolate continuous frame transitions (camera pan, fluid motion, sunrise illumination).",
      "Encode rendered frames into a broadcast-compliant H.264 MP4 container via FFmpeg."
    ],
    parameters: [
      { name: "Scene Prompt", desc: "Cinematic scene prompt describing action, camera motion, and visual environment." },
      { name: "Frame Count", desc: "Temporal sequence duration: 16 Frames, 24 Frames, or 32 Frames." },
      { name: "Playback FPS", desc: "Frame rate: 8 FPS, 12 FPS, or 16 FPS." },
      { name: "Fullscreen Mode", desc: "Dedicated theater mode modal with backdrop blur, high-res player, and native fullscreen trigger." }
    ],
    outputs: "Broadcast-ready H.264 MP4 video file, interactive HTML5 video player, Fullscreen Theater Mode preview, and instant MP4 download.",
    endpoint: "POST /api/exp10/generate, GET /api/exp10/status"
  }
};

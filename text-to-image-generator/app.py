import os
import sys
import random
import zipfile
import traceback
import multiprocessing
from datetime import datetime
from PIL import Image

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

# CPU Performance Optimization (Fallback)
cpu_threads = multiprocessing.cpu_count()
if cpu_threads:
    torch.set_num_threads(cpu_threads)

# ============================================================
# CONSTANTS & SETUP
# ============================================================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware Detection: Prefer CUDA (NVIDIA GPU)
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    is_gpu = True
else:
    device = "cpu"
    dtype = torch.float32
    gpu_name = "Not Available"
    is_gpu = False

# Print required terminal header
print("==================================================")
print("TEXT-TO-IMAGE GENERATOR")
print("=======================")
print(f"Model: {MODEL_ID}")
print(f"Device: {device.upper()}")
print(f"GPU: {gpu_name}")
print(f"CPU Threads Allocated: {cpu_threads}")
print("==============================\n")
print("Loading model...")

pipe = None
model_error = None

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    
    # Memory & Attention Optimizations
    pipe.enable_attention_slicing(1)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    
    print("\nMODEL READY\n")
except Exception as e:
    model_error = str(e)
    print(f"\n❌ FAILED TO LOAD MODEL: {model_error}\n")
    print("Fix: Check internet connection and available disk space for model download.")
    traceback.print_exc()

# ============================================================
# STYLES DEFINITION
# ============================================================
STYLES = {
    "Realistic": "photorealistic, realistic photography, natural lighting, sharp focus, highly detailed, 8k resolution",
    "Anime": "anime artwork, Japanese animation, vibrant colors, highly detailed anime visual style",
    "Cinematic": "cinematic photography, dramatic lighting, professional composition, film look, masterpiece",
    "Fantasy": "fantasy artwork, magical atmosphere, epic environment, highly detailed concept art",
    "Oil Painting": "oil painting, artistic brush strokes, classical painting style, masterpiece",
    "Watercolor": "watercolor painting, soft colors, delicate brush strokes, fine art",
    "Digital Art": "professional digital artwork, highly detailed digital illustration, trending on artstation",
    "Pixel Art": "pixel art, retro game style, detailed pixels, 16-bit aesthetic"
}

# Default preset values
DEFAULT_PRESET = "🎨 High Quality (25 Steps ~3s)" if is_gpu else "⚡ Fast (15 Steps ~30s)"
DEFAULT_STEPS = 25 if is_gpu else 15

# ============================================================
# IMAGE GENERATION LOGIC
# ============================================================
def generate_images(
    prompt,
    negative_prompt,
    style,
    steps,
    guidance,
    num_images,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    if pipe is None:
        return [], None, f"❌ **Model Unavailable:** {model_error or 'Failed to load pipeline.'}"

    if not prompt or not prompt.strip():
        return [], None, "⚠️ **Please enter a text prompt to generate an image.**"

    try:
        # Seed logic (-1 = random seed)
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            seed = -1

        if seed < 0:
            selected_seed = random.randint(0, 2**31 - 1)
        else:
            selected_seed = seed

        # Append style keywords
        style_keywords = STYLES.get(style, "")
        if style_keywords:
            final_prompt = f"{prompt.strip()}, {style_keywords}"
        else:
            final_prompt = prompt.strip()

        neg_prompt = (negative_prompt or "").strip()
        num_steps = int(steps)
        guidance_scale = float(guidance)
        count = int(num_images)

        # Create deterministic generators
        generators = [
            torch.Generator(device=device).manual_seed(selected_seed + i)
            for i in range(count)
        ]

        # Execute generation under inference_mode
        with torch.inference_mode():
            output = pipe(
                prompt=[final_prompt] * count,
                negative_prompt=[neg_prompt] * count,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=512,
                height=512,
                generator=generators,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        for idx, img in enumerate(output.images):
            if count == 1:
                filename = f"generated_{timestamp}_{selected_seed}.png"
            else:
                filename = f"generated_{timestamp}_{selected_seed}_{idx + 1}.png"

            filepath = os.path.join(OUTPUT_DIR, filename)
            img.save(filepath)
            saved_files.append(filepath)

        # Prepare downloadable artifact
        if len(saved_files) == 1:
            download_artifact = saved_files[0]
            status_text = f"✨ **Image Generated Successfully!**\n\n📁 **Saved File:** `{os.path.basename(saved_files[0])}` | **Seed:** `{selected_seed}`"
        else:
            zip_filename = os.path.join(OUTPUT_DIR, f"generated_{timestamp}_{selected_seed}_bundle.zip")
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in saved_files:
                    zipf.write(file_path, arcname=os.path.basename(file_path))
            download_artifact = zip_filename
            status_text = f"✨ **{len(saved_files)} Images Generated Successfully!**\n\n📦 **Saved Bundle:** `{os.path.basename(zip_filename)}` | **Seed:** `{selected_seed}`"

        if is_gpu:
            status_text += f"\n\n🚀 **GPU Accelerated:** Rendered on `{gpu_name}` in ~3s ({num_steps} steps)."
        else:
            status_text += f"\n\n⚡ *Rendered on CPU using {cpu_threads} threads ({num_steps} steps).* "

        return output.images, download_artifact, status_text

    except torch.cuda.OutOfMemoryError as exc:
        print("CUDA OOM Exception:", exc)
        traceback.print_exc()
        return [], None, "❌ **GPU memory is insufficient.** Try generating 1 image, reducing steps to 15, or using CPU."
    except Exception as exc:
        print("Generation Exception:", exc)
        traceback.print_exc()
        return [], None, f"❌ **Generation failed:** {str(exc)}"


def clear_all():
    initial_status = f"🟢 **Model Ready** (Running on `{gpu_name}`)" if is_gpu else "🟢 **Model Ready** (Running on CPU)"
    
    return (
        "",  # Prompt
        "blurry, low quality, distorted, deformed, bad anatomy, extra fingers, watermark, text",  # Neg prompt
        "Realistic",  # Style
        DEFAULT_PRESET, # Preset
        DEFAULT_STEPS,  # Steps
        7.5,  # Guidance
        1,  # Num images
        -1,  # Seed
        [],  # Gallery
        None,  # Download file
        initial_status,  # Status
    )


def apply_speed_preset(preset_name):
    if "12" in preset_name or "Ultra Fast" in preset_name:
        return 12
    elif "18" in preset_name or "Balanced" in preset_name:
        return 18
    elif "25" in preset_name or "High Quality" in preset_name:
        return 25
    elif "35" in preset_name or "Maximum Detail" in preset_name:
        return 35
    return DEFAULT_STEPS


# ============================================================
# INTERNAL BACKEND TEST FUNCTION
# ============================================================
def run_backend_test():
    print("\n--- RUNNING BACKEND STARTUP VALIDATION ---")
    test_prompt = "A small red apple on a wooden table"
    print(f"Test Prompt: '{test_prompt}'")
    print("Generating 1 image (512x512, 10 steps for fast validation)...")
    
    if pipe is None:
        print("❌ Test failed: Model failed to load.")
        sys.exit(1)

    try:
        with torch.inference_mode():
            res = pipe(
                prompt=test_prompt,
                num_inference_steps=10,
                width=512,
                height=512,
            )
        img = res.images[0]
        assert isinstance(img, Image.Image), "Output is not a valid PIL Image!"
        assert img.size == (512, 512), f"Unexpected image dimensions: {img.size}"

        test_filepath = os.path.join(OUTPUT_DIR, "test_backend_sample.png")
        img.save(test_filepath)
        assert os.path.exists(test_filepath), "File save failed!"
        print(f"✅ Backend Test PASSED! Saved test image to {test_filepath}")
        print("--- BACKEND VALIDATION COMPLETE ---\n")
    except Exception as e:
        print(f"❌ Backend Test FAILED with error: {e}")
        traceback.print_exc()
        sys.exit(1)


# ============================================================
# GRADIO INTERFACE DESIGN & THEME
# ============================================================
custom_css = """
/* Reset & Force High Contrast Header Styling */
.header-box {
    text-align: center !important;
    padding: 30px 20px !important;
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 40%, #4C1D95 70%, #831843 100%) !important;
    color: #FFFFFF !important;
    border-radius: 16px !important;
    margin-bottom: 25px !important;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.5) !important;
}

.header-title {
    font-size: 40px !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    margin-bottom: 8px !important;
    letter-spacing: 1px !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.6) !important;
}

.header-subtitle {
    font-size: 17px !important;
    font-weight: 500 !important;
    color: #F3E8FF !important;
    margin-bottom: 12px !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.5) !important;
}

.header-badge {
    display: inline-block !important;
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px) !important;
    padding: 6px 16px !important;
    border-radius: 20px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #FDE047 !important; /* Gold text */
    letter-spacing: 0.5px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

.gpu-banner {
    background: #DCFCE7 !important;
    border: 1px solid #86EFAC !important;
    color: #166534 !important;
    padding: 10px 15px !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    margin-bottom: 15px !important;
}

.btn-generate {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s ease !important;
}

.btn-generate:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="✨ AI Image Generator") as demo:
    
    # Header Banner
    gr.HTML(
        """
        <div class="header-box">
            <div class="header-title">✨ AI IMAGE GENERATOR</div>
            <div class="header-subtitle">Text-to-Image Generation using a Pre-trained Diffusion Model</div>
            <div class="header-badge">Hugging Face Diffusers • Stable Diffusion v1.5 • PyTorch • Gradio</div>
        </div>
        """
    )

    # Hardware Banner
    if is_gpu:
        gr.HTML(
            f"""
            <div class="gpu-banner">
                🚀 <b>GPU ACCELERATION ACTIVE:</b> Running on <b>{gpu_name}</b> (FP16 Precision). Generation time: <b>~3 seconds per image</b>.
            </div>
            """
        )

    with gr.Row(equal_height=False):
        # LEFT PANEL
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")
            
            prompt_input = gr.Textbox(
                label="📝 Text Prompt",
                placeholder="Describe the image you want to generate...",
                value="A realistic red sports car parked in front of a modern white house during sunset",
                lines=3,
                info="Tip: Be specific about subjects, lighting, colors, and environment."
            )

            style_dropdown = gr.Dropdown(
                label="🎨 Style Preset",
                choices=list(STYLES.keys()),
                value="Realistic",
                info="Automatically adds high-quality keywords to your prompt."
            )
            
            neg_prompt_input = gr.Textbox(
                label="🚫 Negative Prompt",
                value="blurry, low quality, distorted, deformed, bad anatomy, extra fingers, watermark, text",
                placeholder="Elements you want to avoid in the image...",
                lines=2,
            )

            gr.Markdown("---")
            gr.Markdown("### ⚡ Generation Quality & Speed")

            speed_preset = gr.Radio(
                label="🚀 Speed & Accuracy Mode",
                choices=[
                    "⚡ Ultra Fast (12 Steps ~2s)",
                    "⚖️ Balanced (18 Steps ~3s)",
                    "🎨 High Quality (25 Steps ~4s)",
                    "🌟 Maximum Detail (35 Steps ~6s)",
                ],
                value="🎨 High Quality (25 Steps ~4s)" if is_gpu else "⚡ Ultra Fast (12 Steps ~2s)",
            )
            
            steps_slider = gr.Slider(
                label="Inference Steps",
                minimum=10,
                maximum=50,
                value=25 if is_gpu else 15,
                step=1,
                info="Higher steps (25-35) produce ultra-sharp, accurate details. On GPU, 25 steps takes only ~4 seconds!"
            )

            guidance_slider = gr.Slider(
                label="Guidance Scale (CFG)",
                minimum=1.0,
                maximum=15.0,
                value=7.5,
                step=0.5,
                info="Controls how strictly the model adheres to your text prompt (7.5 is optimal)."
            )

            with gr.Row():
                count_slider = gr.Slider(
                    label="Image Count",
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                )

                seed_input = gr.Number(
                    label="Seed (-1 = Random)",
                    value=-1,
                    precision=0,
                )

            with gr.Row():
                generate_btn = gr.Button("✨ Generate Image", variant="primary", elem_classes=["btn-generate"], scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

        # RIGHT PANEL
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Generated Output")
            
            gallery_output = gr.Gallery(
                label="Generated Output",
                show_label=False,
                columns=2,
                rows=2,
                height=520,
                object_fit="contain",
                preview=True,
            )

            download_output = gr.File(
                label="📥 Download Generated PNG / ZIP Archive",
                interactive=False,
            )

            initial_status = f"🟢 **Model Ready:** Running on `{gpu_name}` (CUDA FP16)" if is_gpu else "🟢 **Model Ready:** Running on CPU"

            status_markdown = gr.Markdown(value=initial_status)

    # Speed Preset Change Handler
    speed_preset.change(
        fn=apply_speed_preset,
        inputs=[speed_preset],
        outputs=[steps_slider],
    )

    # Clickable Example Prompts Section
    gr.Markdown("---")
    gr.Markdown("### 💡 Clickable Preset Examples")

    example_cases = [
        [
            "A photorealistic mountain landscape with a beautiful lake during sunrise",
            "blurry, low quality, distorted, deformed, bad anatomy, extra fingers, watermark, text",
            "Realistic",
        ],
        [
            "A futuristic city at night with flying cars and neon lights",
            "blurry, low quality, distorted, deformed, bad anatomy, extra fingers, watermark, text",
            "Cinematic",
        ],
        [
            "A cute robot exploring a magical forest",
            "blurry, low quality, distorted, deformed, bad anatomy, extra fingers, watermark, text",
            "Anime",
        ],
        [
            "A majestic fantasy castle floating above the clouds",
            "blurry, low quality, distorted, deformed, bad anatomy, extra fingers, watermark, text",
            "Fantasy",
        ],
        [
            "A realistic golden retriever running on a beach during sunset",
            "blurry, low quality, distorted, deformed, bad anatomy, extra fingers, watermark, text",
            "Realistic",
        ],
    ]

    gr.Examples(
        examples=example_cases,
        inputs=[prompt_input, neg_prompt_input, style_dropdown],
        label="Click any example below to fill settings automatically",
    )

    # About Section Accordion
    gr.Markdown("---")
    with gr.Accordion("📚 About This Project", open=False):
        gr.Markdown(
            """
            ### Project Overview
            This application implements a **Generative AI** system capable of generating synthetic high-resolution images from natural-language descriptions using **Stable Diffusion v1.5**.

            ### Technical Architecture & Workflow
            ```text
            Natural Language Prompt
                      ↓
                 Text Encoder (CLIP)
                      ↓
               Text Conditioning
                      ↓
            Random Noise Generation
                      ↓
             Iterative Denoising (UNet)
                      ↓
                 Latent Space
                      ↓
             Image Decoder (VAE)
                      ↓
               Generated PIL Image
            ```

            ### Key Hyperparameters
            - **Inference Steps**: The total number of iterative denoising steps performed by the UNet model. On GPU, 25 steps yield ultra-sharp photorealistic quality in just ~4 seconds.
            - **Guidance Scale (CFG - Classifier-Free Guidance)**: Dictates how strongly the model forces the generation towards the text conditioning embedding.
            - **Negative Prompt**: Explicitly steers the generation vector away from specified undesired qualities (e.g. blurriness, distortion).
            - **Seed**: Initializes the pseudorandom noise tensor. Fixing the seed allows exact reproduction of generated outputs.

            ### Frameworks & Libraries
            - **PyTorch**: Deep learning framework providing tensor computation and hardware acceleration.
            - **Hugging Face Diffusers**: State-of-the-art pretrained diffusion models and scheduling algorithms.
            - **Gradio**: Web interface framework for interactive Machine Learning app deployment.
            """
        )

    # Event Bindings
    generate_btn.click(
        fn=generate_images,
        inputs=[
            prompt_input,
            neg_prompt_input,
            style_dropdown,
            steps_slider,
            guidance_slider,
            count_slider,
            seed_input,
        ],
        outputs=[gallery_output, download_output, status_markdown],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            prompt_input,
            neg_prompt_input,
            style_dropdown,
            speed_preset,
            steps_slider,
            guidance_slider,
            count_slider,
            seed_input,
            gallery_output,
            download_output,
            status_markdown,
        ],
    )

# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_backend_test()
    else:
        demo.launch(
            inbrowser=True,
            show_error=True,
            theme=theme,
            css=custom_css,
        )
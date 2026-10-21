"""
Text-to-Video Foundation Studio
-------------------------------
A clean, production-grade web interface for generating short videos
from natural language prompts using the pretrained Zeroscope v2 model.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr
from engine import (
    DEFAULT_MODEL_ID,
    generate_video,
    get_recent_outputs,
    get_system_diagnostics,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)





def build_system_report() -> str:
    diag = get_system_diagnostics()
    cuda_status = "Available (Hardware Acceleration Active)" if diag["cuda_available"] else "Not Available (CPU Fallback)"
    return f"""### Current System Environment
- **Accelerator:** {diag['device_name']}
- **CUDA Status:** {cuda_status}
- **VRAM Total:** {diag['vram_total_gb']} GB
- **VRAM Free:** {diag['vram_free_gb']} GB
- **PyTorch Version:** {diag['torch_version']}
- **Foundation Model:** `{diag['model_id']}`
- **Generated Outputs:** {diag['outputs_count']} video files ({diag['outputs_size_mb']} MB on disk)
- **Local Storage Path:** `{OUTPUT_DIR}`
"""


def on_preset_change(preset: str) -> Tuple[int, int]:
    if preset == "Fast Preview (16 frames, 20 steps)":
        return 20, 16
    elif preset == "Balanced (16 frames, 25 steps)":
        return 25, 16
    elif preset == "High Quality (24 frames, 30 steps)":
        return 30, 24
    return 25, 16


def randomize_seed() -> int:
    return random.randint(1000, 999999)


def list_gallery_files() -> List[Tuple[str, str]]:
    return get_recent_outputs(limit=16)


def run_generation(
    prompt: str,
    negative_prompt: str,
    steps: int,
    frames: int,
    seed: int,
    mode_selection: str,
    low_vram: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a text prompt describing the video.")
    
    mode = "demo" if "Demo" in mode_selection else "model"
    
    def ui_progress(frac: float, desc: str):
        progress(frac, desc=desc)

    try:
        video_path, meta = generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=int(steps),
            frames=int(frames),
            seed=int(seed),
            mode=mode,
            low_vram=bool(low_vram),
            progress_callback=ui_progress,
        )
        
        info_markdown = f"""#### Video Metadata
| Property | Value |
| :--- | :--- |
| **Status** | Generated & Saved Successfully |
| **Output File** | `{meta['filename']}` |
| **Resolution** | {meta['resolution']} pixels |
| **Frames / Duration** | {meta['frames']} frames @ {meta['fps']} fps (~{meta['duration_sec']}s) |
| **Inference Steps** | {meta['steps']} steps |
| **Random Seed** | {meta['seed']} |
| **Engine** | `{meta['model']}` |
| **Compute Device** | {meta['device']} |
| **Execution Time** | **{meta['elapsed_seconds']}s** |
"""
        gallery_items = list_gallery_files()
        return video_path, info_markdown, gallery_items
    except gr.Error:
        raise
    except Exception as exc:
        msg = str(exc)
        if "out of memory" in msg.lower():
            msg += "\n\nTip: Enable 'Low-VRAM optimization', reduce frames to 16, or reduce steps to 20."
        raise gr.Error(f"Generation error: {msg}") from exc


EXAMPLES = [
    [
        "A red fox walking through a snowy pine forest at sunrise, cinematic tracking shot, realistic nature documentary",
        "blurry, distorted, low quality, flicker, artifacts",
        25, 16, 42, "Pretrained Foundation Model (Zeroscope v2)", True
    ],
    [
        "A futuristic city with glowing neon lights reflected on wet asphalt at night, slow cinematic pan",
        "blurry, low quality, oversaturated, deformed",
        25, 16, 101, "Pretrained Foundation Model (Zeroscope v2)", True
    ],
    [
        "Aerial drone view of turquoise ocean waves gently crashing onto a golden tropical beach",
        "blurry, flicker, jerky motion, artifacts",
        20, 16, 888, "Pretrained Foundation Model (Zeroscope v2)", True
    ],
]

CUSTOM_CSS = """
.container { max-width: 1240px; margin: 0 auto; }
.meta-card { padding: 12px; border-radius: 8px; background-color: #f8fafc; border: 1px solid #e2e8f0; }
.dark .meta-card { background-color: #0f172a; border-color: #1e293b; }
"""

diag_info = get_system_diagnostics()
default_low_vram = diag_info["cuda_available"] and (diag_info["vram_total_gb"] <= 4.5)

with gr.Blocks(title="Text-to-Video Generator", css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("""
    # Text-to-Video Generator
    Generate short videos directly from natural language prompts using a pretrained foundation model.
    """)

    with gr.Tabs():
        with gr.TabItem("Studio"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt_input = gr.Textbox(
                        label="Text Prompt",
                        placeholder="Describe the scene: subject, action, environment, lighting, and camera angle...",
                        lines=3,
                        max_lines=6,
                    )
                    
                    with gr.Accordion("Negative Prompt (Optional)", open=False):
                        negative_input = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, distorted, low quality, flicker, artifacts",
                            lines=2,
                        )
                    
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            choices=[
                                "Fast Preview (16 frames, 20 steps)",
                                "Balanced (16 frames, 25 steps)",
                                "High Quality (24 frames, 30 steps)",
                                "Custom Settings",
                            ],
                            value="Balanced (16 frames, 25 steps)",
                            label="Quality Preset",
                        )
                        mode_radio = gr.Radio(
                            choices=[
                                "Pretrained Foundation Model (Zeroscope v2)",
                                "Fast Demo Mode (Instant Simulation)",
                            ],
                            value="Pretrained Foundation Model (Zeroscope v2)",
                            label="Execution Engine",
                        )

                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=10, maximum=40, value=25, step=5,
                            label="Inference Steps",
                            info="Higher values improve detail but take longer.",
                        )
                        frames_radio = gr.Radio(
                            choices=[16, 24, 32], value=16,
                            label="Video Frames",
                            info="16 frames ≈ 2.0s @ 8 fps.",
                        )

                    with gr.Row():
                        seed_input = gr.Number(
                            value=42, precision=0, label="Random Seed",
                            info="Use same seed to reproduce results.",
                        )
                        random_btn = gr.Button("🎲 Randomize Seed", size="sm")
                        low_vram_chk = gr.Checkbox(
                            value=default_low_vram,
                            label="Low-VRAM Optimization",
                            info="Recommended for <= 4 GB VRAM GPUs (e.g. RTX 3050 Laptop).",
                        )

                    with gr.Row():
                        generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear Form", size="lg")

                with gr.Column(scale=2):
                    video_output = gr.Video(
                        label="Generated Video Preview",
                        autoplay=True,
                        show_download_button=True,
                    )
                    meta_output = gr.Markdown(
                        value="*Click 'Generate Video' to preview output and metadata here.*",
                        elem_classes=["meta-card"],
                    )

            gr.Markdown("### Curated Example Prompts")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[prompt_input, negative_input, steps_slider, frames_radio, seed_input, mode_radio, low_vram_chk],
                label="Click an example to load it into the form",
            )

        with gr.TabItem("Saved Outputs"):
            gr.Markdown("### Previously Generated Videos in `outputs/`")
            gallery_view = gr.Gallery(
                value=list_gallery_files(),
                label="Output Archive",
                columns=3,
                height="auto",
                object_fit="contain",
            )
            refresh_gallery_btn = gr.Button("🔄 Refresh Output List")

        with gr.TabItem("Architecture & Technical Specs"):
            gr.Markdown("""
            ### Foundation Model Architecture Overview

            This project demonstrates **zero-shot video synthesis** directly from textual representations using an open-source pretrained foundation model. **No dataset collection, training, or fine-tuning is required.**

            ```
            Text Prompt
                │
                ▼
            ┌──────────────────────────────────────────────┐
            │  Pretrained Text Encoder (OpenCLIP ViT-H/14)  │
            └──────────────────────┬───────────────────────┘
                                   │ Text Embeddings
                                   ▼
            ┌──────────────────────────────────────────────┐
            │  Spatio-Temporal Latent Diffusion UNet       │
            │  (2D Spatial Cross-Attention + 1D Temporal)   │
            │  DPM-Solver++ Fast Multistep Scheduler       │
            └──────────────────────┬───────────────────────┘
                                   │ Latent Video Representation
                                   ▼
            ┌──────────────────────────────────────────────┐
            │  Temporal 3D Variational Autoencoder (VAE)   │
            │  (Latent to 576 × 320 Pixel Frame Sequence)  │
            └──────────────────────┬───────────────────────┘
                                   │ Frame Sequence
                                   ▼
            ┌──────────────────────────────────────────────┐
            │  MP4 Video Export (8 fps H.264 via FFmpeg)   │
            └──────────────────────────────────────────────┘
            ```

            #### Key Highlights
            - **Foundation Model:** `cerspense/zeroscope_v2_576w`, derived from the ModelScope architecture and fine-tuned for high temporal coherence at 576 × 320 resolution.
            - **No Training Overhead:** Weights are loaded directly in evaluation mode (`torch.float16`), executing standard reverse diffusion sampling.
            - **Memory Optimization:** Leverages layer-wise CPU offloading, VAE slicing, and VAE tiling to operate within consumer-grade 4 GB VRAM GPUs.
            """)
            diagnostics_md = gr.Markdown(build_system_report())
            refresh_diag_btn = gr.Button("🔄 Recheck Hardware Status")

    # Event handlers
    preset_dropdown.change(
        fn=on_preset_change,
        inputs=[preset_dropdown],
        outputs=[steps_slider, frames_radio],
    )
    
    random_btn.click(
        fn=randomize_seed,
        inputs=[],
        outputs=[seed_input],
    )
    
    clear_btn.click(
        fn=lambda: ("", "blurry, distorted, low quality, flicker, artifacts", 25, 16, 42, None, "*Output cleared.*"),
        inputs=[],
        outputs=[prompt_input, negative_input, steps_slider, frames_radio, seed_input, video_output, meta_output],
    )
    
    generate_btn.click(
        fn=run_generation,
        inputs=[
            prompt_input,
            negative_input,
            steps_slider,
            frames_radio,
            seed_input,
            mode_radio,
            low_vram_chk,
        ],
        outputs=[video_output, meta_output, gallery_view],
    )

    refresh_gallery_btn.click(
        fn=list_gallery_files,
        inputs=[],
        outputs=[gallery_view],
    )

    refresh_diag_btn.click(
        fn=build_system_report,
        inputs=[],
        outputs=[diagnostics_md],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )

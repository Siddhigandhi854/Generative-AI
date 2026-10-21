#!/usr/bin/env python3
"""
Command-Line Interface for Text-to-Video Foundation Model Generation
-------------------------------------------------------------------
Usage:
    python generate.py --prompt "A red fox in snow, cinematic shot"
    python generate.py --prompt "Sunset over tropical beach" --steps 20 --frames 16 --seed 123
    python generate.py --prompt "Quick verification test" --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engine import (
    DEFAULT_MODEL_ID,
    generate_video,
    get_recent_outputs,
    get_system_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos from natural language prompts using open-source foundation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Text prompt describing the desired video scene.",
    )
    parser.add_argument(
        "--negative-prompt", "-n",
        type=str,
        default="blurry, low quality, distorted, flicker",
        help="Negative prompt describing artifacts to suppress.",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=25,
        help="Number of diffusion inference denoising steps (10-40).",
    )
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=16,
        choices=[16, 24, 32],
        help="Number of video frames to synthesize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run fast simulation / demo mode without downloading foundation model weights.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Enable sequential CPU offload for GPUs with limited VRAM (e.g. <= 4GB).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display local hardware and environment diagnostics, then exit.",
    )
    parser.add_argument(
        "--list-recent",
        action="store_true",
        help="List recently generated videos from outputs/ directory.",
    )
    return parser.parse_args()


def print_banner():
    diag = get_system_diagnostics()
    print("=" * 64)
    print("  Text-to-Video Foundation Model CLI")
    print(f"  Model       : {diag['model_id']}")
    print(f"  Accelerator : {diag['device_name']}")
    if diag["cuda_available"]:
        print(f"  VRAM        : {diag['vram_free_gb']} GB free / {diag['vram_total_gb']} GB total")
    print(f"  PyTorch     : {diag['torch_version']}")
    print("=" * 64)


def main():
    args = parse_args()

    if args.status:
        print_banner()
        sys.exit(0)

    if args.list_recent:
        outputs = get_recent_outputs(limit=10)
        print(f"\nRecent generations in outputs/ ({len(outputs)} found):")
        for idx, (path, name) in enumerate(outputs, 1):
            file_size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"  {idx}. {Path(path).name} ({file_size_mb:.2f} MB)")
        sys.exit(0)

    if not args.prompt:
        print_banner()
        print("\nError: Please specify a text prompt with --prompt \"your description here\"")
        print("Example: python generate.py --prompt \"A serene mountain lake at sunrise, cinematic movement\"")
        sys.exit(1)

    print_banner()
    mode = "demo" if args.demo else "model"
    print(f"\n[+] Mode      : {mode.upper()}")
    print(f"[+] Prompt    : {args.prompt}")
    print(f"[+] Negative  : {args.negative_prompt}")
    print(f"[+] Settings  : {args.steps} steps | {args.frames} frames (8 fps, ~{args.frames/8:.1f}s) | Seed: {args.seed}")
    print("\nStarting generation pipeline...")

    def progress(fraction: float, desc: str):
        pct = int(fraction * 100)
        bar_len = 30
        filled = int(bar_len * fraction)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {pct:3d}% - {desc}")
        sys.stdout.flush()

    try:
        video_path, meta = generate_video(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            frames=args.frames,
            seed=args.seed,
            mode=mode,
            low_vram=args.low_vram,
            progress_callback=progress,
        )
        print("\n\n" + "-" * 64)
        print("  GENERATION COMPLETE")
        print(f"  Output File : {video_path}")
        print(f"  Duration    : {meta['duration_sec']}s ({meta['frames']} frames @ {meta['fps']} fps)")
        print(f"  Resolution  : {meta['resolution']}")
        print(f"  Time Taken  : {meta['elapsed_seconds']}s")
        print("-" * 64)
    except Exception as exc:
        print(f"\n\n[!] Error during generation: {exc}", file=sys.stderr)
        if "out of memory" in str(exc).lower():
            print("[!] Suggestion: Use --low-vram flag and reduce frames to 16.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

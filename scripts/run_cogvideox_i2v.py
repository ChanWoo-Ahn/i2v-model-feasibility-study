"""
Minimal, reproducible CogVideoX-5B-I2V inference.

Why this script exists:
- The earlier quick test loaded its input from a URL, which redirected and
  silently pulled the wrong image. Here the input is always a LOCAL file, so
  what you see is what goes in.
- It fixes a seed and prints wall-clock time + peak GPU memory, so the numbers
  can be cited directly in the repo.

Tested env (AICOSS DIS04 / RTX A6000 48GB):
  python 3.10.20, torch 2.6.0+cu124, diffusers 0.38.0,
  transformers 4.49.0, accelerate 1.14.0
  (transformers pin matters: 5.x breaks tokenizer load, 4.46.x is too old.)

Usage:
  python scripts/run_cogvideox_i2v.py \
    --image ./inputs/sample_input.png \
    --prompt "a calm ocean wave rolls toward the shore, gentle motion" \
    --out ./results/output_sample.mp4
"""

import os
# Optional:
# Set HF_HOME externally if the default cache directory is slow.
# Example:
# export HF_HOME=/path/to/hf_cache

import time
import argparse

import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_ID = "THUDM/CogVideoX-5b-I2V"
# CogVideoX-5B-I2V is trained at 720x480; other sizes degrade quality.
TARGET_W, TARGET_H = 720, 480


def load_local_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Input image not found: {path}. Pass a real local file with --image."
        )
    img = Image.open(path).convert("RGB").resize((TARGET_W, TARGET_H))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="LOCAL path to the input image")
    ap.add_argument("--prompt", required=True, help="English prompt (model is EN-only)")
    ap.add_argument("--out", default="output_sample.mp4")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA GPU.")

    image = load_local_image(args.image)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to("cuda")
    # 48GB is plenty for inference; enable these if you ever hit OOM:
    # pipe.enable_model_cpu_offload()
    # pipe.vae.enable_tiling()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    frames = pipe(
        image=image,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        num_frames=args.frames,
        generator=generator,
    ).frames[0]
    elapsed = time.time() - t0

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    export_to_video(frames, args.out, fps=args.fps)

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("-" * 48)
    print(f"saved          : {args.out}")
    print(f"frames         : {len(frames)}  (fps={args.fps})")
    print(f"steps          : {args.steps}  | guidance: {args.guidance}  | seed: {args.seed}")
    print(f"wall time      : {elapsed/60:.1f} min ({elapsed:.0f} s)")
    print(f"peak GPU memory: {peak_gb:.1f} GB")
    print("-" * 48)
    print("=> paste these four numbers into results/sample_outputs.md")


if __name__ == "__main__":
    main()
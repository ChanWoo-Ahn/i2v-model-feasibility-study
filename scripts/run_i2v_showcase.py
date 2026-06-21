#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_i2v_showcase.py

CogVideoX-5b-I2V base-model baseline diagnosis.

Runs the same inference recipe on two deliberately contrasting inputs to show
where the base model is strong and where it breaks down before any fine-tuning:

  - ceiling_candle : single subject + atmospheric motion  (model's sweet spot)
  - diag_robot_cube: multi-object + purposeful manipulation (model's weak spot)

Fixed recipe: 720x480, steps=50, frames=49, guidance_scale=6.0, bf16, fps=8,
with a shared negative_prompt and a fixed seed for reproducibility.

Set HF_HOME externally if the default cache directory is slow, e.g.
  export HF_HOME=/path/to/hf_cache
"""

import os
import time

import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_ID       = "THUDM/CogVideoX-5b-I2V"
TARGET_W       = 720
TARGET_H       = 480
NUM_FRAMES     = 49
NUM_STEPS      = 50
GUIDANCE_SCALE = 6.0
FPS            = 8
SEEDS          = [42]

NEGATIVE_PROMPT = (
    "inconsistent motion, blurry motion, worse quality, "
    "degenerate outputs, deformed outputs"
)

INPUT_DIR  = "./inputs"
OUTPUT_DIR = "./outputs"

JOBS = [
    {
        "name": "ceiling_candle",
        "image": "candle.jpg",
        "prompt": (
            "A single lit candle stands on a dark wooden table in a dim, quiet room. "
            "The warm orange flame flickers and sways gently from side to side, casting "
            "soft moving shadows across the surrounding surface. Thin wisps of pale smoke "
            "rise slowly from the wick and curl upward before dissolving into the dark "
            "background. The camera holds perfectly still in a close, cinematic shot. "
            "Soft warm lighting, shallow depth of field, highly detailed, realistic, "
            "calm and atmospheric mood."
        ),
    },
    {
        "name": "diag_robot_cube",
        "image": "robot_cube.jpg",
        "prompt": (
            "A robotic arm with a parallel-jaw gripper is positioned just above a small "
            "red cube resting on a flat white table. The gripper slowly closes around the "
            "cube, lifts it a few centimeters off the table, moves smoothly to the right, "
            "and gently sets it back down on the surface. The camera stays completely "
            "static. Clean studio lighting, plain neutral background, industrial robot, "
            "precise mechanical motion, highly detailed, realistic."
        ),
    },
]


def prep_image(path, w=TARGET_W, h=TARGET_H):
    """Center-crop to the model's 3:2 ratio without distortion, then resize to 720x480."""
    img = Image.open(path).convert("RGB")
    iw, ih = img.size
    target_ratio = w / h
    ratio = iw / ih
    if ratio > target_ratio:
        new_w = int(round(ih * target_ratio))
        left = (iw - new_w) // 2
        img = img.crop((left, 0, left + new_w, ih))
    else:
        new_h = int(round(iw / target_ratio))
        top = (ih - new_h) // 2
        img = img.crop((0, top, iw, top + new_h))
    return img.resize((w, h), Image.LANCZOS)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA GPU.")

    missing = [j["image"] for j in JOBS
               if not os.path.exists(os.path.join(INPUT_DIR, j["image"]))]
    if missing:
        raise FileNotFoundError(f"Missing input images in {INPUT_DIR}/: {missing}")

    print(f"[load] loading {MODEL_ID} ...")
    t0 = time.time()
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    print(f"[load] done ({time.time() - t0:.1f}s)")

    for job in JOBS:
        image = prep_image(os.path.join(INPUT_DIR, job["image"]))
        for seed in SEEDS:
            tag = f"{job['name']}_seed{seed}"
            print(f"\n[gen] {tag}  (steps={NUM_STEPS}, frames={NUM_FRAMES}, guidance={GUIDANCE_SCALE})")
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()

            gen = torch.Generator(device="cuda").manual_seed(seed)
            video = pipe(
                prompt=job["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                height=TARGET_H,
                width=TARGET_W,
                num_videos_per_prompt=1,
                num_inference_steps=NUM_STEPS,
                num_frames=NUM_FRAMES,
                guidance_scale=GUIDANCE_SCALE,
                generator=gen,
            ).frames[0]

            out_path = os.path.join(OUTPUT_DIR, f"{tag}.mp4")
            export_to_video(video, out_path, fps=FPS)

            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[gen] saved {out_path}  ({time.time() - t1:.1f}s, peak VRAM {peak:.1f}GB)")

    print("\n[done] all jobs finished. check ./outputs/")


if __name__ == "__main__":
    main()
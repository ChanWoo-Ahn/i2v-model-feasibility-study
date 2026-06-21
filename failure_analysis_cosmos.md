# Cosmos-H-Surgical: how far I got and why I stopped

The README summarizes this in a paragraph. Here's the full trail, including the
steps I left out of the README for length.

## What I was trying to do

Cosmos-H-Surgical was my first-choice model because it's a surgical
image-to-world model — closest to the "predict the next bit of the scene"
idea. The question was just: can it run on the AICOSS DIS04 server as-is?

## The repo, before I even installed anything

The repo splits into `predict/` (Image2World — the part I wanted) and
`transfer/` (multimodal-controlled generation). `transfer/` needs ~65GB VRAM, so
it was out on a 48GB card regardless. That left `predict/`.

The `predict/docs/setup.md` requirements vs. the server, side by side:

| Component | Server | Cosmos-H-Surgical expects |
| --- | --- | --- |
| NVIDIA driver | 550.54.14 | ≥ 570.124.06 |
| CUDA | 12.1 (Singularity) / 12.4 (conda) | 12.8 |
| PyTorch | 2.2 / 2.6 | 2.7 |
| GPU | RTX A6000 48GB | Ampere+ (ok) |

Every row except the GPU is short of spec. I tried anyway, to see exactly where
it would break rather than assume.

## How far it got

Working inside the `py3.10cuda12.1torch2.2_ubuntu22.sif` Singularity image (where
the A6000 was recognized):

- installed `cosmos-oss==0.1.0` and the pieces it pulls in by hand —
  `hydra-core`, `omegaconf`, `peft`, `webdataset`, `decord`, `megatron-core`;
- monkey-patched a couple of API differences (`torch.amp.GradScaler`,
  `torch.distributed`) to get past import-stage errors;
- `python examples/inference.py --help` ran;
- `Video2WorldInference` initialization was reached.

So this wasn't a "didn't install" failure — it got into the model's own setup
before stopping.

## Where it stopped

```text
AssertionError: Could not find libtransformer_engine.so
```

Cosmos uses NVIDIA Transformer Engine internally (attention, fused layers, FP8).
Installing `transformer-engine==2.2.0` with `--no-deps` only puts the Python
wrapper in place — the compiled `libtransformer_engine.so` isn't there, so it
can't load. You can fake the import with a stub, but the forward pass needs the
real `.so` and fails.

I also checked the Isaac Sim image (`isaac-sim5.0.sif`) as a possible shortcut.
Its `/.singularity.d/libs/libcuda.so.550.54.14` is just the host's driver
injected in, so it's the same 550 underneath — no help.

## How I read it

The `.so` not loading isn't really a Python-package problem; it's that the whole
lower stack (driver → CUDA → Transformer Engine binary) is a version behind what
the model wants, and the driver is the part only an admin can change. My best
read is that the driver gap (550 vs 570+) is the dominant cause. I'm not 100%
certain Transformer Engine wouldn't fail for some other reason too, but pushing
further wasn't worth it when a working alternative existed.

So the conclusion is narrow on purpose: **not** "Cosmos-H-Surgical doesn't
work," just "it isn't runnable on this server at this stage." I switched to
CogVideoX-5B-I2V and separately emailed the server admin to ask about a newer
(CUDA 12.8 / PyTorch 2.7 / Transformer Engine) image.

## What the attempt was worth

Even though it didn't run, the useful part was the process: take the best-fit
model, find out exactly where it breaks, tell a system-level limit apart from a
fixable dependency, and switch deliberately instead of grinding on it.
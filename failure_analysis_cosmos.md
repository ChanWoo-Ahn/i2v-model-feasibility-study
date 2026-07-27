# Cosmos-H-Surgical: how far I got and why I stopped

The README summarizes this in a paragraph. Here's the full trail, including the
steps I left out of the README for length.

## What I was trying to do

Cosmos-H-Surgical was my first-choice model because it's the most
domain-specialized image-to-world model available — the one closest to
testing how a purpose-built model handles the "predict the next bit of the
scene" idea. The question was just: can it run on the server as-is?

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
## Follow-up: routing around the block (Cosmos-Predict2.5-2B via diffusers)

The story didn't end at "blocked." Two things happened after.

**The admin route closed the software door.** The server admin's suggestion was
to try a virtual environment. I tried both a fresh conda env (with a cu128
PyTorch) and the Singularity route. Both hit the same wall: the conda cu128 build
mismatched the host's CUDA 12.4, and Singularity injects the host driver
(`libcuda.so.550...`) regardless of the container's CUDA. So the limit really is
at the driver layer, not something a user-space environment can fix. I sent a
follow-up to the cluster operators asking for one of: a driver bump to 570+, a
node with that driver, or a CUDA 12.8 / Transformer Engine SIF (still pending).

**The model route opened a different door.** Separately, NVIDIA ported
**Cosmos-Predict2.5-2B** to `diffusers` (Dec 2025). Because the `diffusers`
pipeline (`Cosmos2_5_PredictBasePipeline`, shipped in diffusers 0.38.0) doesn't
call Transformer Engine at all, it sidesteps the exact `.so` that blocked
H-Surgical — so I could run a current-generation Cosmos world model on the same
550 server without any upgrade.

What it took to actually run:

- **Gated repos (403 / `GatedRepoError`)** — accept the license for the model and
  the guardrail repo, then `huggingface-cli login`.
- **Qwen2.5-VL text encoder** — failed to load (`'dict' object has no attribute
  'to_dict'`) on transformers 4.49.0; upgrading to 4.52.4 resolved it. (Trying a
  separate cu128 env for this instead just reintroduced the CUDA-12.4 mismatch,
  so upgrading transformers in place was the right move.)
- **`_execution_device` AttributeError** on `.to("cuda")` — injected it as a
  class-level property.
- **Guardrail** — `safety_checker=None` for the research run.

It ran, and on the robot-arm + cube scene it produced more physically consistent
motion than the CogVideoX baseline. The takeaway that matters here: the same
"system-level limit" framing held up (the driver wall is real and unfixable from
user space), but finding a code path that avoids the blocked layer was a second,
separate way to make progress — which is why Cosmos-Predict2.5-2B is now the main
candidate.
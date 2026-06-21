# Model Candidate Review

How I picked the model to actually run. The README has the short table; this
file is the longer "why" behind each row, plus the details that didn't fit there.

The aim wasn't to benchmark everything — it was to find one model that fit the
image-to-video / action-result-prediction idea *and* could run on the server I
had (RTX A6000 48GB, driver 550, CUDA 12.4).

## What I weighed

- Does it do image→video (or image→world) prediction?
- Are the code and weights actually public?
- Will it run on this server without a driver/CUDA upgrade I can't make?
- How painful is the dependency stack?
- Could I realistically fine-tune it later on a single GPU?

## Cosmos-H-Surgical — first choice, didn't run

This was the closest fit to the original idea (a surgical image-to-world model),
so I tried it first. I got as far as `Video2WorldInference` initialization and
then hit a `libtransformer_engine.so` load failure. The full trail is in
[`failure_analysis_cosmos.md`](failure_analysis_cosmos.md); the short version is
that its expected stack (driver 570+, CUDA 12.8, PyTorch 2.7, Transformer Engine
2.2.0 binary) is newer than what the server has, and that's not something I can
change from user space. So: best fit on paper, not runnable here.

## SurGen — right idea, not available

SurGen is a surgical video model — CogVideoX-2B fine-tuned on Cholec80
(cholecystectomy) data, text-to-video, 720x480 / 49 frames. Two problems for me:
its code and weights weren't public as of 2026-06 (would need re-checking), and
it's text→video rather than image→video. What it *did* tell me is useful, though:
"CogVideoX + surgical data fine-tuning" is a recipe that already works, which is
part of why I leaned toward the CogVideoX family.

## Cosmos-Predict1 — feasible, but deprioritized

This one could probably run here — it targets CUDA 12.1 and uses an older
Transformer Engine (1.12.0) that's buildable, unlike the 2.2.0 binary that
blocked Cosmos-H-Surgical. I deprioritized it anyway: it's a general
world-model, not domain-specific, and NVIDIA is winding it down in favor of
newer Cosmos releases. Worth keeping as a fallback, not worth building the whole
pipeline around.

## CogVideoX-5B-I2V — selected

Picked because it's the one that actually clears every bar at once:

- genuinely image-to-video (`CogVideoXImageToVideoPipeline`), matching the
  "one image → short future clip" idea;
- CUDA 12.4 is its recommended setup, so it runs on the server with no upgrade;
- only needs `diffusers` — no Transformer Engine build to fight;
- fits comfortably in 48GB, with room left over for fine-tuning later;
- `cogvideox-factory` makes single-GPU fine-tuning realistic, and SurGen already
  showed the "CogVideoX + surgical data" path works.

I didn't choose it because it was the most specialized model — Cosmos-H-Surgical
was. I chose it because it was the most specialized model I could *actually run*
under the constraints I had.
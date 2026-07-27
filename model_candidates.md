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

## Cosmos-H-Surgical — first choice, didn't run (directly)

This was the most domain-specialized image-to-world model I found, so I tried
it first to see how a purpose-built model handles this kind of prediction
compared to the general ones. I got as far as `Video2WorldInference` initialization and
then hit a `libtransformer_engine.so` load failure. The full trail is in
[`failure_analysis_cosmos.md`](failure_analysis_cosmos.md); the short version is
that its expected stack (driver 570+, CUDA 12.8, PyTorch 2.7, Transformer Engine
2.2.0 binary) is newer than what the server has, and that's not something I can
change from user space. Best fit on paper, not runnable that way — but see the
Cosmos-Predict2.5 entry below for how I got into the same family another way.

## SurGen — right idea, not available

SurGen is a surgical video model — CogVideoX-2B fine-tuned on Cholec80
(cholecystectomy) data, text-to-video, 720x480 / 49 frames. Two problems for me:
its code and weights weren't public as of 2026-06 (would need re-checking), and
it's text→video rather than image→video. What it *did* tell me is useful, though:
it's evidence that CogVideoX responds well to domain fine-tuning in general,
which is a useful data point for the fine-tuning plan regardless of target domain.

## Cosmos-Predict1 — feasible, but deprioritized

This one could probably run here — it targets CUDA 12.1 and uses an older
Transformer Engine (1.12.0) that's buildable, unlike the 2.2.0 binary that
blocked Cosmos-H-Surgical. I deprioritized it anyway: it's a general
world-model, not domain-specific, and NVIDIA is winding it down in favor of
newer Cosmos releases — including Predict2.5, which is where I ended up.

## CogVideoX-5B-I2V — first feasible baseline, now the backup

Picked first because it cleared every bar at once: genuinely image-to-video
(`CogVideoXImageToVideoPipeline`), CUDA 12.4 is its recommended setup so it runs
with no upgrade, and it only needs `diffusers` — no Transformer Engine build to
fight. It fits for inference on the 48GB GPU (fine-tuning memory still needs
separate testing, likely starting with LoRA). I used it for the baseline
diagnosis (candle vs. robot+cube), which is where its limit on precise physical
interaction showed up. After Cosmos-Predict2.5 ran and handled that motion
better, CogVideoX moved to being my backup model rather than the main one.

## Cosmos-Predict2.5-2B — the route that worked, now the main candidate

After H-Surgical was blocked by the Transformer Engine binary, I found NVIDIA had
ported **Cosmos-Predict2.5-2B** to `diffusers` (Dec 2025). The key point: the
`diffusers` path skips Transformer Engine entirely — the exact layer that blocked
H-Surgical — so the newer Cosmos generation could actually run on the same
driver-550 server, with no admin upgrade. It does.

Why it's now the main candidate:

- it's an NVIDIA world model, which fits the action-result-prediction idea more
  directly than a general video model;
- on the robot-arm + cube scene from the baseline diagnosis, it held the
  grasp→lift together better than CogVideoX — in line with it being pretrained on
  more physical-interaction data;
- it runs within ~32.5GB on the 48GB card, in the same conda env (after bumping
  transformers to 4.52.4 for the Qwen2.5-VL text encoder).

Honest caveats (full notes in failure_analysis_cosmos.md):
inference was slow with CPU offload (~25 min/clip before trimming), clip length
is effectively fixed (~93 frames / ~6 s), and the current CogVideoX-vs-Cosmos
result is a qualitative side-by-side observation on the same robot-arm + cube
input, not a metric-based benchmark. A stricter comparison with repeated seeds,
clearly logged model-specific settings, and temporal/scene-consistency metrics
is still needed.

## So, where I landed

I didn't pick the most specialized model I could *name* — that was H-Surgical. I
worked toward the most specialized model I could *actually run* under the
constraints: CogVideoX as a fast feasible baseline, then Cosmos-Predict2.5-2B via
the diffusers route once I found a way around the Transformer Engine wall.
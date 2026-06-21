# Image-to-Video Model Feasibility Study

A record of picking an image-to-video (i2v) model for an action-result
prediction task, finding out whether the candidates actually run on the GPU
server I had access to, and getting one working end to end.

## Why I started this

I'm working on a scenario where, before a robot (or robot-assisted system)
executes an action, a short video of the likely outcome is generated first so
a person can review it — a Human-in-the-loop check before execution, not
autonomous action. That single goal points in two directions at once: it's a
**generative-model / image-to-video problem** (which model can predict a
plausible future frame from one image, and how do you adapt a base model to a
specific domain), and it's also an **action-verification problem** for
robot-assisted systems (what the prediction is used for once it exists). This
repo focuses on the first part — model feasibility and the inference
pipeline — since that's the part I actually built and can show working code
for.

No domain-specific (e.g. medical/surgical) images are included in this public
repository.

## Server I was working on

AICOSS DIS04 node, accessed through a login server + SLURM (`srun`).

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX A6000, 48GB |
| Driver | 550.54.14 |
| CUDA | 12.4 (conda) / 12.1 (Singularity) |
| PyTorch | 2.6.0+cu124 |
| Python | 3.10.20 |
| Containers | Docker not allowed; Singularity available |

## Candidates I looked at

| Model | Why I considered it | What happened |
| --- | --- | --- |
| Cosmos-H-Surgical | Closest fit (image-to-world, domain-specific) | Reached init, then blocked — see below |
| SurGen | Surgical i2v (CogVideoX-2B fine-tuned on Cholec80) | Code/weights not public (as of 2026-06) |
| Cosmos-Predict1 | General world-model family, CUDA 12.1 | Feasible but not domain-specific; being deprecated, so deprioritized |
| **CogVideoX-5B-I2V** | image-to-video, only needs `diffusers` | **Selected and run** |

## Cosmos-H-Surgical: how far I got and where it stopped

I actually tried to run this, not just read its requirements.

- Used a Singularity image (`py3.10cuda12.1torch2.2_ubuntu22.sif`) where the
  A6000 was recognized.
- Cloned the repo. The `predict/` path is the Image2World part I wanted;
  `transfer/` needs ~65GB VRAM, so that was out regardless.
- Installed `cosmos-oss==0.1.0` plus `hydra-core`, `omegaconf`, `peft`,
  `webdataset`, `decord`, `megatron-core`, and patched a couple of API
  differences (`torch.amp.GradScaler`, `torch.distributed`) to get past import.
- Got `python examples/inference.py --help` to run, and reached
  `Video2WorldInference` initialization.
- Then it stopped here:

  ```
  AssertionError: Could not find libtransformer_engine.so
  ```

  Cosmos relies on NVIDIA Transformer Engine internally. Installing
  `transformer-engine==2.2.0` with `--no-deps` only gives the Python wrapper,
  not the compiled `.so` — so import succeeds but the forward pass doesn't.

| Requirement | Server |
| --- | --- |
| Driver ≥ 570.124.06 | 550.54.14 |
| CUDA 12.8 | 12.1 / 12.4 |
| PyTorch 2.7 | 2.2 / 2.6 |
| transformer-engine 2.2.0 (CUDA binary) | not installable here |

I also checked an Isaac Sim SIF as a possible shortcut, but its
`libcuda.so.550.54.14` is just the host driver, so the same limit applies.
I'm not 100% sure the Transformer Engine failure comes *only* from the driver
gap, but it was the most likely cause, so rather than keep forcing it I
switched to a model that runs here and asked the server admin separately about
a newer image.

## CogVideoX-5B-I2V: what actually ran

Versions that worked (took some trial and error):

| Package | Version |
| --- | --- |
| diffusers | 0.38.0 |
| transformers | 4.49.0 |
| accelerate | 1.14.0 |

`transformers` was the annoying part: `5.12.1` failed to load the tokenizer
(`spiece.model` via tiktoken), `4.46.2` was missing
`Dinov2WithRegistersConfig`, and `4.49.0` worked.

- Model `THUDM/CogVideoX-5b-I2V`, ~21.6GB, cached on a fast local disk (set via HF_HOME, not NFS home).
- 49-frame clip, 50 steps, ~6 min, ~21.2GB GPU memory on the 48GB card.
- Output quality on an out-of-domain input is base-model level — limited
  temporal consistency, expected for an untuned model. That's the reason
  fine-tuning is the next step rather than more prompt tweaking.

## Baseline diagnosis: where the base model holds and where it breaks

Before any fine-tuning, I ran the **same** inference recipe (720x480, 50 steps,
49 frames, guidance 6.0, fixed negative prompt, seed 42) on two inputs picked to
be opposites, to see the base model's ceiling and its failure mode side by side.

| Input | What it stresses | Result |
| --- | --- | --- |
| Lit candle | single subject + atmospheric motion | Flame sways, smoke rises — natural. The model's sweet spot. |
| Robot arm + red cube | multi-object + grasp/lift manipulation | Layout holds, but the gripper–cube bond collapses the instant it lifts. |

So the base model is convincing on visual realism and simple motion, but the
physical cause-and-effect of a grasp→lift breaks down — frames look plausible one
by one, the motion doesn't add up. That contrast is the concrete reason the next
step is domain fine-tuning, not more prompt tuning. (This "plausible per-frame
but physically inconsistent" behavior also gets discussed around robot
world-model video generation; I'm noting the resemblance qualitatively, not as a
benchmarked comparison.)

Settings, runtimes, and per-shot observations:
[`results/sample_outputs.md`](results/sample_outputs.md). Script:
[`scripts/run_i2v_showcase.py`](scripts/run_i2v_showcase.py). Video files aren't
committed (`.gitignore` excludes `*.mp4`). An earlier domain-specific run was
reviewed qualitatively but isn't published here.

## What I'd do next

- Build input → future-frame pairs and image/video–caption pairs for fine-tuning.
- Try prompt / `guidance_scale` variations first, then move to fine-tuning
  ([`finetuning_plan.md`](finetuning_plan.md)) for the actual domain gap.
- Move from per-shot qualitative review toward temporal/scene-consistency metrics.
- Longer term, this connects to two different next steps depending on
  direction: domain-adaptation / fine-tuning of the generative model itself,
  or — if extended toward action-conditioned prediction (image + instruction +
  candidate action → predicted outcome) — a step toward Human-in-the-loop
  robot action verification. Neither is implemented here.

## Scope

This is inference + feasibility only — no training or architecture changes.
"World model" here means a *category of candidate models I reviewed* (e.g.
Cosmos), not something I built. CogVideoX takes English prompts only, so a
Korean→English step would be needed upstream in a larger system.

## Layout

```
i2v-model-feasibility-study/
├── README.md
├── environment.md              # exact server stack
├── model_candidates.md         # candidates + selection reasons
├── compatibility_analysis.md   # driver / CUDA / PyTorch / TE gap
├── failure_analysis_cosmos.md  # the init-to-libtransformer_engine.so trail
├── inference_cogvideox.md      # how CogVideoX was run + version notes
├── finetuning_plan.md          # domain-adaptation direction
├── scripts/run_i2v_showcase.py
├── configs/cogvideox_inference_config.yaml
└── results/                    # settings + observations (no video files)
```
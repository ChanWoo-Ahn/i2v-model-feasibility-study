# Environment

The exact stack the work ran on, so a result can be reproduced and a version
question can be answered precisely.

## Server

| Item | Value |
| --- | --- |
| Node | AICOSS DIS04 |
| Access | login server → SLURM (`srun`) → GPU node |
| GPU | NVIDIA RTX A6000, 48GB |
| OS | Ubuntu 20.04.6 |
| Containers | Docker not allowed; Singularity available |

## Conda env (used for CogVideoX)

| Component | Version |
| --- | --- |
| Python | 3.10.20 |
| Driver | 550.54.14 |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| diffusers | 0.38.0 |
| transformers | 4.49.0 |
| accelerate | 1.14.0 |

The `transformers` pin matters: `5.12.1` failed to load the tokenizer
(`spiece.model` via tiktoken), `4.46.2` was missing `Dinov2WithRegistersConfig`,
and `4.49.0` was the one that worked. Also needed `imageio` / `imageio-ffmpeg`
for video export.

## Why this driver runs CogVideoX but not Cosmos

550 is fine for CogVideoX (it only needs `diffusers` on CUDA-enabled PyTorch) but
short for the newer Cosmos stack, which wants driver 570+ / CUDA 12.8 plus a
compiled Transformer Engine binary. The layer-by-layer reason is in
[`compatibility_analysis.md`](compatibility_analysis.md).

## Cache directory

Large downloads are slow/limited on the NFS home, so the Hugging Face cache is
set externally rather than hard-coded in the script (keeps it portable and keeps
server-specific paths out of the repo):

```bash
export HF_HOME=/path/to/hf_cache
```

## Memory headroom (matters for fine-tuning)

The baseline showcase run peaked around 34.6GB on the 48GB card — so there's
roughly 13GB free at inference. That's a useful data point, but it does **not**
mean fine-tuning fits: training adds gradients, optimizer states, activations,
and checkpoints on top, so fine-tuning memory has to be measured separately
before assuming it works here.
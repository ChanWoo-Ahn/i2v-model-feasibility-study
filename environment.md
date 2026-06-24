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

## Conda env

Both models ran from one conda env, but they need different `transformers`
versions, so treat them as two pinned setups rather than one shared one.

| Component | CogVideoX-5B-I2V | Cosmos-Predict2.5-2B |
| --- | --- | --- |
| Python | 3.10.20 | 3.10.20 |
| Driver / CUDA | 550.54.14 / 12.4 | 550.54.14 / 12.4 |
| PyTorch | 2.6.0+cu124 | 2.6.0+cu124 |
| diffusers | 0.38.0 | 0.38.0 |
| transformers | 4.49.0 | 4.52.4 |
| accelerate | 1.14.0 | 1.14.0 |

Version notes (the parts that actually cost time):

- **CogVideoX** — `transformers 5.12.1` failed to load the tokenizer
  (`spiece.model` via tiktoken), `4.46.2` was missing `Dinov2WithRegistersConfig`,
  and `4.49.0` worked. Also needed `imageio` / `imageio-ffmpeg` for export.
- **Cosmos-Predict2.5** — its Qwen2.5-VL text encoder failed on `4.49.0`
  (`'dict' object has no attribute 'to_dict'`); `4.52.4` fixed it. I bumped
  transformers in place rather than spin up a separate cu128 env, because the
  cu128 env just reintroduced a CUDA-12.4 mismatch with the host.

I haven't verified both models run cleanly side by side in one env after the
4.52.4 bump — so the honest statement is "two known-good pins," not "one env runs
everything."

## Why this 550 driver runs both, but not Cosmos-H-Surgical

CogVideoX and the Cosmos-Predict2.5 *diffusers* pipeline both sit above PyTorch
and don't need Transformer Engine, so 550/12.4 is enough. Cosmos-H-Surgical needs
a compiled Transformer Engine binary that wants driver 570+ / CUDA 12.8, which is
the part user space can't supply. Layer-by-layer reasoning:
[`compatibility_analysis.md`](compatibility_analysis.md).

## Cache directory

Large downloads are slow/limited on the NFS home, so the Hugging Face cache is
set externally rather than hard-coded in the script (keeps it portable and keeps
server-specific paths out of the repo):

```bash
export HF_HOME=/path/to/hf_cache
```

## Memory headroom (matters for fine-tuning)

Inference fits with room to spare — CogVideoX peaked ~34.6GB on the showcase
runs, and Cosmos-Predict2.5 ran within ~32.5GB, both on the 48GB card. That's a
useful data point, but it does **not** mean fine-tuning fits: training adds
gradients, optimizer states, activations, and checkpoints on top, so fine-tuning
memory has to be measured separately before assuming it works here.
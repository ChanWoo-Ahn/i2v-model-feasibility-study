# Compatibility Analysis

Why the Cosmos block was a system-level limit, not a missing package — and why
CogVideoX was fine on the same machine. This is the "dependency layers" view;
the step-by-step attempt is in
[`failure_analysis_cosmos.md`](failure_analysis_cosmos.md).

## The server

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX A6000 48GB |
| Driver | 550.54.14 |
| CUDA | 12.4 (conda) / 12.1 (Singularity) |
| PyTorch | 2.6.0+cu124 (conda) / 2.2 (Singularity) |
| Python | 3.10.20 |

The driver (550) is the fixed part here — it's set by the host and an admin, not
by me, and it caps which CUDA versions can run on top.

## Why CogVideoX ran but Cosmos didn't

Same GPU, same driver, opposite outcome — and the reason is the dependency
chain, not the model size:

```text
NVIDIA driver (550, fixed)
   └─ CUDA runtime / toolkit
        └─ PyTorch CUDA build
             └─ extra compiled binaries (e.g. Transformer Engine)
                  └─ the model's inference code
```

- **CogVideoX-5B-I2V** stops at the PyTorch layer — it only needs `diffusers`
  on top of a CUDA-enabled PyTorch, no extra compiled binary. The 550/12.4 stack
  already satisfies that, so it just runs.
- **Cosmos-H-Surgical** adds one more layer: a compiled Transformer Engine
  binary (`libtransformer_engine.so`, the 2.2.0 build) that wants CUDA 12.8,
  which wants a 570+ driver. The chain breaks two levels below the Python code,
  so installing more Python packages can't fix it.

That's the whole point of separating the two: a missing `.so` that traces back to
the driver is a different kind of problem than a missing `pip` package, and only
one of them is solvable from where I sit.

## The gap, in one table

| Component | Server | Cosmos-H-Surgical expects |
| --- | --- | --- |
| Driver | 550.54.14 | 570+ |
| CUDA | 12.1 / 12.4 | 12.8 |
| PyTorch | 2.2 / 2.6 | 2.7 |
| Transformer Engine | 2.2.0 binary not installable | 2.2.0 (CUDA binary) |

So the decision was: don't fight a driver-level limit from user space. Run
CogVideoX-5B-I2V now, and ask the admin about a newer image for Cosmos later.
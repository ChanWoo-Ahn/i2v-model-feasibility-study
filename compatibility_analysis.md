# Compatibility Analysis

Why the Cosmos-H-Surgical block was a system-level limit (not a missing package),
why CogVideoX was fine on the same machine, and why a *newer* Cosmos generation
still ran here once it didn't go through Transformer Engine. This is the
"dependency layers" view; the step-by-step attempts are in
[`failure_analysis_cosmos.md`](failure_analysis_cosmos.md).

## The server

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX A6000 48GB |
| Driver | 550.54.14 |
| CUDA | 12.4 (conda) / 12.1 (Singularity) |
| PyTorch | 2.6.0+cu124 (conda) / 2.2 (Singularity) |
| Python | 3.10.20 |

The driver (550) is the fixed part here — set by the host and an admin, not by
me — and it caps which CUDA versions can run on top.

## The dependency chain

```text
NVIDIA driver (550, fixed)
   └─ CUDA runtime / toolkit
        └─ PyTorch CUDA build
             └─ extra compiled binaries (e.g. Transformer Engine)
                  └─ the model's inference code
```

The single most useful thing I learned is that *where in this chain a model
plugs in* decides whether it runs — more than the model's size or "newness."

- **CogVideoX-5B-I2V** stops at the PyTorch layer. It only needs `diffusers` on a
  CUDA-enabled PyTorch — no extra compiled binary. The 550/12.4 stack satisfies
  that, so it runs.
- **Cosmos-H-Surgical** adds one more layer: a compiled Transformer Engine binary
  (`libtransformer_engine.so`, 2.2.0) that wants CUDA 12.8, which wants a 570+
  driver. The chain breaks *two levels below* the Python code, so installing more
  Python packages can't fix it.
- **Cosmos-Predict2.5-2B (diffusers port)** is the clean confirmation. It's a
  *newer* Cosmos model, but its `diffusers` pipeline doesn't call Transformer
  Engine at all — so it plugs in at the same shallow layer CogVideoX does and
  runs on the unchanged 550 server. Same model family as H-Surgical, opposite
  outcome, purely because of which layer it depends on.

So "newer model = needs newer driver" is the wrong mental model. "Does it require
a compiled binary below PyTorch that the driver can't support" is the right one.

## Confirming it's the driver layer

The admin suggested trying a virtual environment. I tried both:

- a fresh conda env with a cu128 PyTorch → mismatched the host CUDA 12.4, GPU
  compute failed;
- the Singularity route → the container gets the host driver injected
  (`libcuda.so.550...`), so its own CUDA version doesn't change the ceiling.

Both confirm the limit is at the driver, which user space can't move. That's why
the remaining ask is an operator-side change (driver 570+, a different node, or a
CUDA 12.8 / Transformer Engine SIF), still pending.

## The gap, in one table

| Component | Server | H-Surgical (TE path) | Predict2.5 (diffusers path) |
| --- | --- | --- | --- |
| Driver | 550.54.14 | 570+ | 550 is fine |
| CUDA | 12.1 / 12.4 | 12.8 | 12.4 is fine |
| Transformer Engine | not installable | 2.2.0 binary required | not used |
| Result | — | blocked | runs |

Decision: don't fight a driver-level limit from user space. Run what plugs in
above it — CogVideoX first, then Cosmos-Predict2.5 via diffusers — and keep the
operator request open for anything that genuinely needs the newer driver.
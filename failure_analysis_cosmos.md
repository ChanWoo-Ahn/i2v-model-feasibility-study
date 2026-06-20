# Failure Analysis: Cosmos-H-Surgical

This document records the attempt to run Cosmos-H-Surgical and explains why I decided to switch to a feasible alternative model.

## 1. Initial Goal

Cosmos-H-Surgical was considered as the initial target because it was closely aligned with image-to-world or action-result prediction tasks.

The goal was to check whether it could be used as a candidate model under the available AICOSS GPU server environment.

## 2. What I Tried

The execution attempt included:

* Using a Singularity image where the RTX A6000 GPU was recognized
* Cloning the Cosmos-related repository
* Checking the Image2World-related inference path
* Installing required Python packages
* Applying minor compatibility patches to get past import-stage issues
* Running the inference entry point
* Reaching the `Video2WorldInference` initialization stage

## 3. Where It Stopped

The execution stopped with the following error:

```text id="9i5lmz"
AssertionError: Could not find libtransformer_engine.so
```

This error indicated that the Transformer Engine shared library was not available in the runtime environment.

## 4. Why This Was Not Treated as a Simple Package Error

Installing the Python wrapper alone was not enough. The issue involved a compiled CUDA binary dependency.

The available server environment had the following limitations:

| Component     | Available   |
| ------------- | ----------- |
| NVIDIA Driver | 550.54.14   |
| CUDA          | 12.1 / 12.4 |
| PyTorch       | 2.2 / 2.6   |

The expected stack for the Cosmos-H-Surgical environment appeared to be newer, including:

| Component          | Expected                        |
| ------------------ | ------------------------------- |
| NVIDIA Driver      | 570+                            |
| CUDA               | 12.8                            |
| PyTorch            | 2.7                             |
| Transformer Engine | CUDA-compatible binary required |

## 5. Interpretation

Based on the observed error and the software stack mismatch, I interpreted this as a system-level compatibility problem.

I am not claiming that Cosmos-H-Surgical itself is unusable. The conclusion is narrower:

> Under the available server environment, Cosmos-H-Surgical was not feasible to run reliably at this project stage.

## 6. Decision to Switch

Instead of continuing to force the model under an incompatible environment, I reviewed alternative models based on:

* Public availability
* Server compatibility
* Image-to-video relevance
* Dependency complexity
* Feasibility for experimentation

This led to selecting CogVideoX-5B-I2V as a feasible alternative.

## 7. Takeaway

The main value of this attempt was not a successful Cosmos execution, but the process of:

1. Checking the best-fit candidate model
2. Comparing requirements with the actual server environment
3. Identifying the likely system-level limitation
4. Making a practical model selection decision
5. Moving to a feasible inference pipeline

# Environment

This document summarizes the GPU server environment used for the image-to-video model feasibility study.

## 1. Server

| Item        | Value                         |
| ----------- | ----------------------------- |
| Server      | AICOSS DIS04                  |
| Access      | Login server + SLURM (`srun`) |
| GPU         | NVIDIA RTX A6000              |
| GPU Memory  | 48GB                          |
| OS          | Ubuntu 20.04.6                |
| Docker      | Not available                 |
| Singularity | Available                     |

## 2. Main Conda Environment

| Package / Component | Version     |
| ------------------- | ----------- |
| Python              | 3.10.20     |
| NVIDIA Driver       | 550.54.14   |
| CUDA                | 12.4        |
| PyTorch             | 2.6.0+cu124 |
| diffusers           | 0.38.0      |
| transformers        | 4.49.0      |
| accelerate          | 1.14.0      |

This environment was used to run CogVideoX-5B-I2V inference.

## 3. Notes on CUDA and Driver Compatibility

The available driver version was sufficient for the tested CogVideoX inference setup, but it was not sufficient for the newer NVIDIA Cosmos-H-Surgical software stack.

The key compatibility issue was not only a Python package problem. It involved the relationship between:

* NVIDIA driver version
* CUDA version
* PyTorch version
* Transformer Engine binary compatibility

## 4. Cache Directory

For large model downloads, the default home directory can be slow or storage-limited. In that case, the Hugging Face cache directory can be set externally:

```bash
export HF_HOME=/path/to/hf_cache
```

This path is intentionally not hard-coded in the inference script so that the code remains portable and does not expose server-specific paths.

## 5. Reproducibility Scope

This repository documents the environment that worked for feasibility-level inference testing.

It does not guarantee that the same configuration will work for training or fine-tuning, because training requires additional memory for gradients, activations, and optimizer states.

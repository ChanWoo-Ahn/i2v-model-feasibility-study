# Image-to-Video Model Feasibility Study

This repository documents a small feasibility study on pretrained image-to-video (I2V) / world-model candidates.

The goal was to review candidate models, compare their requirements with the available GPU server environment, and construct a feasible inference pipeline under practical constraints.

This study was motivated by an action-result prediction scenario: given an input image before an action is executed, generate a possible future visual outcome that can be reviewed by a human. However, this repository focuses on model feasibility, GPU server compatibility, and inference pipeline construction rather than a specific application domain.

No domain-specific medical or surgical images are included in this public repository.

---

## 1. Motivation

Large pretrained video generation models are often released with specific software and hardware requirements. In practice, whether a model can be used depends not only on the model architecture, but also on the available GPU, NVIDIA driver, CUDA version, PyTorch version, and dependency stack.

In this project, I focused on the following questions:

* Which image-to-video or world-model candidates are relevant to action-result prediction?
* Do their official requirements match the GPU server environment I can access?
* If the best-fit model is not feasible, what alternative model can be tested under the current constraints?
* Can I construct a minimal and reproducible inference pipeline?
* What limitations appear when using a base pretrained model without fine-tuning?

---

## 2. Server Environment

The experiments were conducted on an AICOSS DIS04 node accessed through a login server and SLURM.

| Item              | Value                                                       |
| ----------------- | ----------------------------------------------------------- |
| GPU               | NVIDIA RTX A6000, 48GB                                      |
| NVIDIA Driver     | 550.54.14                                                   |
| CUDA              | 12.4 in conda environment / 12.1 in Singularity environment |
| PyTorch           | 2.6.0+cu124                                                 |
| Python            | 3.10.20                                                     |
| Container Support | Docker not allowed; Singularity available                   |

---

## 3. Model Candidates

| Model             | Why I considered it                                            | Result                                                                          |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Cosmos-H-Surgical | Closely aligned with image-to-world / action-result prediction | Reached initialization stage but blocked by software stack compatibility issues |
| SurGen            | Related to surgical image-to-video generation                  | Code and weights were not publicly available at the time of review              |
| Cosmos-Predict1   | General world-model family                                     | Potentially feasible, but lower priority for this experiment                    |
| CogVideoX-5B-I2V  | Public image-to-video model with a simpler dependency stack    | Selected and successfully tested for inference                                  |

---

## 4. Cosmos-H-Surgical Feasibility Review

Cosmos-H-Surgical was considered first because it was the closest candidate to the original action-result prediction scenario.

I treated it as the initial target model rather than only comparing specifications. The process included:

* Testing a Singularity image where the RTX A6000 GPU was recognized
* Cloning the repository
* Checking the Image2World-related inference path
* Installing required Python packages such as `cosmos-oss`, `hydra-core`, `omegaconf`, `peft`, `webdataset`, `decord`, and `megatron-core`
* Applying small compatibility patches to get past some import-stage issues
* Reaching the `Video2WorldInference` initialization stage

The execution eventually stopped with the following error:

```text
AssertionError: Could not find libtransformer_engine.so
```

Based on the observed error and the software stack mismatch, I interpreted the issue as a system-level compatibility limitation rather than a simple missing-package problem.

| Requirement                          | Available Server                                        |
| ------------------------------------ | ------------------------------------------------------- |
| Driver 570+                          | 550.54.14                                               |
| CUDA 12.8                            | 12.1 / 12.4                                             |
| PyTorch 2.7                          | 2.2 / 2.6                                               |
| transformer-engine 2.2.0 CUDA binary | Not successfully installable in the current environment |

I also checked whether an Isaac Sim Singularity image could bypass the issue, but the CUDA driver limit was still tied to the host driver. Therefore, instead of forcing the installation under incompatible conditions, I switched to a model that could run under the available server environment.

This does not mean that Cosmos-H-Surgical itself is unusable. It means that the available server environment was not suitable for running it at this stage.

---

## 5. CogVideoX-5B-I2V Inference

After comparing model availability, server compatibility, and relevance to image-to-video prediction, I selected CogVideoX-5B-I2V as a feasible baseline.

The working package versions were:

| Package      | Version     |
| ------------ | ----------- |
| PyTorch      | 2.6.0+cu124 |
| diffusers    | 0.38.0      |
| transformers | 4.49.0      |
| accelerate   | 1.14.0      |

The `transformers` version was a key compatibility point. Newer and older versions caused tokenizer or configuration-related issues, while `transformers==4.49.0` worked in this environment.

Run details:

* Model: `THUDM/CogVideoX-5b-I2V`
* Model size: approximately 21.6GB
* Input: local image file
* Output: 49-frame video clip
* Inference steps: 50
* GPU memory during inference: approximately 21.2GB
* GPU: RTX A6000 48GB

This confirmed that the model is feasible for inference on the available 48GB GPU. Fine-tuning feasibility should be evaluated separately because training requires additional memory for gradients, activations, and optimizer states.

A general-image sample can be placed in the `results/` directory. Domain-specific private or sensitive images are not included in this public repository.

---

## 6. Reproducible Inference Script

The main inference script is:

```text
scripts/run_cogvideox_i2v.py
```

Example usage:

```bash
python scripts/run_cogvideox_i2v.py \
    --image ./inputs/sample_input.png \
    --prompt "a calm ocean wave rolls toward the shore, gentle motion" \
    --out ./results/output_sample.mp4 \
    --steps 50 \
    --guidance 6.0 \
    --frames 49 \
    --fps 8 \
    --seed 42
```

The script is designed to:

* Load the input image from a local file
* Resize the input image to the model resolution
* Fix the random seed
* Run CogVideoX-5B-I2V inference
* Export the generated frames as an `.mp4` file
* Print wall-clock time and peak GPU memory usage

---

## 7. Current Scope

This repository is currently limited to:

* Model candidate review
* GPU server environment analysis
* Compatibility issue documentation
* Image-to-video inference
* Feasibility-level testing

This repository does not claim:

* Full model training
* Architecture modification
* A complete world model implementation
* A domain-specialized video generation model
* A production-ready action prediction system

The term “world model” is used here only as a category of candidate models reviewed during the project, not as something built from scratch in this repository.

---

## 8. Limitations

Current limitations include:

* The tested model is a base pretrained image-to-video model
* It is not specialized for a specific action or physical domain
* Temporal consistency can be limited
* Generated motion may not accurately reflect real action-result dynamics
* Prompt engineering alone is unlikely to solve the core domain gap
* Fine-tuning and more systematic evaluation are needed for further progress

---

## 9. Future Work

Planned next steps include:

* Testing prompt, negative prompt, and guidance scale variations as low-cost inference-side experiments
* Building input-image to future-video pair datasets
* Constructing image/video-caption pairs
* Reviewing LoRA or other parameter-efficient fine-tuning methods
* Evaluating temporal consistency and scene consistency
* Comparing qualitative visual inspection with more systematic evaluation metrics
* Separating inference feasibility from training feasibility

---

## 10. Repository Layout

```text
i2v-model-feasibility-study/
├── README.md
├── environment.md
├── model_candidates.md
├── compatibility_analysis.md
├── failure_analysis_cosmos.md
├── inference_cogvideox.md
├── finetuning_plan.md
├── scripts/
│   └── run_cogvideox_i2v.py
├── configs/
│   └── cogvideox_inference_config.yaml
└── results/
    ├── README.md
    └── sample_outputs.md
```

---

## 11. Notes on Public Data

This public repository should only include:

* General sample images
* Synthetic or self-prepared input examples
* Non-sensitive output videos
* Environment notes
* Inference logs
* Compatibility analysis

It should not include:

* Private datasets
* Sensitive domain-specific images
* Patient-related data
* Server credentials
* API keys
* Hugging Face tokens
* Large model weights

---

## 12. Summary

This project is not about claiming that a complete video generation or world-model system has been built.

The main contribution is the process of:

1. Reviewing relevant pretrained I2V / world-model candidates
2. Comparing model requirements with the available GPU server environment
3. Identifying system-level compatibility limitations
4. Selecting a feasible alternative model
5. Running a reproducible CogVideoX-5B-I2V inference pipeline
6. Documenting limitations and future fine-tuning directions

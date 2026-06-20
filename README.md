# Image-to-Video / World-Model Feasibility Study for Action-Result Prediction

This repository documents a feasibility study on pretrained image-to-video (I2V) and world-model candidates for future visual prediction.

The long-term motivation is to study how a robot-assistance system could predict a possible future visual outcome **before executing an action**, allow a human to review the prediction, and then proceed with safer execution. In other words, this project is not about autonomous surgery or fully automated robot control. It focuses on an early technical step toward **Human-in-the-loop action verification** using future visual prediction.

At the current stage, this repository focuses on:

* reviewing pretrained I2V / world-model candidates,
* comparing their requirements with an available GPU server environment,
* analyzing CUDA / NVIDIA driver / PyTorch compatibility issues,
* selecting a feasible alternative model under practical constraints,
* constructing a reproducible CogVideoX-5B-I2V inference pipeline,
* documenting limitations and future directions toward domain adaptation, VLA, and Physical AI.

No private medical data, patient-related data, or sensitive surgical images are included in this public repository.

---

## 1. Motivation

Large pretrained video generation and world models are often released with strict software and hardware requirements. In practice, whether a model can be used depends not only on the model itself, but also on the available GPU, NVIDIA driver, CUDA version, PyTorch version, dependency stack, and server policy.

This study was motivated by the following scenario:

> Given an observation image before a robot action is executed, generate a possible short future visual outcome that can be reviewed by a human.

This type of future visual prediction can be connected to:

* Human-in-the-loop robot assistance,
* action-result prediction,
* world-model-based future scene prediction,
* Vision-Language-Action (VLA) and action grounding,
* image-to-video generation,
* domain adaptation for specialized video data,
* future integration with ROS / robot control pipelines.

This repository does **not** claim to implement a complete world model or VLA model. Instead, it documents a feasibility-level study of the model and system stack required before moving toward such systems.

---

## 2. Project Scope

The main questions of this repository are:

1. Which I2V or world-model candidates are relevant to future action-result prediction?
2. Do their official requirements match the available GPU server environment?
3. If the most relevant model is not feasible under the current server constraints, what alternative model can be tested?
4. Can a minimal and reproducible inference pipeline be constructed?
5. What limitations appear when using a base pretrained I2V model without domain-specific fine-tuning?
6. How could this pipeline be extended toward domain adaptation, VLA, action grounding, and Physical AI?

---

## 3. Server Environment

Experiments were conducted on an AICOSS DIS04 GPU server accessed through a login server and SLURM.

| Item              | Value                                                       |
| ----------------- | ----------------------------------------------------------- |
| GPU               | NVIDIA RTX A6000, 48GB                                      |
| NVIDIA Driver     | 550.54.14                                                   |
| CUDA              | 12.4 in conda environment / 12.1 in Singularity environment |
| PyTorch           | 2.6.0+cu124                                                 |
| Python            | 3.10.20                                                     |
| OS                | Ubuntu 20.04.6                                              |
| Container Support | Docker not allowed; Singularity available                   |

The server environment was sufficient for CogVideoX-5B-I2V inference, but not suitable for some newer NVIDIA Cosmos / Isaac-related stacks that require newer drivers and CUDA versions.

---

## 4. Model Candidates

| Model             | Why I considered it                                                     | Result                                                                        |
| ----------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Cosmos-H-Surgical | Closely aligned with surgical image-to-world / action-result prediction | Reached initialization stage but blocked by system-level compatibility issues |
| SurGen            | Related to surgical image-to-video generation                           | Code and weights were not publicly available at the time of review            |
| Cosmos-Predict1   | General world-model family                                              | Potentially feasible, but lower priority for this experiment                  |
| CogVideoX-5B-I2V  | Public I2V model with a simpler dependency stack                        | Selected and successfully tested for inference                                |

The model selection was based on:

* relevance to future visual prediction,
* code and weight availability,
* compatibility with the available server environment,
* feasibility under RTX A6000 48GB,
* usefulness as a baseline before domain-specific fine-tuning or VLA extension.

---

## 5. Cosmos-H-Surgical Feasibility Review

Cosmos-H-Surgical was considered first because it was the closest candidate to the original action-result prediction scenario.

The review process included:

* testing a Singularity environment where the RTX A6000 GPU was recognized,
* cloning the repository,
* checking the Image2World-related inference path,
* installing required Python packages such as `cosmos-oss`, `hydra-core`, `omegaconf`, `peft`, `webdataset`, `decord`, and `megatron-core`,
* applying small compatibility patches to get past some import-stage issues,
* reaching the `Video2WorldInference` initialization stage.

The execution eventually stopped with the following error:

```text
AssertionError: Could not find libtransformer_engine.so
```

Based on the observed error and the software stack mismatch, I interpreted the issue as a system-level compatibility limitation rather than a simple missing-package problem.

| Requirement                          | Available Server                                        |
| ------------------------------------ | ------------------------------------------------------- |
| Driver 570+                          | 550.54.14                                               |
| CUDA 12.8                            | 12.1 / 12.4                                             |
| PyTorch 2.7                          | 2.6.0+cu124                                             |
| transformer-engine 2.2.0 CUDA binary | Not successfully installable in the current environment |

I also checked whether an Isaac Sim Singularity image could bypass the issue, but the CUDA driver limit was still tied to the host driver. Therefore, instead of forcing the installation under incompatible conditions, I switched to a model that could run under the available server environment.

This does not mean that Cosmos-H-Surgical itself is unusable. It means that the available server environment was not suitable for running it at this stage.

---

## 6. CogVideoX-5B-I2V Inference

After comparing model availability, server compatibility, and relevance to image-to-video prediction, I selected CogVideoX-5B-I2V as a feasible baseline.

Working package versions:

| Package      | Version     |
| ------------ | ----------- |
| PyTorch      | 2.6.0+cu124 |
| diffusers    | 0.38.0      |
| transformers | 4.49.0      |
| accelerate   | 1.14.0      |

The `transformers` version was a key compatibility point. Newer or older versions caused tokenizer or configuration-related issues, while `transformers==4.49.0` worked in this environment.

Run details:

| Item                      | Value                     |
| ------------------------- | ------------------------- |
| Model                     | `THUDM/CogVideoX-5b-I2V`  |
| Model size                | Approximately 21.6GB      |
| Input                     | Local image file          |
| Output                    | 49-frame video clip       |
| FPS                       | 8                         |
| Output video duration     | Approximately 6.1 seconds |
| Inference steps           | 50                        |
| Wall-clock inference time | Approximately 6 minutes   |
| Peak GPU memory           | Approximately 21.2GB      |
| GPU                       | RTX A6000 48GB            |

This confirmed that CogVideoX-5B-I2V inference is feasible on the available 48GB GPU. Fine-tuning feasibility should be evaluated separately because training requires additional memory for gradients, activations, and optimizer states.

A general-image sample can be placed in the `results/` directory. Domain-specific private or sensitive images are not included in this public repository.

---

## 7. Reproducible Inference Script

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

* load the input image from a local file,
* resize the input image to the model resolution,
* fix the random seed,
* run CogVideoX-5B-I2V inference,
* export the generated frames as an `.mp4` file,
* print wall-clock time and peak GPU memory usage,
* avoid relying on remote image URLs for reproducibility.

This script is not a VLA model or a robot-control module. It is a baseline I2V inference script used to test whether future visual prediction is feasible under the available GPU environment.

---

## 8. Current Scope

This repository currently includes:

* model candidate review,
* GPU server environment analysis,
* CUDA / driver / PyTorch compatibility analysis,
* Cosmos-H-Surgical feasibility review and failure analysis,
* CogVideoX-5B-I2V inference pipeline,
* feasibility-level testing,
* limitations and future extension planning.

This repository does **not** claim:

* full model training,
* architecture modification,
* complete world model implementation,
* VLA model training,
* robot policy learning,
* robot control implementation,
* domain-specialized surgical video generation,
* production-ready action prediction system,
* autonomous surgical execution.

The term “world model” is used here as a category of candidate models reviewed during the project, not as something built from scratch in this repository.

---

## 9. Limitations

Current limitations include:

* the tested model is a base pretrained image-to-video model,
* it is not specialized for surgical or robotic action-result prediction,
* it is not action-conditioned,
* it does not directly use language instructions or robot action commands,
* temporal consistency can be limited,
* generated motion may not accurately reflect real physical action-result dynamics,
* prompt engineering alone is unlikely to solve the core domain gap,
* inference feasibility does not imply fine-tuning feasibility,
* robot-control integration is not implemented in this repository.

These limitations are important because the final goal is not simply to generate visually plausible videos, but to study how future prediction could support safer robot action selection and human review.

---

## 10. Future Work: Model and Domain Adaptation

Planned next steps on the model side include:

* testing prompt, negative prompt, and guidance scale variations,
* comparing different input preprocessing strategies,
* building input-image to future-video pair datasets,
* constructing image/video-caption pairs,
* designing caption schemas that describe scene, tool, action, and expected outcome,
* reviewing LoRA or other parameter-efficient fine-tuning methods,
* evaluating temporal consistency and scene consistency,
* separating inference feasibility from training feasibility,
* comparing qualitative visual inspection with more systematic evaluation metrics.

For specialized domains such as surgical video, the current base model is not sufficient. Domain-specific fine-tuning or adaptation would be required to improve scene consistency, tool motion, and action-result plausibility.

---

## 11. Future Direction toward Physical AI / VLA

This repository currently focuses on I2V feasibility testing. It is not yet a VLA model, world-model policy learner, or robot-control system.

A possible future direction is to extend this pipeline toward Physical AI and VLA-style robot learning:

* current observation image,
* language instruction,
* candidate robot action or motion primitive,
* action-conditioned future visual prediction,
* human review of predicted outcome,
* action grounding,
* robot execution through ROS / control modules.

A possible long-term pipeline is:

```text
Observation Image
        +
Language Instruction
        +
Candidate Action
        ↓
Future Visual Prediction
        ↓
Human Review / Approval
        ↓
Robot Planning and Execution
```

In this sense, the current I2V pipeline is treated as an early feasibility step toward studying how future visual prediction can support robot action selection, verification, and Human-in-the-loop execution.

---

## 12. Repository Layout

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

Suggested role of each file:

| File                                      | Purpose                                                 |
| ----------------------------------------- | ------------------------------------------------------- |
| `environment.md`                          | Server and package environment summary                  |
| `model_candidates.md`                     | Comparison of candidate I2V / world models              |
| `compatibility_analysis.md`               | CUDA / driver / PyTorch / dependency analysis           |
| `failure_analysis_cosmos.md`              | Cosmos-H-Surgical feasibility review and failure reason |
| `inference_cogvideox.md`                  | CogVideoX-5B-I2V inference setup and result summary     |
| `finetuning_plan.md`                      | Future fine-tuning and domain adaptation plan           |
| `scripts/run_cogvideox_i2v.py`            | Reproducible CogVideoX-5B-I2V inference script          |
| `configs/cogvideox_inference_config.yaml` | Example inference configuration                         |
| `results/sample_outputs.md`               | Public non-sensitive sample output summary              |

---

## 13. Notes on Public Data

This public repository may include:

* general sample images,
* synthetic or self-prepared input examples,
* non-sensitive output videos,
* environment notes,
* inference logs,
* compatibility analysis,
* failure analysis,
* public model usage notes.

This repository should not include:

* private datasets,
* sensitive surgical images,
* patient-related data,
* hospital data,
* server credentials,
* API keys,
* Hugging Face tokens,
* large model weights,
* private research materials.

---

## 14. Summary

This project does not claim that a complete video generation system, world model, VLA model, or robot-control system has been built.

The main contribution is the process of:

1. reviewing relevant pretrained I2V / world-model candidates,
2. comparing model requirements with the available GPU server environment,
3. identifying system-level compatibility limitations,
4. selecting a feasible alternative model,
5. running a reproducible CogVideoX-5B-I2V inference pipeline,
6. documenting limitations and future fine-tuning directions,
7. connecting future visual prediction to possible Human-in-the-loop robot action verification.

This repository is intended as an early feasibility study at the intersection of:

* image-to-video generation,
* world-model-based future prediction,
* GPU server compatibility analysis,
* domain adaptation,
* VLA / Physical AI,
* Human-in-the-loop robot assistance.

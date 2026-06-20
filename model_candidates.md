# Model Candidate Review

This document summarizes the pretrained video generation / world-model candidates reviewed for this feasibility study.

The goal was not to exhaustively benchmark all models. The goal was to choose a model that was relevant to image-to-video or action-result prediction and feasible under the available GPU server environment.

## 1. Selection Criteria

The models were reviewed based on the following criteria:

* Relevance to image-to-video or image-to-world prediction
* Public availability of code and weights
* Compatibility with the available GPU server environment
* Dependency complexity
* Feasibility for undergraduate-level experimentation
* Potential connection to future fine-tuning or domain adaptation

## 2. Candidate Summary

| Model             | Type                                        | Why I Considered It                                               | Result                                                                          |
| ----------------- | ------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Cosmos-H-Surgical | Image-to-world / video generation candidate | Closest to the original action-result prediction scenario         | Reached initialization stage but blocked by software stack compatibility issues |
| SurGen            | Image-to-video model candidate              | Related to CogVideoX-based video generation and domain adaptation | Code and weights were not publicly available at the time of review              |
| Cosmos-Predict1   | General world-model family                  | Potential alternative in the NVIDIA Cosmos family                 | Lower priority for this experiment                                              |
| CogVideoX-5B-I2V  | Image-to-video generation model             | Public, feasible, and directly aligned with I2V inference         | Selected and successfully tested                                                |

## 3. Cosmos-H-Surgical

Cosmos-H-Surgical was reviewed first because it was the closest candidate to the original action-result prediction scenario.

However, the available server environment did not match the expected software stack. The main issue was related to CUDA, PyTorch, driver, and Transformer Engine compatibility.

Therefore, I treated Cosmos-H-Surgical as the initial best-fit candidate, but not as a model successfully executed in this repository.

## 4. SurGen

SurGen was considered because it appeared to be related to image-to-video generation and domain adaptation.

However, the code and weights were not publicly available at the time of review, so it was not usable for this project stage.

## 5. Cosmos-Predict1

Cosmos-Predict1 was considered as a general world-model family candidate.

It was not selected as the main experiment target because the goal of this repository was to construct a minimal and feasible image-to-video inference pipeline under the currently available environment.

## 6. CogVideoX-5B-I2V

CogVideoX-5B-I2V was selected because:

* It is a public image-to-video generation model
* It can be used through `diffusers`
* It does not require NVIDIA Transformer Engine
* It was feasible under the available RTX A6000 48GB environment
* It directly supports image-conditioned video generation

## 7. Final Decision

The final model used in this repository is:

```text
THUDM/CogVideoX-5b-I2V
```

The model was selected not because it was the most specialized candidate, but because it was the most feasible model that could actually be tested under the available GPU server constraints.

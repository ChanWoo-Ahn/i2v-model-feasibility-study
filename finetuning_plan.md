# Fine-tuning Plan

This document outlines a future fine-tuning and domain adaptation plan.

No fine-tuning has been completed yet. This file is a planning document for the next stage after inference feasibility testing.

## 1. Motivation

The tested CogVideoX-5B-I2V model is a base pretrained image-to-video model.

Base pretrained models can generate video-like outputs, but they may not capture the dynamics of a specific action, object, scene, or domain without additional adaptation.

Therefore, future work should focus on domain adaptation or fine-tuning rather than only prompt engineering.

## 2. Dataset Construction

A possible dataset structure:

```text id="6y67bu"
dataset/
├── train/
│   ├── sample_0001/
│   │   ├── input.png
│   │   ├── target.mp4
│   │   └── caption.txt
│   ├── sample_0002/
│   │   ├── input.png
│   │   ├── target.mp4
│   │   └── caption.txt
│
└── validation/
    ├── sample_0001/
    │   ├── input.png
    │   ├── target.mp4
    │   └── caption.txt
```

Each sample should contain:

* Input image
* Future video clip
* Text caption describing the expected motion or scene transition

## 3. Caption Schema

A caption should describe:

* Scene context
* Main object or tool
* Current state
* Expected motion
* Expected visual outcome
* Important constraints

Example structure:

```text id="ud13d7"
A [scene description] showing [main object]. The object moves [motion description], resulting in [future visual outcome]. The camera remains [camera condition].
```

## 4. Fine-tuning Options

Possible approaches:

| Method               | Description                     | Notes                                  |
| -------------------- | ------------------------------- | -------------------------------------- |
| LoRA fine-tuning     | Parameter-efficient fine-tuning | More feasible under limited GPU memory |
| Full fine-tuning     | Update full model parameters    | Likely requires more GPU memory        |
| Adapter-based tuning | Add smaller trainable modules   | Depends on available implementation    |
| Prompt-only tuning   | No model update                 | Useful as a baseline but limited       |

## 5. Memory Consideration

Inference memory usage does not directly determine fine-tuning feasibility.

Fine-tuning requires additional memory for:

* Gradients
* Optimizer states
* Activations
* Checkpoints
* Longer training batches

Therefore, fine-tuning feasibility must be tested separately.

## 6. Evaluation

Evaluation should not rely only on visual inspection.

Possible evaluation aspects:

* Temporal consistency
* Scene consistency
* Object motion consistency
* Frame-to-frame stability
* Prompt-video alignment
* Qualitative human evaluation
* Comparison between base model and fine-tuned model

## 7. Short-Term Plan

1. Run more inference experiments with controlled prompts
2. Record runtime, memory, and output characteristics
3. Construct a small input-video-caption dataset
4. Review `cogvideox-factory` or similar training pipelines
5. Test LoRA fine-tuning feasibility
6. Compare base model outputs and adapted model outputs

## 8. Current Status

Current status:

* Inference pipeline: tested
* Candidate model review: completed at feasibility level
* Cosmos-H-Surgical compatibility issue: documented
* CogVideoX-5B-I2V inference: tested
* Fine-tuning: not yet performed
* Architecture modification: not yet performed

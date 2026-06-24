# CogVideoX-5B-I2V Inference

This document summarizes the CogVideoX-5B-I2V inference setup used in this repository.

## 1. Selected Model

```text
THUDM/CogVideoX-5b-I2V
```

CogVideoX-5B-I2V was selected as a feasible image-to-video baseline because it was public, relevant to image-conditioned video generation, and compatible with the available server environment.

## 2. Working Environment

| Component    | Version               |
| ------------ | --------------------- |
| GPU          | NVIDIA RTX A6000 48GB |
| PyTorch      | 2.6.0+cu124           |
| diffusers    | 0.38.0                |
| transformers | 4.49.0                |
| accelerate   | 1.14.0                |
| Python       | 3.10.20               |

The `transformers` version was a key compatibility point. Some versions caused tokenizer or model configuration issues, while `transformers==4.49.0` worked in the tested environment.

## 3. Inference Script

Main script:

```text
scripts/run_i2v_showcase.py
```

It runs the same recipe over two contrasting inputs (`candle.jpg`,
`robot_cube.jpg`) placed in `./inputs/`, writing `.mp4` files to `./outputs/`:

```bash
# put candle.jpg and robot_cube.jpg in ./inputs/ first
python scripts/run_i2v_showcase.py
```

## 4. Script Features

The script is designed to:

* Load local input images (no remote URLs, for reproducibility)
* Center-crop to the model's 3:2 ratio, then resize to 720x480
* Apply a shared negative prompt and a fixed seed
* Run CogVideoX-5B-I2V inference for each input
* Export each result as an `.mp4`
* Print runtime and peak GPU memory per job
* Stop clearly if CUDA is not available

## 5. Tested Run Configuration

Baseline diagnosis run (two contrasting inputs, same recipe):

| Item            | Value                                  |
| --------------- | -------------------------------------- |
| Inputs          | candle (sweet spot), robot+cube (weak spot) |
| Output          | 49-frame video per input               |
| Inference steps | 50                                     |
| Guidance scale  | 6.0                                    |
| FPS             | 8                                      |
| Seed            | 42                                     |
| Peak GPU memory | ~34.6GB (showcase run)                 |
| Runtime         | ~6 min per job                         |

An earlier single-image test measured ~21.2GB peak under the same
resolution/frame settings; the difference is most likely down to when peak
memory was sampled, not a settings change. A selected public preview of the robot-arm/cube result is shown in
results/sample_outputs.md. The earlier candle run
was used as an internal sanity check for simple atmospheric motion, while the
current public results page focuses on robot/action-result examples.

## 6. Current Limitation

This is inference-only testing.

The current result should not be interpreted as a trained action prediction model or a domain-specialized video generation model. It only shows that a pretrained image-to-video inference pipeline can be constructed under the available GPU server constraints.

## 7. Next Step

The next step is to move from simple inference testing to:

* Better prompt and parameter testing
* Dataset pair construction
* Fine-tuning feasibility review
* Temporal consistency evaluation
* Domain adaptation planning
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

## 4. Script Features

The script is designed to:

* Load a local input image
* Resize the image to the model resolution
* Use a fixed seed for reproducibility
* Run CogVideoX-5B-I2V inference
* Export the generated output as an `.mp4` file
* Print wall-clock time
* Print peak GPU memory usage
* Create the output directory automatically if needed
* Stop clearly if CUDA is not available

## 5. Tested Run Configuration

| Item            | Value                                             |
| --------------- | ------------------------------------------------- |
| Input           | Local image                                       |
| Output          | 49-frame video                                    |
| Inference steps | 50                                                |
| Guidance scale  | 6.0                                               |
| FPS             | 8                                                 |
| Seed            | 42                                                |
| Peak GPU memory | Approximately 21.2GB during inference             |
| Runtime         | Approximately 6 minutes in the tested environment |

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

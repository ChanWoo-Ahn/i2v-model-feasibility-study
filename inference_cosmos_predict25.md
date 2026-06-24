# Cosmos-Predict2.5-2B Inference Notes

This document records the Cosmos-Predict2.5-2B inference route used for the qualitative result previews in `results/sample_outputs.md`.

Unlike `scripts/run_i2v_showcase.py`, which is a cleaned CogVideoX baseline script, the Cosmos-Predict2.5 run currently depends on gated Hugging Face access, model-specific environment fixes, and local runtime patches. For that reason, I document the run path and settings here instead of presenting it as a fully reusable public script.

## 1. Purpose

Cosmos-Predict2.5-2B was tested as a follow-up to the CogVideoX baseline diagnosis.

CogVideoX-5B-I2V was useful as an accessible image-to-video baseline, but it struggled with robot-object interaction, especially the gripper-cube relationship during grasping and lifting.

Cosmos-Predict2.5-2B was tested because it is closer to a world-model-style video generation model and was expected to handle physical interaction better.

## 2. Why This Route Worked

Cosmos-H-Surgical was blocked by NVIDIA Transformer Engine, specifically the missing `libtransformer_engine.so` issue under the available server environment.

Cosmos-Predict2.5-2B was different because it had a `diffusers` route. This route avoided the Transformer Engine layer that blocked Cosmos-H-Surgical.

## 3. Environment

Tested server:

| Item         | Value                 |
| ------------ | --------------------- |
| GPU          | NVIDIA RTX A6000 48GB |
| Driver       | 550.54.14             |
| CUDA         | 12.4 in conda         |
| PyTorch      | 2.6.0+cu124           |
| diffusers    | 0.38.0                |
| transformers | 4.52.4                |

CogVideoX was validated with `transformers==4.49.0`, while Cosmos-Predict2.5 required `transformers==4.52.4` because of the Qwen2.5-VL text encoder.

## 4. Key Issues and Fixes

### Gated model access

The model and related guardrail repositories required Hugging Face access approval and login.

The run required:

* accepting the model license on Hugging Face
* running `huggingface-cli login`
* using a local Hugging Face cache directory

### Qwen2.5-VL text encoder

With `transformers==4.49.0`, the text encoder failed to load due to a configuration-related error.

Upgrading to `transformers==4.52.4` fixed this issue.

### `_execution_device` issue

The pipeline produced an `_execution_device`-related runtime issue after moving the model to CUDA.

A local class-level property patch was used during the research run.

### Safety checker

For the controlled research run, the safety checker was disabled with `safety_checker=None`.

## 5. Inference Settings

The qualitative Cosmos-Predict2.5 previews in `results/` used the following general settings:

| Item          | Value                                                |
| ------------- | ---------------------------------------------------- |
| Model         | Cosmos-Predict2.5-2B                                 |
| Route         | diffusers pipeline                                   |
| Output length | ~93 frames / ~6 seconds                              |
| Steps         | 50                                                   |
| Runtime       | ~25 min with CPU offload                             |
| Peak VRAM     | ~32.5 GB                                             |
| Output format | local `.mp4`, converted to downscaled `.gif` preview |

The full `.mp4` files are not committed to the repository. Only lightweight GIF previews are included in `results/`.

## 6. Public Result Previews

The following Cosmos-Predict2.5 outputs are documented in `results/sample_outputs.md`:

* `output_cube_cosmos.gif`
* `output_tool_approach_cosmos.gif`
* `output_tool_contact_cosmos.gif`

These are qualitative previews, not benchmark results.

## 7. Interpretation

On the robot-arm + cube scene, Cosmos-Predict2.5 produced a qualitatively more stable gripper-object relationship than the CogVideoX baseline.

On the real-photo DOFBOT tool-use scenes, the model produced smoother approach and sliding-contact motion.

However, these results should not be interpreted as a reliable robot policy, a physics-guaranteed prediction system, or a domain-specialized model. They are qualitative feasibility observations.

## 8. Why a Full Script Is Not Included Yet

A cleaned public Cosmos script is not included yet because the current run depends on:

* gated Hugging Face model access
* local Hugging Face authentication
* model-specific dependency versions
* local runtime patching
* CPU offload choices
* environment-specific memory behavior

For now, the reproducible public script is limited to the CogVideoX baseline diagnosis. Cosmos-Predict2.5 is documented through this inference note, the README, and the sample output previews.

## 9. Next Step

The next step is to turn the current qualitative side-by-side result into a stricter benchmark with:

* repeated seeds
* clearly logged model-specific settings
* temporal consistency checks
* scene consistency checks
* object-interaction consistency evaluation
* later domain adaptation or fine-tuning

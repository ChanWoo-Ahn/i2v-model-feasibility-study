# Fine-tuning Plan

The next stage, not done yet. What makes this more than a generic to-do list is
that the baseline diagnosis already gave a concrete failure to aim at, rather
than a vague "improve quality."

## What the baseline told me

From the showcase (see [`results/sample_outputs.md`](results/sample_outputs.md)):
the base CogVideoX model is strong on visual realism and simple, ambient motion
(the candle flickers naturally) but breaks on purposeful physical interaction —
with the robot arm, the gripper–cube bond falls apart the instant it lifts.
Frames look fine one by one; the grasp→lift cause-and-effect doesn't hold.

Cosmos-Predict2.5-2B handled that same scene better, which is why it's now the
main candidate — but "better" is not "solved." It's a world model pretrained on
general physical-interaction data, not on any target domain, so fine detail and
domain-specific motion still need adaptation. So the target is the same for
either model: the physical consistency of an object being acted on, which prompt
tuning won't fix because the model hasn't seen enough of that motion.

## Data

Build input → future-clip → caption triples:

```text
dataset/
  train/sample_0001/{input.png, target.mp4, caption.txt}
  val/  sample_0001/{input.png, target.mp4, caption.txt}
```

For the surgical direction, Cholec80 is the obvious source (it's what SurGen used).
Caption should name the scene, the main tool/object, its current state, the
expected motion, and the expected outcome — e.g. "A [scene] with [tool/object].
It [motion], resulting in [outcome]. Camera [static/moving]." The point is to give
the model the action→result mapping the base model is missing.

## Method

| Approach | Trade-off |
| --- | --- |
| LoRA | lightest on memory; first thing to try |
| Full fine-tune | best capacity, heaviest; may not fit |
| Adapters | middle ground, depends on implementation |
| Prompt-only | no training; already shown to be not enough |

For CogVideoX the plan is `cogvideox-factory` with LoRA first (SurGen showed
CogVideoX + surgical data works, and it's built for single-GPU runs). For
Cosmos-Predict2.5 I'd check what the official / `diffusers` fine-tuning path
supports before committing — its tuning story is less settled than CogVideoX's.

## Memory caveat

Inference fits (CogVideoX ~34.6GB, Cosmos-Predict2.5 ~32.5GB of 48GB) — but that
does **not** mean training fits. Fine-tuning adds gradients, optimizer states,
activations, and checkpoints, so training memory has to be measured on its own
before committing to full vs. LoRA.

## How I'll judge it

The honest baseline is the diagnosis itself, so the evaluation is concrete:
re-run the same robot-arm/cube scene after fine-tuning and check whether the
grasp→lift holds together this time, alongside temporal/scene consistency on
held-out clips and a side-by-side base-vs-tuned comparison.

## Status

- candidate review — done
- Cosmos-H-Surgical compatibility limit — documented
- CogVideoX inference (baseline) — done
- baseline strength/weakness diagnosis (candle vs robot+cube) — done
- Cosmos-Predict2.5-2B inference via diffusers — done, now main candidate
- controlled CogVideoX vs Cosmos comparison — not started
- dataset construction — not started
- fine-tuning — not started
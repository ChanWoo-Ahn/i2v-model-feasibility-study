# Fine-tuning Plan

The next stage, not done yet. What makes this more than a generic to-do list is
that the baseline showcase already gave a concrete failure to aim at, rather than
a vague "improve quality."

## What the baseline told me

From the showcase (see [`results/sample_outputs.md`](results/sample_outputs.md)):
the base model is strong on visual realism and simple, ambient motion (the candle
flickers naturally) but breaks on purposeful physical interaction — with the
robot arm, the gripper–cube bond falls apart the instant it lifts. Frames look
fine one by one; the grasp→lift cause-and-effect doesn't hold.

So the target isn't "sharper video." It's specifically the physical consistency
of an object being acted on — which prompt tuning won't fix, because the model
just hasn't seen enough of that kind of motion. That's the case for fine-tuning.

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

Plan is `cogvideox-factory` with LoRA first, since SurGen showed CogVideoX +
surgical data works and `cogvideox-factory` is built for single-GPU runs.

## Memory caveat

Inference peaked ~34.6GB of 48GB, so ~13GB free at inference — but that does
**not** mean training fits. Fine-tuning adds gradients, optimizer states,
activations, and checkpoints, so training memory has to be measured on its own
before committing to full vs. LoRA.

## How I'll judge it

The honest baseline is the showcase itself, so the evaluation is concrete:
re-run the same robot-arm/cube input after fine-tuning and check whether the
grasp→lift holds together this time, alongside temporal/scene consistency on
held-out clips and side-by-side base-vs-tuned comparison. Same inference recipe
(720x480, 50 steps, 49 frames, guidance 6.0, negative prompt) so the before/after
is apples-to-apples.

## Status

- candidate review — done
- Cosmos compatibility limit — documented
- CogVideoX inference — running
- baseline strength/weakness diagnosis (candle vs robot+cube) — done
- dataset construction — not started
- fine-tuning — not started
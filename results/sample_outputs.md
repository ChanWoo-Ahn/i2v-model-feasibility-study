# Sample Outputs — Baseline Diagnosis

Two contrasting inputs run with the **same** inference recipe, to show where the
base CogVideoX-5B-I2V model is strong and where it breaks down before any
fine-tuning. Video files themselves are not committed (`.gitignore` excludes
`*.mp4`); these tables record the settings and observations so the runs stay
reproducible.

Shared recipe: `720x480`, `steps=50`, `frames=49`, `guidance_scale=6.0`,
`bf16`, `fps=8`, fixed `negative_prompt`, `seed=42`. Script:
[`scripts/run_i2v_showcase.py`](../scripts/run_i2v_showcase.py).

## Ceiling shot — model's sweet spot (candle)

| Item | Value |
| --- | --- |
| Input | single lit candle, still image |
| Why this input | single subject + atmospheric/particle motion |
| Frames / steps | 49 / 50 |
| Guidance / seed | 6.0 / 42 |
| Runtime | ~6 min 13 s |
| Peak GPU memory | ~34.6 GB |
| Observation | Flame sways naturally, smoke rises plausibly. Visual realism and simple motion are strong. |

## Diagnostic shot — model's weak spot (robot arm + cube)

| Item | Value |
| --- | --- |
| Input | cobot arm + parallel-jaw gripper + red cube |
| Why this input | multi-object + spatial relation + purposeful manipulation |
| Frames / steps | 49 / 50 |
| Guidance / seed | 6.0 / 42 |
| Runtime | ~6 min 16 s |
| Peak GPU memory | ~34.6 GB |
| Observation | Spatial layout holds, but the gripper–cube bond collapses the moment it lifts. Individual frames look plausible; the physical cause–effect of grasp→lift does not. |

## Takeaway

The base model is strong on visual realism and simple motion (candle) but breaks
on precise physical interaction (grasp/lift) without domain adaptation. This is
the concrete evidence behind the fine-tuning plan: prompt engineering alone is
unlikely to close that gap. The "plausible per-frame but physically inconsistent
motion" pattern is one that also comes up in recent discussion around robot
world-model video generation — I'm noting the resemblance qualitatively, not as a
benchmarked comparison.

> Note on memory figures: an earlier single-image test measured ~21.2 GB peak;
> this showcase run measured ~34.6 GB under the same resolution/frame settings.
> The difference is most likely down to when peak memory was sampled
> (`reset_peak_memory_stats` was used here), not a settings change.
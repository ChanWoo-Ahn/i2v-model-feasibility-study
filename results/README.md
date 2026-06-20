# Results

This directory is intended for general, non-sensitive sample outputs.

## 1. What Can Be Included

Recommended contents:

* General sample input images
* Generated sample videos if file size is small
* Output notes
* Runtime and memory summaries
* Qualitative observations

## 2. What Should Not Be Included

Do not upload:

* Private data
* Sensitive images
* Medical or patient-related data
* Server logs containing private paths or credentials
* Large model weights
* Large raw datasets
* API keys or access tokens

## 3. Large Output Files

For now, large `.mp4` files are ignored by `.gitignore`.

If sample videos are needed for a presentation or private review, they should be stored separately rather than committed directly to GitHub.

## 4. Recommended Result Recording Format

Use `sample_outputs.md` to record:

* Input filename
* Output filename
* Prompt
* Number of frames
* FPS
* Inference steps
* Guidance scale
* Seed
* Wall-clock time
* Peak GPU memory
* Qualitative observation

This makes the result reproducible without committing large binary files.

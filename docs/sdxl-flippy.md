# Experimental SDXL Flippy Illusions

Open [the new notebook in Colab](https://colab.research.google.com/github/adrianIam/Diffusion-Illusions/blob/codex/sdxl-flippy-colab/flippy_illusions_sdxl_colab.ipynb). Its setup selects this fork and the review branch. After merging, change the link and notebook `REF` to `master`, or select a full commit SHA for an immutable experiment.

Select **Runtime → Change runtime type → GPU**. Run notebook cells 1–2, restart the session, rerun cell 1, then run cells 3–7. Cell 5 defaults to a four-step exploratory smoke test. Cell 6 runs two steps, reloads the models and checkpoint, and resumes to four, with both views exercised every step. Save the manifest and comparison PNG before trying a long experiment. There is no CPU generation fallback and no specific GPU/VRAM fit guarantee.

The smoke test is a prerequisite for a longer user experiment, not a quality assessment. For the SDXL reference, set `MODE = "experiment"` in cell 5; defaults are 1,000 iterations, 256-square learnable image and 1024-square model input. SD1 is selectable with `BACKEND = "sd1"`; it uses 512-square model input. These input sizes are separate settings. Model-size 512 SDXL is an explicit lower-memory exploration with a possible quality cost. Do not use turbo/distilled models with this SDS formulation without separate research and validation.

## Implementation and baseline

Inspected fork base: `71646f3203e31f126cf2d7998d2507ee326b7c54`, default branch `master`. No applicable `AGENTS.md` or existing user edits were present in the fresh checkout. The fork still used a single CLIP encoder, fixed `[2B,77,768]` label embeddings, 512 resizing, 0.18215 scaling and a hard-coded PNDM scheduler. `source/peekaboo.py` and the other notebooks call that public API, so `source/stable_diffusion.py`, `stable_diffusion_labels.py` and `learnable_textures.py` remain unchanged. Only the original Flippy notebook's destructive clone cell is replaced with safe fork-aware setup.

`source/illusion_backends.py` owns model loading, frozen components, prompt caching, differentiable VAE encoding and UNet scoring. `source/flippy.py` owns the Fourier image, two views, SGD, configuration, persistence and exports. The Fourier image is a dependency-light adaptation of the original square-image representation. A CPU regression check compares its initialized output and parameter gradients to the original implementation exactly. Another check compares the SD1 noise-only SDS gradient to the old `train_step` under identical fake components, noise and timestep. These checks establish conventions, not pretrained-model or visual equivalence.

SDXL uses Diffusers' version-pinned `StableDiffusionXLPipeline.encode_prompt` for both text encoders, including negative token and pooled embeddings. `_get_add_time_ids` supplies `(original_h, original_w, crop_y, crop_x, target_h, target_w)` and validates the UNet's added embedding dimension. Every cached view is ordered `[negative, positive]` for tokens, pooled embeddings and time IDs. Fixed SD1 labels are not reused for SDXL.

The backend constructs `DDPMScheduler.from_config(pipe.scheduler.config)` for the **training forward marginal**. It preserves the pretrained beta schedule, timestep count and prediction type; it does not run an inference sampler's Euler trajectory or import SD1's hard-coded PNDM values. Epsilon and velocity prediction are explicitly supported; other prediction types fail. Timestep fractions are converted to inclusive indices and clamped to the configured training range. The checked SDXL base config uses 1,000 steps, epsilon prediction, scaled-linear betas 0.00085–0.012, and VAE scaling **0.13025** with `force_upcast=True`. VAE scaling comes from each model, never a shared SD1 constant.

The summed latent SDS gradient is `view_weight * noise_coef * (1 - alpha_t) * (epsilon_cfg - noise)`. A detached-gradient surrogate applies it once, without pixel/latent averaging. Its scalar value is not an image-quality loss. Model parameters are frozen; text conditioning and UNet scoring use no-grad, while resizing and VAE encoding retain autograd to the learnable image. The default UNet/text policy is fp16 on CUDA with a float32 VAE and float32 SDS arithmetic. The official SDXL fp16 weight variant is selected for reduced-precision loading to limit download/host RAM use; the VAE is upcast before encoding. bfloat16 requires GPU support and a compatible VAE. No whole-pipeline fp16 cast or inference-mode VAE is used. Default attention uses Diffusers/PyTorch SDPA; optional attention slicing is explicit. VAE tiling, sequential CPU offload, xformers and VAE gradient checkpointing have not been validated here.

The new SD1 reference uses the same modern dependency set as SDXL, a frozen VAE, model-config noise/scaling, the original Fourier architecture and SGD, random one-view updates, normalized view weights, noise coefficient 0.1 and guidance 60. SDXL starts at guidance 20; neither value is a measured optimum. The original notebook remains the historical workflow. Its unpinned historical dependencies have **not** been reconstructed, so it is not an exact environment/performance baseline. Random-view selection now uses seeded Python RNG instead of `rp.random_batch`; equal seeds do not imply matching old draws or matching model noise trajectories.

## Dependencies and available checks

`requirements-sdxl.txt` is separate from the untouched historical `requirements.txt`. Verified installation/imports on Python 3.12.14 with **torch 2.6.0+cpu, diffusers 0.35.1, transformers 4.56.2, accelerate 1.10.1, huggingface-hub 0.34.4, safetensors 0.6.2, numpy 2.2.6 and Pillow 11.3.0**. `pip check` passes in the isolated environment. The notebook keeps Colab's CUDA-enabled PyTorch (`>=2.6,<3`) and records its actual version. This range is not a claim that every Colab package combination has been tested; restart and run the smoke test on the assigned runtime.

CPU checks include configuration rejection, rotation and its gradient, CFG/view batch ordering, epsilon/velocity conversion, correct weighting, finite nonzero image gradients with frozen fake model weights, unchanged SD1 shapes/conventions, failed atomic writes, full model/optimizer/RNG checkpoint round trips, mismatch rejection, output/manifest writes, OOM preservation, and exact uninterrupted-versus-resumed fake-component CPU execution. An offline integration check builds tiny random real SDXL pipeline/CLIP/VAE/UNet components, then runs the actual conditioning APIs and image gradients without downloading weights. It does not test the pretrained SDXL model.

Implementation validation result: **39 tests passed**, new notebook schema and all seven Python cells validated, compilation passed, `pip check` found no broken requirements, and `git diff --check` passed. The comparison preparer produced twelve configurations without starting generation. The historical Flippy computation cells remain byte-for-byte equivalent in source after its setup cell.

From the repository root:

```bash
python -m pip install -r requirements-sdxl-test.txt
python -m pytest -q tests
python -m compileall -q source/flippy.py source/illusion_backends.py scripts
python -m pip check
```

No GPU exists in the implementation environment. **Real SDXL/SD1 loading, CUDA dtype/memory behavior, GPU save/resume, completed image generation, visual quality and comparative speed/memory remain pending. No improvement is claimed.** The minimal deferred commands, after installing the new dependency set in a GPU runtime, are:

```bash
python -m source.flippy --backend sd1 --smoke --output-dir /content/sd1-smoke
python -m source.flippy --backend sdxl --smoke --output-dir /content/sdxl-smoke
```

The notebook additionally verifies an actual reload/resume. On failure, return the traceback, GPU/package summary, `manifest.json` if created, and `comparison.png` if present. CPU tests do not replace these user-run checks. A failed initial download/load may precede checkpoint creation; the notebook explicitly identifies that case.

## Checkpoints, outputs and failure handling

Each new run requires an empty output directory. Prefer optional Drive output for long runs, or download the ZIP from notebook cell 7. Colab local disk does not survive runtime recycling. Drive synchronization itself is not transactional: retain downloaded backups.

At initialization, save intervals and completion, `checkpoint.pt` is written to a temporary file, flushed, then atomically renamed. It stores image parameters **and buffers**, optimizer state, completed iteration, full settings, Python/NumPy/CPU and available CUDA RNG states, cumulative measurements and model/scheduler/dependency/device identity. The DDPM training noise calculation is stateless: there is no inference sampler trajectory to restore. Its full scheduler configuration and actual weight revision when obtainable are recorded and checked on resume. The loader resolves a Hub SHA before downloading when possible and passes that SHA to Diffusers.

Only total iterations, per-invocation time budget, save interval and output directory may differ on resume. Prompts, seed, image/network sizes, backend/model/revision, dtype, optimizer settings, view sampling and weights must match. The GPU model/device and package versions must match too; moving to another Colab GPU may require a fresh run. CUDA kernels can still be nondeterministic on the same device, so bitwise GPU reproducibility is not promised. A checkpoint is a trusted Python/pickle artifact; do not load one from an unknown source.

Output files are `image.png` and `upright.png` (the same optimized image), `flipped.png`, `comparison.png`, periodic `preview-NNNNNN.png`, `checkpoint.pt`, and `manifest.json`. Previewing temporarily switches BatchNorm to eval and restores its training mode, avoiding hidden buffer or RNG changes. The manifest records source commit, backend/model revision, scheduler config, conditioning shapes, package/CUDA/GPU versions, all settings, completed iterations, scored-view counts (UNet evaluation counts), elapsed optimization time, load time per session and peak allocated/reserved GPU memory. Optimization time includes progress/export/checkpoint overhead; model loading is measured separately. Peaks include model loading and preparation. Cumulative peaks are maxima across sessions, not sums.

On OOM, a nonfinite gradient or interruption, the last complete checkpoint is kept; a partially applied step does not replace it. Resume that checkpoint with unchanged settings, or start a new lower-memory experiment. Suggested new-run settings: `image_size=64`, `hidden_dim=64`, `num_features=32`, `model_size=512`, optional `attention_slicing=True`. Nothing silently changes the experiment. No compute is purchased or launched by these tools.

## Bounded comparison and human rubric

The suite in `configs/flippy_evaluation.json` pairs existing giraffe/penguin, cat/dog and cow/walrus prompts, each with seeds 0 and 1. These are plausible starting pairs, not known successes. Preparation writes twelve reviewable JSON configurations and a blank human-rating CSV; it starts no GPU work:

```bash
python scripts/compare_flippy.py --output-root /content/equal-iterations --iterations 1000
# Inspect the generated JSON first. Explicit execution, optionally limited:
python scripts/compare_flippy.py --output-root /content/equal-iterations --iterations 1000 --run --limit 2
# Resume unfinished jobs in that batch without changing experiment settings:
python scripts/compare_flippy.py --output-root /content/equal-iterations --iterations 1000 --run --resume-existing
# Equal optimization wall-clock budget; iteration ceiling prevents an unbounded run:
python scripts/compare_flippy.py --output-root /content/equal-time --iterations 100000 --max-seconds 1800
# Add --run only after reviewing the desired batch and available compute.
python scripts/compare_flippy.py --output-root /content/equal-iterations --summarize
```

Equal-time mode stops after the first complete iteration at or above the session budget, so one iteration and checkpoint/export overhead can exceed it. Loading is excluded but reported. Reference SD1/SDXL use different model input sizes (512/1024) and starting guidance values (60/20); report them alongside GPU, iterations, view counts and runtime. Keep the same prompt pair and seed per comparison. Complete as many of the 12 jobs as compute permits; disclose missing runs and interrupted/budget-limited trials. Run SD1 first before attempting long SDXL trials if resources are tight. Do not tune repeatedly until one result looks good: record every setting tried and bound any next tuning experiment.

Inspect upright and flipped views separately, ideally with backend labels hidden. Fill `ratings.csv` with 0–4 scores for each view's **subject recognition**, plus aesthetics and balance. Recognition anchors: 0 absent, 1 vague resemblance, 2 recognizable with the prompt, 3 recognizable without the prompt, 4 clear and convincing. Aesthetics: 0 severe artifacts to 4 coherent and appealing. Balance: 0 only one view works to 4 both are comparably convincing. Use the **minimum** of the two recognition scores as the primary two-view assessment; retain both individual scores and notes. Report all prompt/seed pairs, including failures. Optional CLIP scores would only support human inspection; none are currently computed.

`measurements.csv` summarizes actual manifests, including failed/incomplete statuses. Human quality ratings remain separate. A smoke test establishes execution; a longer run supplies one example; neither establishes general visual improvement. Until comparable runs are measured, quality, speed and memory conclusions remain pending.

## Version-specific sources checked

- [Diffusers 0.35.1 SDXL API](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/stable_diffusion/stable_diffusion_xl), including installed `encode_prompt` and `_get_add_time_ids` signatures/source.
- [Diffusers SDXL guide](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/sdxl).
- [SDXL scheduler config](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/scheduler/scheduler_config.json) and [VAE config](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/config.json).
- [Colab resource FAQ](https://research.google.com/colaboratory/faq.html).

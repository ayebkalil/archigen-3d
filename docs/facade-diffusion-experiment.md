# Local facade diffusion experiment

Branch: `experiment/facade-diffusion-control`, based on the validated 512 experiment
at `3dc23fc`. No production model selection, backend route, or viewer code changes.

The previous 512 GAN gained about 1.3% LPIPS and reduced opening edge error by
about 11.9%, but its facade textures remain soft. This experiment tests a pretrained
SDXL image prior before spending more time training the same GAN. A higher pixel
count alone does not establish photorealism.

## Implementation

- SDXL base 1.0, smaller SDXL Canny ControlNet, and FP16-compatible SDXL VAE.
  Exact model revisions and file allowlists are in `configs/facade_diffusion.json`.
  Required weights are approximately 7.4 GB; duplicate fp32/base VAE/ONNX exports
  are excluded. Upstream model licences apply (see source model cards below).
- 1024 x 1024, one candidate at a time, 30 steps, guidance 5.0, control strength 1.0,
  CPU-seeded noise (42), FP16, sequential CPU or disk offload and tiled VAE decoding.
  This is native 1024 synthesis, not an upscale of the 256 GAN output.
- Viewer controls use authoritative opening UV rectangles, converting bottom-origin
  UVs to top-origin image coordinates. CMP controls use original indexed class
  boundaries; the CMP colour palette is not passed to a pretrained segmentation model.
  Controls never use target-photo edges. No depth is invented from a floorplan.
- Text prompts support the four requested material directions plus neutral. They
  use pretrained text guidance; no four-class style training on CMP is claimed.
  Named-style accuracy remains unvalidated.
- Optional generated viewer fixtures preserve the existing UV bounds, facade frame,
  dimensions and openings. Generation does not install a texture into the live app.

## Resource guards

The download action requires 15 GiB free on the cache drive (2 GiB when all pinned
files are already cached). Generation additionally requires 8 GiB available system
RAM for CPU offload and at least 1536 MiB headroom below the 3500 MiB device-wide
budget. Disk offload instead requires 3 GiB available RAM and 12 GiB free for a
separate memory-mapped weight copy (22 GiB when source weights are not yet cached).
These are conservative preflight thresholds, not measured minimum requirements.

Disk mode loads and immediately offloads one component at a time using Accelerate.
It retains meta parameters plus disk-backed arrays instead of all model tensors in
RAM. Its cache is revision-keyed and kept outside OneDrive by default. This is slower
than CPU offload, but the user's available RAM did not meet the CPU-mode threshold.

The PyTorch allocator is capped at the remaining device budget minus 512 MiB.
Device usage is checked after loading, every denoising step and after decoding.
Run metadata records sampled total GPU usage and PyTorch peak allocated/reserved
memory. This is **not** a hard device-wide guarantee: external applications and
non-PyTorch allocations are outside the allocator cap, and samples can miss peaks.
An OOM or a budget violation records a failed run and frees model references.
No silent resolution downgrade or fallback to the old GAN occurs.

Only the previously authorized ArchiGen Uvicorn process was stopped. Unrelated
applications are not stopped automatically. Docker is not needed.

## Reproduce in PowerShell

Run from the repository root. Keep the new environment separate from production:

```powershell
python -m venv --system-site-packages .venv-facade-diffusion
.\.venv-facade-diffusion\Scripts\python.exe -m pip install --no-cache-dir -r requirements-facade-diffusion.txt
python scripts/facade_diffusion.py prepare
.\.venv-facade-diffusion\Scripts\python.exe scripts/facade_diffusion.py preflight
$env:HF_HUB_DISABLE_XET='1'
$env:HF_HUB_DOWNLOAD_TIMEOUT='120'
.\.venv-facade-diffusion\Scripts\python.exe -u scripts/facade_diffusion.py download
.\.venv-facade-diffusion\Scripts\python.exe -u scripts/facade_diffusion.py generate --case viewer --style neutral --seed 42
```

On the current laptop use disk mode for preflight and generation:

```powershell
.\.venv-facade-diffusion\Scripts\python.exe scripts/facade_diffusion.py preflight --offload disk
.\.venv-facade-diffusion\Scripts\python.exe -u scripts/facade_diffusion.py generate --offload disk --case viewer --style neutral --seed 42
```

The environment reuses existing CUDA Torch 2.6 and OpenCV/Pillow/NumPy. Its older
Transformers/Hub versions override those packages only inside this venv; production
versions remain unchanged. Inherited `sentence-transformers` has incompatible
requirements with this isolated stack and must not be used here. The diffusion
pipeline imports have been verified. Use the normal Python for the existing app.

All actions accept `--cache <folder>` for an alternate local model drive. Inference
sets Hub and Transformers offline and refuses missing weights; download is explicit.
The default output directory is `outputs/phase3`. A candidate directory includes
case/style/seed and refuses overwrite. Failed candidates remain recorded; use a
different `--output` directory with a fresh `prepare` to repeat the same seed.

Large weights use four concurrent 8 MiB HTTP range requests, with strict response
range/length validation, resumable contiguous partial files, and final SHA-256
verification against pinned Hugging Face LFS metadata. This avoids the stalled full
GET observed on this network. Restarting `download` resumes complete saved chunks;
it never overwrites an existing file that fails its checksum.

For the prepared validation cases:

```powershell
.\.venv-facade-diffusion\Scripts\python.exe -u scripts/facade_diffusion.py generate --case val_00 --style neutral --seed 42
```

## Evaluation and acceptance

`prepare` fixes the first four sorted validation IDs and the existing five-opening
viewer fixture. It records source/control SHA-256 values. It leaves the held-out
test split unused for tuning. CMP references may be smaller than 1024; their native
dimensions are recorded, and resized photos are not claimed as native 1024 targets.

Inspect every candidate, including failures. For the viewer compare with the previous
512 GAN; validation sheets compare with CMP photographs and label them as references.
These are different comparisons, not paired model improvement metrics. Generation
records prompt, seed, package versions, pinned models, timing and memory evidence.
Generated images remain pending visual review until checked for:

1. Correct opening count, locations and bounds, with no extra or warped windows.
2. Material realism at wall scale: mortar, stucco and glass rather than blur or grids.
3. Consistency in the actual Three.js viewer, including seams and lighting.

ControlNet guidance cannot promise exact geometry. If opening locations drift, reject
the candidate; investigate stronger structural conditioning or geometry-masked material
generation, rather than presenting frame UV alignment as image alignment. Glass and
frames can remain real 3D materials. Keep diffusion's baked lighting restrained.

Only after validation selection, evaluate the frozen candidate once on held-out CMP,
record perceptual/structural measures and visual failures, and decide whether to add an
opt-in backend adapter. No photorealism, benchmark gain, local speed, or four-style
support is claimed by the prototype alone.

## Verification so far

- Existing suite plus geometry/resource/downloader tests: **92 passed**, 20 existing warnings.
- The subsequent disk-resource contract also passes (14 experiment tests in total).
- Isolated SDXL/ControlNet/VAE imports and CUDA Torch 2.6 verified.
- Five 1024 control cases prepared; viewer opening control visually inspected.
- All required weights downloaded with pinned SHA-256 verification.
- Tiny random-weight SDXL CUDA smoke tests passed with CPU and disk offload, two steps
  each, 11.9 MiB peak allocated / 26 MiB reserved. These are API checks, not real model
  memory benchmarks or image quality evidence.
- Full-size disk-offloaded inference initiated; output quality remains unverified.
- Production frontend code unchanged; existing normalized UVs are resolution-independent.

GitHub workflow runs the new CPU contracts without downloading weights or using a
cloud GPU. Heavy weights and outputs are ignored by Git.

## Primary references

- [SDXL base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Small SDXL Canny ControlNet](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small)
- [FP16 SDXL VAE fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- [Diffusers memory optimizations](https://huggingface.co/docs/diffusers/v0.35.2/optimization/memory)
- [Accelerate disk offload](https://huggingface.co/docs/accelerate/v1.12.0/en/package_reference/big_modeling#accelerate.disk_offload)

# Facade V2: semantic ControlNet and UNet LoRA

Experimental branch: `codex/facade-semantic-controlnet-v2`. Production UI,
ONNX weights and the completed V1 experiment are unchanged. This is a candidate,
not a demonstrated quality improvement until matched evaluation is available.

## What changes

| Component | Completed V1 | V2 candidate |
|---|---|---|
| Frozen prior | SD1.5 pinned revision | Same SD1.5 revision |
| Geometry | Three identical binary edge channels | Edge channel + six one-hot classes: background, wall, window, door, balcony/railing, plant |
| Trainable model | ControlNet only | ControlNet + rank-8 UNet attention LoRA |
| Initialization | Pretrained Canny ControlNet | Selected V1 step-750 checkpoint, SHA-verified; edge response preserved, new semantic kernels zero-initialized |
| Training photos | 400 CMP | Locked CMP train + audited original IRFs train |
| Input geometry | Square stretch | Aligned facade crop + optional storey strip, aspect-preserving 512 letterbox |
| Loss | Opening-weighted epsilon MSE | Valid-area opening-weighted epsilon MSE + Min-SNR gamma 5; padding excluded |
| Selection | Four CMP validation examples | Fixed stratified 12 CMP + 4 IRF, source-macro LPIPS |

LoRA learns a small facade-domain adaptation without distributing/retraining the
whole base model. No adversarial or pixel-perceptual loss is added directly to
noisy epsilon predictions: these are not RGB images. LPIPS is used for selection
and paired image evaluation, not mislabeled as a denoising training objective.
Physical batch 1, accumulation 4, FP16, gradient checkpointing and 8-bit Adam
target a single Kaggle T4. Laptop **inference**, not training, uses sequential CPU
offload, FP16 and tiled/sliced VAE. Actual local VRAM/runtime must be measured on
the selected export before production use. A 3.5 GB limit cannot be guaranteed
for arbitrary simultaneous workloads.

## Data and leakage controls

The regularization collection contains alternative processed versions of the
same facades, including CMP. They are not independent additional photographs.
The original **IRFs** source retains human masks for wall and vegetation as well
as openings, so V2 attaches `liushuyuu/irregular-facades-irfs` directly. Its
processed/predicted regularization variants are not ground truth.

CMP's 400/100/106 split is immutable. Original IRFs pairs require matching
photo/PNG/annotation JSON dimensions and building provenance. Exact duplicates
and pHash-near photos are screened against **all** CMP partitions. IRFs building
URLs (without tracking parameters) and pHash-linked views are grouped before a
deterministic 80/10/10 building split. This reduces leakage; it cannot prove the
absence of every unrecognized cross-source view. Exclusions and actual counts
are recorded in `audit.json`; corrupt or unknown-palette inputs are not silently
used. No test image participates in optimization or checkpoint selection.
The sparse original CMP `train/236.jpg` mask has only 2.55% annotated foreground:
its locked membership is retained, but it is excluded from optimization. Effective
CMP training count is therefore 399 rather than silently inventing missing wall
labels. Effective counts are tracked separately from original split sizes.

IRFs numeric order is background/plant/wall/window/door/fence; its exact palette
is mapped to the common six classes. CMP maps shutter/shop to window, balcony
to railing, and other architectural trim to wall. **Neither dataset has the
four named UI style labels.** Prompt styling remains based on pretrained text
knowledge plus unlabelled domain adaptation; no supervised four-style claim.

IRFs/ArchDaily terms: academic research, CC BY-NC-SA 4.0; acknowledge ArchDaily,
credited architects/photographers, and Wei et al. (2024). CMP is CC BY-SA. Keep
the experiment/data private; do not describe this mixture as unrestricted
commercial training material. See generated attribution and original metadata.

## Reproduce on Kaggle

```powershell
python -m experiments.facade_semantic_v2.manage stage-kernel --output outputs/kaggle-v2-smoke --mode smoke
# In that directory, authenticated Kaggle CLI:
kaggle kernels push --path . --accelerator NvidiaTeslaT4
```

Stage again with `--mode train` only after smoke passes. Notebook is private,
Git source is pinned to a full SHA, parent weights are pinned by SHA, and dataset
input versions and hashes are captured in the manifest. It attaches the private
CMP original dataset, original public IRFs, and the saved V1 notebook output.
Retain Kaggle native Torch; install the separate experiment requirements.

Smoke executes two optimizer steps, real validation, resumable checkpoint,
FP16 export, and fresh-process CPU-offloaded export reload/inference. Smoke
weights are explicitly marked and rejected by the local installer. Full mode
trains 2000 steps, validates every 500, exports the best **paired** ControlNet and
LoRA, then evaluates both V1 and V2 on all CMP/IRF validation examples with the
same new input geometry, prompt, sampling settings and per-image seeds.

Checkpoint contains ControlNet, optimizer/scaler/RNG, separate LoRA, epoch/batch
position and config/data/validation contract. For resume, copy the entire saved
run (including best-controlnet and best-lora) to a writable Kaggle folder and
invoke `train --resume PATH/checkpoints/step_XXXXXX` with identical config/data
and parent. Old optimizer checkpoints are pruned before saving the next to
limit temporary disk usage; interruption during replacement can lose the latest
resumable state, while the separately saved best model remains available.

## Evaluation and local installation

Report LPIPS, SSIM, PSNR and opening-edge MAE separately for CMP and IRF, with
paired LPIPS bootstrap intervals and visual comparison panels. Metrics exclude
letterbox padding. Very narrow valid rectangles are upscaled proportionally to
a minimum short axis of 64 for LPIPS only (AlexNet pooling requirement); RGB
metrics retain their original valid-area resolution. Opening-edge MAE is a
photo-edge proxy, **not semantic IoU**.
V1's published square-stretch benchmark is a different protocol; compare V2 only
with the matched V1 results generated here. Selection-subset performance is
optimistic; full validation is still not the untouched final test. Inspect extra
windows, missing doors and texture repetition before declaring improvement.

Changes to architecture and data are combined in this first trial, so an
improvement cannot be attributed to either alone. A later CMP-only ablation and
semantic-only/no-LoRA ablation are needed for causal attribution.

```powershell
python -m experiments.facade_semantic_v2.manage install --archive PATH.zip --destination models/saved/facade-semantic-v2 --tracking-uri sqlite:///mlflow.db
```

The bundle contains ControlNet + LoRA and portable MLflow events/artifacts;
base SD1.5 is pinned and cached separately. Install dependencies in the isolated
diffusion venv. `SemanticFacadeRuntime(export_folder)` supports `generate_sketch`
for the bridge's already-letterboxed 512 CMP-palette image or `generate_control`
with B×7×512×512 tensors. Local inference needs CUDA and at least 4 GiB available
RAM. Runtime validates both weight hashes and seven-channel architecture before
loading. Export reload is tested in smoke; production service switching and a
new local house-wall profile are deferred until quality is demonstrated.

Sources: [IRFs research/data](https://www.mdpi.com/2075-5309/14/9/2602),
[original IRFs dataset](https://www.kaggle.com/datasets/liushuyuu/irregular-facades-irfs),
[ControlNet](https://huggingface.co/docs/diffusers/training/controlnet),
[LoRA](https://huggingface.co/docs/diffusers/training/lora).

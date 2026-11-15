# V3 reference and material constraints experiment

Branch: `codex/facade-v3-reference-locked`. Research only; the application default,
production checkpoints, and production Viewer3D implementation are unchanged.

## What failed in V2

V2 generates each panel independently as a **photograph**. Semantic ControlNet
conditioning is soft, and the neutral prompt allows different wall materials.
No cross-panel material reference exists. Photo lighting, plants, context, and
invented glass doors can therefore become painted details on a solid 3D wall.
Better paired validation LPIPS did not predict this deployment failure.

## Implemented ablation — no new trained weights

1. Select one existing PBR material (`white_stucco`) for the entire house. Bake
   1024-square albedo, tangent normals, and roughness maps into the measured
   facade UV bounds. Tile in physical metres with phase continuity along each
   closed perimeter. The target 2 m tile period is adjusted per loop; this is
   not a promise of brick coursing continuity between different storeys.
2. Orthographically capture each actual facade skin and its actual frame/leaf
   meshes. Capture **unlit wall albedo**, without tone mapping, so directional
   illumination is not baked into the wall's reference. This is a synthetic
   reference input, never a generated result or a ground-truth photograph.
3. Reuse the exact V2 ControlNet and UNet LoRA in the real Diffusers
   `StableDiffusionControlNetImg2ImgPipeline`. Inputs are the reference plus the
   seven-channel semantic tensor. Seed 42, 25 scheduler steps, strength .35,
   CFG 5, semantic control 1.7. There are eight active denoising steps. No
   weights are retrained, silently substituted, or promoted.
4. Preserve the raw reference-guided prediction for inspection. Separately,
   construct a **hard compositor**: replace neural pixels outside actual
   measured openings plus a 4.5 cm trim band with the shared material. Feather
   only inside that support. Solid panels become byte-identical to the base.
5. In the experimental viewer only, keep existing physical openings, frames,
   glass and door geometry. An optional switch maps the neural opening detail
   onto the existing panes/leaves using physical planar projection. Shared
   normal and roughness maps retain dynamic 3D lighting on protected walls.

Inference is local FP16 with sequential CPU offload, checksummed V2 weights,
RAM/headroom checks and a 3500 MiB device-memory budget. Device-wide sampling is
every .5 seconds, not an instantaneous peak guarantee. Only recognized project
GPU services may be paused; the session wrapper restores them in `finally`.
Each prediction cache key includes weights, LoRA, sketch, reference, material,
settings and protocol; its image checksum must also match.

## What this does and does not establish

The compositor guarantees unchanged protected wall pixels. That is a rendering
constraint, **not a learned semantic accuracy score**. Neural artifacts may
remain inside an allowed opening/trim band. Low denoising strength preserves
layout but adds little photographic richness. A smoother or simpler image must
not be called more photorealistic without evidence.

The first exploratory run used lit PBR references. This preserved orientation-
dependent lighting in the raw output, worsening a naive cross-panel mean-colour
dispersion proxy. The final protocol uses wall albedo references instead. Both
raw trials remain on disk for inspection; neither is a held-out model benchmark.

`assess.py` produces texture-space wall-colour dispersion and blank-wall Sobel
activity, with explicit limitations. These are diagnostics, not semantic false
opening detection, material recognition, human realism ratings, or LPIPS against
an unavailable house ground truth. This one bundled house cannot establish
generalization to arbitrary plans, materials, or four unlabelled style classes.

## Reproduction

Use the existing V2 export installed with the validated ZIP installer. The base
SD1.5 snapshot must already be cached; inference downloads are disabled. Reuse
the existing V2 `house_multiface prepare/generate` commands to produce a source
case, or supply that previously generated case directory. Both its JSON layout
and every source sketch are retained in the output.

From this worktree, substitute your absolute project, source-case and model paths:

```powershell
python -m experiments.facade_reference_v3.run prepare --project PROJECT --source V2_CASE --output outputs/albedo-house
python -m experiments.facade_reference_v3.run stage --project PROJECT --output outputs/albedo-house
```

Serve `PROJECT/frontend` with Vite on 3001. With Puppeteer and Edge/Chrome installed:

```powershell
node experiments/facade_reference_v3/render_references.cjs PROJECT ABS_OUTPUT
python -m experiments.facade_semantic_v2.local_house_session --project PROJECT --model V2_EXPORT --output outputs/albedo-house --module experiments.facade_reference_v3.run
python -m experiments.facade_reference_v3.assess --output outputs/albedo-house
python -m experiments.facade_reference_v3.run stage --project PROJECT --output outputs/albedo-house
node experiments/facade_reference_v3/render_checks.cjs PROJECT ABS_OUTPUT
```

Open <http://127.0.0.1:3001/tmp/facade-reference-v3/>. Compare V2 raw,
raw reference-guided, constrained, synthetic reference input and semantic guide.
Use the side selector followed by **Face this wall**, or the two overview
buttons. All screenshots come from the actual production Viewer3D component.

Python checks:

```powershell
python -m pytest tests/ -q
FACADE_PYTHON -m pytest tests/test_facade_reference_v3.py -q
```

The second command uses an environment with Diffusers, PEFT and Torch. Its real
tiny-model test verifies image-to-image denoising with semantic7, finite output,
and shared component identity; it is not a mock. The full worktree suite may
skip optional model/data dependencies absent from that particular checkout.

## Next learned-model experiment on Kaggle — proposed, not launched

The architectural gap is **conditional material learning and geometric
consistency**, rather than a larger base model alone.

- Initialize from the validated V2 checkpoint. Extend ControlNet's conditioning
  from semantic7 to semantic7 + aligned RGB material reference (10 channels),
  with the three new input kernels initially zero. Keep the affordable SD1.5
  base and train ControlNet plus a small UNet LoRA. Four architectural style
  labels remain unsupported until labelled data or evaluated reference-image
  conditioning exists.
- Train a lightweight six-class facade segmenter only on the training split.
  Validate it separately, freeze it, and use differentiable segmentation
  consistency on decoded predicted-clean images at selected low noise steps.
  Penalize window/door probability outside approved masks as well as missed
  openings. Keep the standard diffusion objective; do not optimize an
  unvalidated teacher as the only reward.
- Add real annotated opening-free wall crops, narrow side elevations and
  single-storey strips. A blank prompt alone cannot counter the dataset's
  strong window-rich photo prior. Protect existing locked splits and building
  groups; do not select checkpoints using the final test split.
- Use material reference crops and adjacent facade crops to supervise wall
  texture consistency while excluding openings, vegetation and background.
  Reference extraction from target photos is acceptable training conditioning,
  but validation must disclose it. Deployment validation should also use
  references from a different building/material bank, not the target photo.
- Select on opening IoU, window/door false-positive area on blank walls,
  cross-panel material consistency, and visual review alongside LPIPS/SSIM.
  Log separate reference-assisted and layout-only protocols in MLflow. Require
  improvement on unseen buildings and several real 3D houses before promotion.

This follows the semantic-feedback idea in [ControlNet++](https://arxiv.org/abs/2404.07987),
while retaining the hard compositor as a deployment guard. Training-only
segmentation feedback need not run on the user's 4 GB laptop. The
[Diffusers image-to-image API](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)
documents the strength/preservation tradeoff used in the implemented ablation.

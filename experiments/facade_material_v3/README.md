# Learned material-reference facade V3

Branch: `codex/facade-material-controlnet-v3`. Separate from production and from
the V3 rendering ablation. This experiment **trains weights**; it does not
describe reference-guided image-to-image compositing as a learned upgrade.

## Architecture and objectives

- Frozen pinned SD1.5 base; train its existing rank-8 UNet LoRA, initialized
  from the selected V2 step-2000 LoRA rather than a random replacement.
- Extend V2 ControlNet from **7 to 10** conditioning channels: boundary edges,
  six semantic one-hots, then an aligned RGB wall-material reference. Preserve
  all seven old kernels and the bias; zero the three new kernels. Initial
  conditioning-embedding response is unchanged; the reference kernels learn.
- Train a small six-class GroupNorm UNet on **training photos and their human
  labels only**, at 256 resolution. Select it on the fixed validation split.
  Full diffusion training requires mIoU >= .30, window/door union IoU >= .35,
  door IoU >= .10, and real-wall false-opening rate <= .05. The smoke path
  bypasses this quality gate explicitly and cannot produce an installable model.
- Freeze the accepted teacher and VAE. Keep V2's opening-weighted Min-SNR noise
  objective. At sampled timesteps <=250, reconstruct predicted-clean latents,
  decode through the **differentiable frozen VAE decoder**, and add semantic CE,
  window/door probability penalties outside allowed openings, and a small wall
  RGB mean/standard-deviation reference loss. Decoder checkpointing limits
  activation memory. This is inspired by [ControlNet++](https://arxiv.org/abs/2404.07987),
  not a claim of reproducing its exact implementation or published results.
- Teacher feedback is an auxiliary reward. It can be biased or exploited; its
  scores are never labelled independent semantic accuracy. Human review and
  the existing hard opening mask remain deployment requirements.

The four UI architectural styles remain unlabelled. Reference conditioning
does not establish learned recognition of those four classes.

## Data and leakage controls

Reuse locked CMP 400/100/106 splits and the existing human-labelled IRFs audit,
including building/near-duplicate grouping, sparse-training exclusions, source
balancing, original credits and academic licences. No new raw-data upload is
needed. Attach the original Kaggle datasets and completed V2 notebook outputs.

Material references are tiled **wall-only patches** extracted with human masks;
they contain no annotated window or door. If a suitable patch does not fit,
use an explicitly counted median-wall-colour fallback. A crop containing no
wall pixels first falls back to annotated wall pixels in the **same full source
photo**. If that photo also has no labelled wall, preserve its photo/semantics,
mark the reference unavailable and zero the three reference-conditioning
channels. Do not fabricate a material from glass or rewrite the annotations.
Such samples contribute no material-matching loss. An invalid independent
reference never falls back to the validation target's material.

Before GPU optimization, audit every training and validation sample at epoch
zero. Save fallback counts and sample IDs in `material-reference-audit.json`,
track the report in MLflow and include it in the bundle. Cache the audit only
when the validated dataset manifest, loader code and crop settings match.
The test split is never read by this audit.

On 20% of training
attempts, extract a genuine annotated opening-free square of at least 64 native
pixels, when one exists. Never remove an opening from a label and leave its
photograph behind. Wall-only crops can be upsampled; they teach absence of
openings and are not evidence of native high-resolution texture supervision.
Existing facade/storey strip augmentation remains active on ordinary examples.

Validation uses the fixed V2 stratified 12 CMP + 4 IRF subset and up to two
real wall-only crops per source. It reports two distinct protocols:

1. **Target-wall-reference assisted:** extract the material reference from the
   target validation photo. LPIPS/SSIM here have extra appearance information
   and must not be compared directly to published layout-only V2 scores.
2. **Independent training reference:** condition on a wall patch from a different
   training building. Report teacher opening consistency and reference-colour
   agreement. Do not compute paired LPIPS against a target with a deliberately
   different requested material.

Opening IoU/FPR on blank crops is reported separately. Selection combines
source-macro teacher geometry diagnostics with .25 assisted LPIPS. Step 0 is
the V2 initialization under the **same V3 sampling protocol**; keep it if no
trained checkpoint improves the selection score. No test split is used for
checkpoint selection. Final independent test and multi-house visual review
must follow training before any production promotion.

## Training and tracking

First run smoke: two teacher optimizer steps and two diffusion optimizer steps,
full low-noise feedback gradient path, validation, checkpoint, export, checksum
validation, and offloaded inference in a fresh process. This tests execution,
not quality. Full mode: at most 25 teacher epochs and 1500 diffusion optimizer
steps, physical batch 1 / effective batch 8, FP16, gradient checkpointing and
8-bit Adam. One Kaggle GPU is used; two T4 memories are not pooled. The laptop's
3500 MiB budget applies to **offloaded inference**, not Kaggle training.

Two MLflow experiments retain separate histories:

- `ArchiGen-Facade-V3-Semantic-Teacher`
- `ArchiGen-Facade-Material10-Semantic-Feedback-V3`

Both record immutable Git SHA, dataset digest, parent/teacher checksums, config,
metrics and validation artifacts. Without remote tracking secrets, Kaggle writes
SQLite plus portable event logs. The installer replays both histories into the
local MLflow server/database; it never transplants Kaggle artifact paths.

```powershell
python -m experiments.facade_material_v3.manage stage-kernel --output tmp/v3-smoke --mode smoke
# From tmp/v3-smoke, with Kaggle authenticated locally:
kaggle kernels push --path . --accelerator NvidiaTeslaT4
```

Staging `--mode train` creates one **staged** notebook: smoke runs first in
`facade-material-v3-smoke`, and full training starts only after successful smoke
execution, checksum validation and offloaded reload. It then trains a fresh
teacher and applies the declared quality gate. Smoke and full outputs and ZIPs
have distinct names; neither overwrites the other. This allows one queued GPU
allocation to complete the workflow without another manual start.

Notebooks are private,
pin a full commit, and attach the existing private CMP dataset, original IRFs
and completed V2 output. No credentials belong in the notebook or Git.

## Laptop integration after evaluation

Download `archigen-facade-material-v3-model-and-mlflow.zip`, then:

```powershell
python -m experiments.facade_material_v3.manage install --archive PATH.zip --destination NEW_MODEL_FOLDER --tracking-uri LOCAL_MLFLOW_URI
```

The installer verifies both weight checksums and architecture, rejects smoke
artifacts/path traversal, and refuses to overwrite existing folders. Use the
separate diffusion environment:

```python
from experiments.facade_material_v3.runtime import MaterialFacadeRuntime
runtime = MaterialFacadeRuntime('NEW_MODEL_FOLDER/export')
image = runtime.generate_material(semantic7, aligned_material_reference_512)
```

The pinned base is cached locally; FP16/sequential CPU offload and the existing
RAM/VRAM checks are retained. The teacher is **not needed at inference**. The
existing production bridge/UI default is untouched. The new weights still
need actual RTX3050 profiling and multi-house tests; successful Kaggle export
alone does not prove laptop compatibility or better visual quality.

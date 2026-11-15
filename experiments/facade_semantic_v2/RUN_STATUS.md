# V2 execution record — 15 November 2026

The full 2000-step experiment and 210-image matched validation have completed.
V2 improves perceptual similarity, with mixed structural results. Production
remains unchanged; the multi-side house test is recorded separately.

## Verified smoke run

- Notebook: [semantic7 LoRA V2 smoke](https://www.kaggle.com/code/khalilayeb/archigen-facade-semantic7-lora-v2-smoke), saved version 2, COMPLETE.
- Source commit: `354d2b8c4a5573196f2de19a6ff1d4ff4b3ad57c`.
- MLflow training run: `8701677bfbe548dab7ebb10e6e987c6c`.
- Imported into the existing local facade experiment database as
  `7e29f82d6bd14ff5985cbe66b20ddc95`, including portable artifacts and metric history.
- Two real optimizer steps, finite losses, validation, resumable checkpoint,
  paired ControlNet/LoRA export, and fresh-process offloaded reload inference passed.
- Smoke training peak allocated: **6444.88 MiB** on Kaggle; this training profile
  requires more than the laptop's 4 GB VRAM.
- Smoke offloaded inference peak allocated: **911.03 MiB**, reserved **1010 MiB**,
  **4.00 seconds with only two denoising steps** on Kaggle. This is neither a
  25-step speed measurement nor an RTX 3050 benchmark or quality result.
- Data audit and provenance are retained in [results](results/).

## Audited data

| Source | Original train membership | Effective training | Validation | Unused test |
|---|---:|---:|---:|---:|
| CMP | 400 | 399 | 100 | 106 |
| Original IRFs | 825 | 825 | 110 | 104 |
| Total | 1225 | **1224** | **210** | **210** |

CMP `train/236.jpg` is retained in its locked split but its sparse mask is
excluded from optimization. IRFs rejected: 18 (one exact duplicate, five
photo/JSON mismatches, twelve insufficient-resolution photos). The 1039 retained
IRFs photos belong to 949 grouped building identities. Grouping/deduplication
limitations are described in the README. No four-style labels were invented.

## Full experiment

- [Kaggle training notebook](https://www.kaggle.com/code/khalilayeb/archigen-facade-semantic7-lora-v2-train), saved version 1.
- Pinned source: `5d9d1e8a988e49e4fbb11e03518603efa303e415`.
- Full-run data audit reproduced the smoke counts. Optimizer steps 1–14 were
  observed with finite losses and no runtime failure; peak allocated VRAM was
  6451.05 MiB. This is an early execution check, not a quality evaluation.
- Training completed all 2000 steps and selected step 2000. Training MLflow run:
  `7d9d80627e1e4c75b39bd7bbd1162353`. Imported locally as
  `c9141e312f6f4a3cbb3cd81d5aac754b`; matched evaluation imported as
  `e7e6f18a9eaa4090a8de8f7f1721bb30`.
- Config: 2000 optimizer steps; physical batch 1, accumulation 4; semantic7
  ControlNet + rank8 UNet LoRA; fixed base and parent weight hashes.
- Best model selected every 500 steps using source-macro LPIPS on fixed 12 CMP
  and 4 IRF validation examples. Full matched V1/V2 evaluation runs afterward on
  all 210 validation images. Both test sets remain unused.
- MLflow SQLite, portable metric history/artifacts and a checksum-protected
  local model bundle are written to saved Kaggle output. Download the bundle
  after completion; do not install the smoke model.
- Local 25-step inference and multi-side house review are complete; see
  [house test](HOUSE_TEST.md). The model is not promoted to production.

## Completed matched validation

See [full results](results/full-validation-v1-v2.json). Both models use identical
new preprocessing, prompts, sampling settings and per-image seeds. These are
validation results; the selection subset overlaps this set, and final tests
remain unused.

| Source | n | V1 LPIPS ↓ | V2 LPIPS ↓ | Reduction | V2 LPIPS wins |
|---|---:|---:|---:|---:|---:|
| CMP | 100 | 0.575715 | 0.536236 | 6.86% | 78 |
| IRFs | 110 | 0.686063 | 0.637672 | 7.05% | 85 |

Paired bootstrap 95% intervals for V2-minus-V1 LPIPS are entirely below zero
on both sources. CMP SSIM falls from 0.181844 to 0.154960; IRFs SSIM rises from
0.154797 to 0.167443. Opening-edge MAE increases slightly on both sources, so
these results do not establish improved opening fidelity. Local model ZIP,
ControlNet and LoRA checksums have been verified against the selected export.

## Code checks

- General suite: **115 passed, 15 skipped**, no failures. Skips reflect missing
  optional diffusion dependencies and model/data assets in the isolated worktree.
- Dedicated facade contracts in the diffusion environment: **20 passed**,
  including actual ControlNet/LoRA gradients, frozen base weights, adapter
  save/reload, seven-channel pipeline inference, dataset audit and split guards.
- [GitHub CI](https://github.com/ayebkalil/archigen-3d/actions/runs/43687049008)
  is also run on the experiment branch; consult the branch checks for the latest status.

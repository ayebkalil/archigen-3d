# Submission status — 16 November 2026

**Version 2 submitted with a data-loader fix; currently awaiting GPU allocation.**

Version 1 did execute. Its GPU smoke passed two optimizer steps, 16 decoded
feedback microbatches, export/checksum validation and fresh-process offloaded
reload. The full teacher then failed before completing its first epoch with
`ValueError: No annotated wall pixels for material reference`. Full facade
fine-tuning therefore never started. This was a loader failure, not a failed
teacher quality gate or a CUDA out-of-memory error.

The loader now uses a wall reference from the same original photo when a crop
has none. If the original annotation also contains no wall, it keeps the
sample, marks the reference unavailable, zeros reference channels and excludes
material loss. Independent references never borrow the target's material.
A complete train/validation sample audit now runs before optimization.

The real local CMP audit passed **399 eligible training + 100 validation
images**, including `CMP/train/242.jpg`, whose reference is unavailable.
No test image was opened. The full CMP+IRFs audit will run on Kaggle.
Evidence: [local audit](results/local-cmp-reference-audit.json) and
[version-1 failure record](results/kaggle-version1-failure.json).

## Current submission

- Notebook: <https://www.kaggle.com/code/khalilayeb/archigen-facade-material10-feedback-v3-train>
- Private version 2, source pinned to
  `e02099d6815424959beae6a195eb8e4522cb0e56` on
  `codex/facade-material-controlnet-v3`.
- Requested GPU T4 x2; code uses one GPU. Runtime limit: six hours after
  allocation. Free quota was checked and pay-to-scale is disabled.
- Two earlier smoke-only queued versions were cancelled through Kaggle's
  active-events controls. The replacement is one staged workflow, not parallel
  full training jobs.
- Execution order: isolated two-step smoke → checksum/export/offloaded-reload
  checks → fresh full teacher training/validation gate → diffusion fine-tuning
  → selected model export and portable MLflow bundle.
- Failures in smoke or teacher quality validation stop the remaining workflow.
  The existing production model is never automatically replaced.

## Verified locally

- **130 passed, 19 skipped** in the experiment worktree's complete Python suite.
  Skips are optional dependency/model/data checks unavailable in this checkout.
- **14 passed** in the diffusion environment's V3-specific suite, including
  actual material10 inference/export/reload, initial-response preservation,
  nonzero gradients through a real frozen VAE decoder, protected training
  splits, genuine wall-only crops, and failed-preflight execution gating.
- Source compiles; no whitespace errors. Both production backends remain
  healthy. No production frontend, API default or existing model weights changed.

## Pending evidence

Version 2 has not yet produced execution logs or new training/quality results.
Version 1's smoke results establish execution only, not improved facade quality.
MLflow teacher/training run IDs will
be created when execution starts; portable logs are included in the output
bundles for later local import. A queued submission is not a successful GPU
smoke test and does not establish improved results.

After completion, inspect teacher gates, compare selection against step 0,
review held-out buildings and blank-wall cases, then profile the export on the
RTX3050 and test multiple house elevations. Only those checks can justify any
production promotion.

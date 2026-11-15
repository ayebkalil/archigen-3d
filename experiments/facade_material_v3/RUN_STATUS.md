# Submission status — 15 November 2026

**Submitted; awaiting Kaggle GPU allocation. Training has not started.**

- Notebook: <https://www.kaggle.com/code/khalilayeb/archigen-facade-material10-feedback-v3-train>
- Private version 1, source pinned to
  `47da048a98612267994b94d37d5f3c42fa4ae3bb` on
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

- **126 passed, 19 skipped** in the experiment worktree's complete Python suite.
  Skips are optional dependency/model/data checks unavailable in this checkout.
- **10 passed** in the diffusion environment's V3-specific suite, including
  actual material10 inference/export/reload, initial-response preservation,
  nonzero gradients through a real frozen VAE decoder, protected training
  splits, genuine wall-only crops, and failed-preflight execution gating.
- Source compiles; no whitespace errors. Both production backends remain
  healthy. No production frontend, API default or existing model weights changed.

## Pending evidence

Kaggle has produced no execution logs, teacher scores, training losses, new
quality metrics or trained V3 weights yet. MLflow teacher/training run IDs will
be created when execution starts; portable logs are included in the output
bundles for later local import. A queued submission is not a successful GPU
smoke test and does not establish improved results.

After completion, inspect teacher gates, compare selection against step 0,
review held-out buildings and blank-wall cases, then profile the export on the
RTX3050 and test multiple house elevations. Only those checks can justify any
production promotion.

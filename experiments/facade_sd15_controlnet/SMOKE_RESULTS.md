# Kaggle GPU smoke verification — 14 November 2026

This verifies execution and tracking, **not facade image quality**. No deployable
model is produced by a two-step smoke run.

- Source: `f6a9bd4e7ab35cdf05e77bc7f0f3b47389822ff0` on
  `codex/facade-controlnet-kaggle-mlflow`.
- [Private Kaggle smoke notebook, version 2](https://www.kaggle.com/code/khalilayeb/archigen-facade-sd15-controlnet-smoke).
  Use the notebook's version history if the link opens its latest version.
- Kaggle execution status: `COMPLETE`; portable MLflow status: `FINISHED`.
- Kaggle MLflow run ID: `6097abb9846f413b9f2301044b92a975`.
- Hardware: one Tesla T4 of the T4 x2 allocation; GPUs are not combined.
- Runtime: PyTorch `2.11.0+cu130`, FP16 mixed precision, gradient checkpointing,
  8-bit AdamW, physical batch 1, accumulation 4.
- Dataset: original CMP photographs and indexed annotations, locked train/val/test
  split `400/100/106`; no four-style supervision labels.

| Effective optimizer step | Mean microbatch loss | Elapsed training seconds | Peak PyTorch allocated MiB |
| --- | --- | --- | --- |
| 1 | 0.2806389215 | 5.8484 | 5655.6948 |
| 2 | 0.1552806576 | 8.1605 | 6370.3384 |

Elapsed training time excludes source checkout, dependencies, downloads and model
loading. Two noisy diffusion-loss observations do not establish convergence.
The smoke learning-rate schedule is deliberately only two steps and differs from
the full 1,500-step experiment.

Downloaded portable events were replayed successfully into a separate local
MLflow SQLite database: one finished run, two history entries for each of loss,
learning rate, elapsed time and memory. Existing production tracking was untouched.

The first smoke attempt failed before any training step because MLflow 3.1.4 was
incompatible with SQLAlchemy 2.1. SQLAlchemy is now pinned to 2.0.43. Bitsandbytes
was also upgraded to 0.50.2 for Kaggle's CUDA 13 runtime. The corrected smoke passed.

Verification also completed: 116 passed / 12 skipped across the local Python
suites; the tiny real ControlNet backpropagation check passed separately in the
diffusion environment. [GitHub contract checks passed for the tested source](https://github.com/ayebkalil/archigen-3d/actions/runs/43615365968).
Skipped local checks reflect unavailable optional assets/dependencies in the
isolated checkout; this does not mean every production model was exercised.

The [full training notebook](https://www.kaggle.com/code/khalilayeb/archigen-facade-sd15-controlnet-mlflow)
was submitted as version 3, pinned to the same source. Consult Kaggle for live
status. Validation quality, production-baseline comparisons and laptop inference
latency/VRAM remain unmeasured until the corresponding runs finish. The local
3,500 MiB inference target is separate from the training measurements above.

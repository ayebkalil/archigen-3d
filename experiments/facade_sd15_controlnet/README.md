# Kaggle facade experiment → MLflow → local ArchiGen

Branch: `codex/facade-controlnet-kaggle-mlflow`, based on production commit `4069fe7`.
The current studio worktree is separate. No existing facade weights are overwritten.

The notebook fine-tunes a **ControlNet initialized from SD1.5 Canny ControlNet** on
CMP semantic boundaries and original facade photographs. SD1.5 UNet, CLIP and VAE
remain frozen. The denoising loss emphasizes latent locations covering windows and
doors. This is an experimental candidate: a stronger pretrained prior does not
establish better results without evaluation, and neither ControlNet nor the UV
mapping guarantees the generated opening locations.

The pinned models are in `config.json`. Training uses 512×512, physical batch 1,
accumulation 4, FP16 AMP, gradient checkpointing and 8-bit Adam. One T4 is used even
if Kaggle assigns two. The default 1500 optimizer steps are a starting experiment,
not a promised duration or optimal training schedule.

## Data and provenance

Use the existing `data/processed/cmp512/manifest.json`, produced by
`scripts/prepare_cmp512.py`. It maps the original 606 CMP photos/index masks to
the project's locked **400 train / 100 validation / 106 test** split. Packaging
verifies source SHA256s, mask dimensions and duplicate identities. It copies the
original photos without recompression and records their native resolutions.
Some originals have an axis below 512; resizing does not add real detail.

CMP has no labels for Modern, Mediterranean, Brick and Lush. No labels are
fabricated. Local style prompts use the frozen pretrained text prior; these are
not four styles learned from CMP. Dataset attribution and its CC BY-SA notice are
included. Raw data and weights stay out of Git.

Prepare the private Kaggle dataset (run from a checkout with the local sources,
or pass the source checkout's absolute path):

```powershell
python -m experiments.facade_sd15_controlnet.manage prepare-data --manifest data/processed/cmp512/manifest.json --source-root . --output outputs/kaggle-cmp --baseline models/saved/generator_perceptual_ema.onnx
pip install kaggle==2.2.5 six
kaggle auth login
Push-Location outputs/kaggle-cmp
kaggle datasets create --path . --dir-mode zip
Pop-Location
```

The `--baseline` option packages the existing ONNX generator for paired evaluation.
The dataset is private by default; do not use `--public`. Passing `.` from its
folder also avoids a Kaggle CLI 2.2.5 upload-state bug with Windows forward-slash paths.

## GitHub ↔ Kaggle notebook

Owner: `khalilayeb`. Training notebook:
`experiments/facade_sd15_controlnet/train_facade_controlnet.ipynb`.

After committing and pushing the experiment code:

```powershell
python -m experiments.facade_sd15_controlnet.manage stage-kernel --output outputs/kaggle-kernel --mode smoke
kaggle kernels push --path outputs/kaggle-kernel --accelerator NvidiaTeslaT4
kaggle kernels status khalilayeb/archigen-facade-sd15-controlnet-smoke
```

The staged notebook pins the exact Git commit. On Kaggle it clones/checks out that
commit, installs pinned Python libraries while retaining Kaggle's CUDA torch build,
verifies the input dataset and logs the commit/config/data hashes. The smoke mode
runs two real optimizer steps but exports no deployable weights.

Once the smoke run passes, stage `--mode train`; its private notebook ID is
`khalilayeb/archigen-facade-sd15-controlnet-mlflow`. Use `--no-run` to publish a
notebook without consuming GPU time. Internet and GPU availability depend on the
account; inspect the saved notebook settings if a version cannot download models.

The optional GitHub Actions workflow `Facade ControlNet experiment` runs contract
tests on branch pushes. Its manual action can publish-only, smoke or train, using
the repository secret **KAGGLE_API_TOKEN**, configured by the account owner.
No token is stored in the notebook, code or artifacts. This uses reproducible Git
checkout and Kaggle API publishing; it does not require granting Kaggle access to
the entire GitHub account. A GitHub push does not automatically spend GPU quota.

## Tracking, checkpointing and evaluation

MLflow logs config, original split counts, source commit, GPU, losses, learning
rates, elapsed time, peak PyTorch allocations, fixed validation metrics/images and
the model manifest. Checkpoints include optimizer, scheduler, scaler/RNG state,
epoch and batch position. Order and flips are deterministic on resume. Copy the
**complete previous run output**, including `best-controlnet`, before resuming a
checkpoint with the same config and dataset; resumed MLflow segments link to the
previous run ID. Keep two optimizer checkpoints to bound storage.

By default tracking uses `/kaggle/working/facade-controlnet/mlflow.db`. Kaggle
absolute artifact paths are not portable, so every run also writes
`tracking-events.jsonl` and `portable-artifacts/`. The installer replays these into
the PC's existing MLflow store, retaining metric histories and source run tags.
Repeated imports are idempotent. Optional **live remote** MLflow uses Kaggle
Secrets `MLFLOW_TRACKING_URI` and required authentication variables. Do not use a
PC's `localhost` URI from Kaggle; no public tunnel is required by the default path.

Checkpoints are selected by LPIPS on four fixed validation images. The final
notebook then evaluates the selected export on all 100 validation images, logging
LPIPS, SSIM, PSNR, opening-edge MAE and diffusion time. If the production ONNX was
packaged, it produces raw-256-baseline comparisons and paired LPIPS bootstrap CIs.
Opening-edge MAE is a proxy, not opening segmentation IoU. Validation was used in
selection; do not call it independent test accuracy. After selection, a separate
final test can be run explicitly:

```bash
python -m experiments.facade_sd15_controlnet.evaluate --model /kaggle/working/facade-controlnet/export --data DATA_ROOT --output /kaggle/working/final-test --split test --final-test
```

No new KID/FID or four-style benchmark numbers are invented. Compare results and
inspect facade geometry before considering promotion. A 512px image alone does not
prove photorealism.

## Download and run locally

The notebook produces `archigen-facade-model-and-mlflow.zip`, containing the selected
FP16 ControlNet, config/checksums, source/run IDs, tracking events and validation
artifacts. The larger optimizer checkpoints remain in saved Kaggle output for
resume; the small deployment archive excludes those and the frozen base weights.

```powershell
python -m experiments.facade_sd15_controlnet.manage install --archive C:/path/archigen-facade-model-and-mlflow.zip --destination models/saved/facade-controlnet-kaggle --tracking-uri sqlite:///mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Install diffusion dependencies in a **separate environment**, with a CUDA-enabled
torch/torchvision pair appropriate for the machine. An existing compatible
`.venv-facade-diffusion` can be reused; installing these packages into the normal
FastAPI environment is unnecessary.

```powershell
.venv-facade-diffusion/Scripts/python.exe -m pip install -r experiments/facade_sd15_controlnet/requirements.txt
$env:ARCHIGEN_CONTROLNET_MODEL=(Resolve-Path models/saved/facade-controlnet-kaggle/export).Path
$env:ARCHIGEN_CONTROLNET_ALLOW_DOWNLOAD='1'
.venv-facade-diffusion/Scripts/python.exe -m uvicorn experiments.facade_sd15_controlnet.service:app --host 127.0.0.1 --port 8002
```

The first local run needs the pinned frozen SD1.5 base (download once, or transfer
a complete Hugging Face cache). ControlNet weights alone are not a standalone SD
model. Set `ARCHIGEN_CONTROLNET_ALLOW_DOWNLOAD=0` for subsequent offline use.
The worker loads models lazily, uses sequential CPU offload, FP16, SDPA, VAE tiling
and slicing, checks available RAM/GPU headroom and sets a PyTorch allocation limit.
The 3500 MiB device target must be profiled on the actual laptop; allocator peaks
and a device reading after inference do not certify the true device-wide peak.
Other GPU processes can still change memory availability. No unmeasured latency
or VRAM guarantee is claimed.

In a separate terminal for an **experimental backend checkout**:

```powershell
$env:ARCHIGEN_FACADE_CONTROLNET_URL='http://127.0.0.1:8002'
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The backend keeps its normal API/UV contract and calls the worker only for an
explicit facade generation request. Floorplan uploads do not invoke diffusion.
The generation endpoint waits in a thread so a slow worker does not block the
ASGI event loop. Selected experimental service failures are reported, not silently
replaced by old textures. Unset `ARCHIGEN_FACADE_CONTROLNET_URL` to use production
Pix2Pix. Existing models remain available for rollback.

## Verification

```powershell
python -m pytest tests/ backend/tests/ -q
python -m experiments.facade_sd15_controlnet.manage check-data --data outputs/kaggle-cmp
```

Tests include split/checksum protection, identical bridge/training conditioning,
MLflow event import/history/idempotence, notebook source pinning, archive integrity,
opening-weighted loss and actual tiny-model backpropagation through a frozen UNet
(the latter requires the isolated diffusion dependencies). These checks do not
substitute for a completed Kaggle training run or a measured local deployment.

References: [Diffusers ControlNet training](https://huggingface.co/docs/diffusers/training/controlnet),
[memory offloading](https://huggingface.co/docs/diffusers/optimization/memory),
[MLflow local databases](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/),
[CMP source](https://cmp.felk.cvut.cz/~tylecr1/facade/).

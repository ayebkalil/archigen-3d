# Model distribution audit and teammate setup

Audit date: 14 November 2026. All 15 model files referenced by DVC on the
development PC matched their recorded MD5 and sizes. We cannot inspect the
teammate's machine, but confirmed the following repository defects:

| Defect | Consequence | Fix |
| --- | --- | --- |
| Default DVC remote points to `C:\Users\ayebk\OneDrive\archigen_dvc_remote_backup` | Another PC cannot retrieve weights from this local folder | Production ONNX pointers import versioned public HTTPS assets |
| Model binaries correctly ignored by Git | Git clone alone contains no trained models | Explicit restore command and SHA-256 inventory |
| Previous clean-clone test uses this PC's path and old branch | Test can succeed using storage teammates cannot access | New isolated clone/cache verification |
| `/health` always reports loaded | Missing models appear healthy | HTTP 503 and actual component readiness |
| Facade failures return the sketch/upload with HTTP 200 | Placeholder mistaken for neural output | Clear model-unavailable errors |
| Failed PyTorch load can leave an initialized model assigned | Potential inference using incomplete weights | Assign model only after strict loading succeeds |
| Docker forces experimental 512 facade | Different default from native EMA model | Experiment is opt-in |
| Missing native checkpoints select older/challenger files if present | Different machines can load different model generations | Use the champion's ONNX version instead; historical models require explicit selection |
| TripoSR source/checkpoint and inherited venv are machine-local | Furniture may silently use procedural geometry | Pinned bootstrap, complete dependency list, no silent substitute |

## Production models: native or Docker

Run from the repository root in the backend Python environment:

```powershell
python -m pip install -r backend/requirements.txt dvc
python scripts/models.py pull
python scripts/models.py check --json
```

The three ONNX files total 365,290,570 bytes (about 348 MiB). Their DVC import
pointers fetch from the [versioned weights release](https://github.com/ayebkalil/archigen-3d/releases/tag/model-weights-v1)
without private paths or credentials. `models/artifacts.json` pins SHA-256, MD5,
sizes and source lineage. A failed check exits nonzero. An explicit `pull` repairs
missing or corrupt workspace files.

Equivalent targeted DVC command:

```powershell
python -m dvc pull models/saved/floorplan_unetpp/floorplan_multitask.onnx.dvc models/saved/floorplan_unetpp/floorplan_fallback.onnx.dvc models/saved/generator_perceptual_ema.onnx.dvc
```

Avoid bare `dvc pull` for runtime setup: it also requests datasets and research
checkpoints. These remain separate training artifacts; configure a shared remote
with `dvc remote add --local -d training-data <shared-location>` if needed.
Historical checkpoints retain inventory hashes but are not included in this runtime
release. No training dataset download is needed to run the application.

For the same backend regardless of locally installed PyTorch packages:

```powershell
$env:ARCHIGEN_FLOORPLAN_BACKEND='onnx'
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Check `/health` and `/model/metadata`. Readiness requires the champion, fallback
and facade generator. Metadata reports the files actually selected and their hashes.
Deliberately selecting different CPU/GPU backends can introduce small numeric differences.

Restore weights **before** `docker compose up --build`. The container mounts
`models/saved` read-only and validates its production checksums. Rebuilding an
image cannot populate an empty host model folder. `ARCHIGEN_FACADE_EXPERIMENT` is
empty by default; opt-in experimental weights and their JSON sidecar are separate.

## Optional Pix2Pix 512 experiment

The three production models above are sufficient for the default application.
`facade512_experiment/generator.onnx` is a separate **experimental residual-FiLM
512 generator**, not Stable Diffusion/ControlNet. It is not promoted to production
and does not have supervised labels for the four architectural styles.

Its earlier DVC pointers relied on the developer's local training remote, so
teammates could not restore it from the production release. The separate
[experimental release](https://github.com/ayebkalil/archigen-3d/releases/tag/facade512-weights-v1)
publishes `generator.onnx`, its required `generator.json`, optional `generator.pth`,
and `SHA256SUMS.txt`. The inventory pins exact sizes and SHA-256 values for the
weights **and JSON**. Metadata uses LF line endings on every platform.

```powershell
git pull
python scripts/models.py pull --profile facade512
python scripts/models.py check --profile facade512 --json
$env:ARCHIGEN_FACADE_EXPERIMENT='models/saved/facade512_experiment/generator.onnx'
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Run these in the backend Python environment, with DVC installed as above.
The optional ONNX profile needs no CUDA/PyTorch installation. The sidecar is
already tracked in Git; `pull` also repairs a missing/corrupt sidecar from the
release. ONNX loading verifies both pinned files and rejects incorrect metadata.

For native PyTorch weights instead, use `--profile facade512-torch`, install the
existing `requirements-facade512.txt`, and select the `.pth` path explicitly.
Leave `ARCHIGEN_FACADE_EXPERIMENT` unset (PowerShell: `Remove-Item
Env:ARCHIGEN_FACADE_EXPERIMENT -ErrorAction SilentlyContinue`) for the production
EMA facade. Docker remains on production unless that variable is explicitly set.
See [experiment results and limitations](facade512-experiment.md).

Clean-machine verification of the optional ONNX profile:

```powershell
python scripts/verify_team_install.py --ref main --profile facade512
```

## Neural furniture on a clean NVIDIA CUDA machine

```powershell
python scripts/setup_furniture.py pull
python scripts/setup_furniture.py check
python -m venv .venv-furniture-quality
.\.venv-furniture-quality\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv-furniture-quality\Scripts\python.exe -m pip install -r requirements-furniture-quality.txt
.\.venv-furniture-quality\Scripts\python.exe -m uvicorn backend.app.furniture_worker:app --host 127.0.0.1 --port 8001
```

The bootstrap obtains the pinned upstream TripoSR source, config and approximately
1.6 GiB checkpoint, verifies hashes and applies the optional marching-cubes import
patch. It preserves existing checkouts and rejects a different revision. Source and
Python dependencies are separate setup steps. Use an NVIDIA driver compatible with
the CUDA wheel and adequate free GPU/RAM. First use also caches DINO configuration
and rembg background-removal weights from upstream.

In the native backend shell, set
`ARCHIGEN_FURNITURE_SERVICE_URL=http://127.0.0.1:8001` before starting the backend.
Docker Desktop uses `http://host.docker.internal:8001`; configure a reachable worker
URL for other container hosts. Check the worker's `/health` separately.

Missing neural inference returns an error. The developer-only
`ARCHIGEN_ALLOW_PROCEDURAL_FURNITURE=1` opts into the old approximation explicitly;
it does not provide TripoSR quality.

## Verification

```powershell
python scripts/verify_team_install.py --ref main
python -m pytest tests/ backend/tests/ -q
```

The first command clones into `tmp`, starts without project cache/local DVC
configuration, downloads the production models, checks SHA-256 and executes all
three real ONNX graphs. CI also restores production weights before Python tests.
Regression tests cover missing/corrupt weights, false readiness and silent
facade/furniture substitutes.

Correct weights fix deployment discrepancies; they do not establish the sample's
reconstruction quality on every unseen floorplan.

# Furniture reconstruction tracking with MLflow

The experiment **furniture-reconstruction** records inference, separately from
the floorplan and facade training experiments. Tracking uses MLflow 3.17.1.

## Install and start

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tracking.txt
.\scripts\setup_furniture_gpu.ps1
.\scripts\start_native_gpu.ps1
```

The native launcher starts or reuses this checkout's MLflow server before starting
the worker and API. Open **http://127.0.0.1:5000**, select **furniture-reconstruction**,
select runs, and use **Compare** to inspect parameters and metrics. The Artifacts
tab provides the input image, processed preview, model GLB, and metadata JSON.

You can also start MLflow independently:

```powershell
.\scripts\start_mlflow.ps1
.\scripts\start_mlflow.ps1 -Restart
```

The server binds only to 127.0.0.1:5000. Metadata persists in the root `mlflow.db`;
artifacts persist in `mlartifacts/`, uploaded through the server. Both paths are
absolute and ignored by Git. Existing experiments are preserved. Logs and the CLI
process ID are in `outputs/mlflow/`. Unrelated services on port 5000 are rejected.

## What each generation records

The shared reconstruction service creates one request-scoped run using
`MlflowClient`; the forwarding API creates no duplicate run. Runs identify their
result using the `furniture_id` tag. Public response formats and the studio UI are
unchanged. The inference environments install only `mlflow-skinny`; the full server
package is listed in `requirements-tracking.txt`.

- Parameters: actual engine, CPU/CUDA, GPU model and CUDA runtime when applicable,
  category, requested height, background removal, preset, resolutions, precision,
  image and checkpoint hashes, Python/library versions, Git revision and dirty flag.
- Metrics: reconstruction duration, vertices/faces, output dimensions, GLB bytes,
  and, when available, texture coverage, surface components/watertightness, sampled
  device memory and PyTorch peak allocated/reserved memory in MiB.
- Artifacts: `input/input.png`, `output/preview.png`, `output/model.glb`,
  `output/metadata.json`, or `diagnostics/error.txt` for a failed reconstruction.

`generation_seconds` times the generator call, including mesh and GLB creation;
it excludes model startup and all MLflow work. CPU runs have no invented GPU
metrics. Checkpoint hashing is cached per file version. Git provenance is captured
once per process, so restart the stack after changing code. A dirty revision is
reported explicitly rather than treated as a reproducible committed revision.

Reconstruction errors retain their original exception and mark the run FAILED.
Tracking errors emit a warning and never discard a successful model. If MLflow is
unavailable, some runs or artifacts may be missing; there is no offline queue or
automatic backfill. The native stack uses a 3-second request timeout and no HTTP
retries for MLflow. Uploaded furniture photos and models remain in the local
MLflow artifact store, independently of the user's furniture catalog.

Tracking is enabled by the native launcher. To run it without tracking:

```powershell
$env:ARCHIGEN_MLFLOW_ENABLED='0'
.\scripts\start_native_gpu.ps1
```

For custom process launchers, set `ARCHIGEN_MLFLOW_ENABLED=1` and
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in the process doing the reconstruction.
Outside the native launcher, tracking is opt-in. The Docker launcher is unchanged;
this tracking-server integration targets the native stack.

## Reproduce the before/after comparison

With the tracked native high-quality stack running:

```powershell
.\.venv-furniture-quality\Scripts\python.exe scripts/benchmark_furniture_migration.py
```

The script reconstructs the same reference chair at a requested height of 0.85 m
with background removal enabled. It forces the CPU volumetric engine, even when
TripoSR is installed, and reuses the live GPU worker instead of loading a second
neural model. Each configuration has one warmup and five measured generations.

The parent run has `run_kind=migration_comparison`. Its children have
`run_kind=benchmark_warmup` or `benchmark_measurement`, a configuration tag, and an
iteration index. Warmups are excluded from median/minimum/maximum summaries.
Per-generation inputs, metadata and GLBs remain attached to the child runs.
The parent stores summary metrics and a JSON report, also saved under
`outputs/furniture_migration/` using the parent run ID as its filename.

The CPU/GPU duration ratio compares **different algorithms and hardware**. It is
not an isolated GPU speedup and says nothing about geometric accuracy. Face count
and texture resolution describe output density, not fidelity to a ground-truth
scan. Existing measurements from earlier sessions are not imported as new runs.

## Verify tracking end to end

```powershell
.\.venv-furniture-quality\Scripts\python.exe scripts/verify_furniture_gpu.py --check-mlflow --base-url http://127.0.0.1:3000 --output outputs/furniture_tracking/studio
```

This sends reference and opaque-photo requests through the frontend proxy, checks
that each creates exactly one FINISHED CUDA run, downloads its GLB through MLflow,
and compares the bytes with the API output. The unit suite additionally exercises
failure logging, concurrent runs, disabled tracking, and MLflow outages.

## Verified local comparison (2026-11-13)

Experiment ID: `1`; comparison parent: `785f824d5e084eb79ff4ccb5334d5f8a`.
The full comparison completed with 10 measured child runs and 2 excluded warmups.

| Configuration | Median | Minimum | Maximum |
| --- | --- | --- | --- |
| Volumetric CPU | 0.243 s | 0.228 s | 0.256 s |
| TripoSR RTX 4050, high | 6.748 s | 6.588 s | 6.940 s |

The median CPU/GPU duration ratio is 0.036. TripoSR takes longer for this input;
the two engines perform different computations and produce different geometry.
These measurements do not establish a hardware-only speedup or geometric accuracy.
The runs record their dirty Git state explicitly.

All comparison children contain the expected input, GLB, preview and metadata
artifacts. Studio-proxy verification produced one FINISHED run per request and
downloaded byte-identical GLBs from MLflow. The 32 targeted tests passed.
All 16 runs present at the persistence check, their parameters/metrics/tags, and
experiment definitions were unchanged after restarting the MLflow server; the
comparison report was downloadable afterward. One initial verification run is
retained as FAILED with a tracking-error tag for the Windows Unicode issue found
and fixed during integration; it is excluded from the comparison.

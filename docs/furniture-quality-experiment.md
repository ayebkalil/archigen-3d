# Furniture reconstruction quality experiment

Branch: `experiment/furniture-reconstruction-quality`, based on facade experiment
commit `791d2b3`. No training or checkpoint replacement was performed. This
experiment recovers the quality of the existing TripoSR neural reconstruction
that the previous extraction and texture projection discarded.

## Changes

- Keep soft alpha and thin parts, crop foreground before the 512 px encoder input,
  and use CPU background removal for opaque photos. The studio now offers an actual
  chair reference instead of flat SVG icons, which Pillow could not decode as photos.
- Extract a canonical XYZ density volume in two-slab batches with 8192-point GPU
  queries. skimage marching cubes returns XYZ directly. The local vendor fallback
  previously swapped axes before querying neural colors; two later swaps happened
  to restore the geometry orientation, but not the field color queries.
- Convert canonical `(X camera-facing, Y right, Z up)` to room `(Y, Z, X)` with
  determinant +1, center the footprint, and use one uniform scale to ground it.
  Do not flatten the lowest 2.5 cm of the legs or apply unconditional smoothing.
- Unwrap with xatlas, rasterize canonical surface positions into an atlas, and query
  TripoSR's color field at those positions. Extend colors into empty atlas texels
  for mipmaps. Remove the hard-coded 20-degree front-photo projection onto all sides.
- Preserve smooth normals across UV seams and the exported GLB material in Three.js.
  Roughness is explicitly a category preset, not a measured PBR texture. Neural colors
  can include illumination; they are not guaranteed intrinsic albedo.
- Strictly load all checkpoint weights. Quality mode raises on failure instead of
  quietly replacing neural reconstruction with a procedural extrusion.
- FP16 encoder/transformer weights and activations; FP32 field decoder and density
  exponential. Serialize generation and singleton construction. Bound PyTorch memory,
  account for other allocations again before inference, sample device memory, and
  clear unused CUDA cache on completion/failure.

## Installation on a fresh Windows machine

Requirements: an NVIDIA CUDA-capable GPU, its working driver (`nvidia-smi`), Git,
Node.js, and a Python installation for bootstrapping uv. The installer downloads
several GB. CUDA inference runs in an independent Python 3.11 environment; do not
reuse the CPU-only Python 3.14 environment or --system-site-packages.

From the repository root in PowerShell:

```powershell
python -m pip install uv
.\scripts\setup_furniture_gpu.ps1
```

The script locates uv on PATH or inside .venv/Scripts, installs project-local
Python 3.11 under outputs/runtime/python, and creates .venv-furniture-quality.
It installs PyTorch 2.6.0 / torchvision 0.21.0 from the official CUDA 12.4 wheel
index, then the standalone dependencies in `requirements-furniture-quality.txt`.
It pins TripoSR source revision 107cefdc244c39106fa830359024f6a2f1c78871, applies
the tracked optional-torchmcubes import patch, and uses ArchiGen's scikit-image
surface extractor rather than compiling torchmcubes.

The official stabilityai/TripoSR configuration and checkpoint are downloaded to
TripoSR/. The installer verifies checkpoint SHA256
429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee, caches the
DINO configuration and u2netp mask model, and checks CUDA with a real GPU tensor
operation. The encoder weights are already included in the TripoSR checkpoint.
Caches and models stay in ignored project directories. No training is required.

For a fresh checkout, prepare the ordinary API and frontend once as well:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd frontend
npm ci
cd ..
```

## Start the native GPU studio

```powershell
.\scripts\start_native_gpu.ps1
```

Open http://127.0.0.1:3000/#studio. The default is **high**: 384 cubed density
and a 2048 px neural texture atlas. Use -FurnitureDetail balanced for 256 cubed /
1024 px. The launcher restarts this checkout's worker to apply the requested
preset, waits for CUDA readiness, then restarts this checkout's API with
ARCHIGEN_FURNITURE_SERVICE_URL=http://127.0.0.1:8001. An existing ArchiGen Vite
frontend is reused; otherwise it is started. Unrelated port occupants are rejected.
The API and frontend remain on ports 8000 and 3000; the worker listens only on
127.0.0.1:8001. The startup timeout is 300 seconds and can be overridden with
-StartupTimeoutSeconds.

Logs and process IDs are in outputs/native_gpu/. Processes run in the background;
restart with the same command after Windows restarts. If CUDA, weights, or the
worker are unavailable, startup or generation reports an explicit error; the
configured API never substitutes the local volumetric synthesizer. Neural inference
and color queries use CUDA with FP16 large-model weights and FP32 density/color
decoding. Mask removal, surface extraction, UV packing, and GLB export use CPU.
The existing conservative 3500 MiB device-memory guard remains enabled.

Select the chair reference or upload a complete furniture photo. Download the GLB
or explicitly save it to the room library. Verification scripts do not save models
to the user library. The general API /health describes the floorplan runtime;
check http://127.0.0.1:8001/health for the furniture CUDA worker.

For Docker API/frontend with the same native worker, use the existing
scripts/start_local_stack.ps1 after setup. Its library is stored in the Docker
volume rather than the native frontend directory.

## Evaluation and limitations

The committed comparison uses the same supplied chair image, a height of 0.85 m,
identical orthographic cameras, lighting and Three.js renderer. Inspect front,
three-quarter, side and back views. There is no paired ground-truth scan, so this
does not establish geometric accuracy or generalization across furniture categories.

The old model visibly projects upholstery onto legs and copies the front photograph
across the seat. Neural surface baking fixes that mapping error and gives coherent
wood/upholstery placement around the model. The learned geometry still has irregular
legs, soft details and small texture artifacts. Unseen surfaces remain inferred.
Higher mesh/atlas resolution cannot supply knowledge absent from the checkpoint.

Presets: balanced = 256 cubed density / 1024 px atlas; high = 384 cubed density /
2048 px atlas. The initial balanced chair had 83,238 triangles; the initial high run
had 189,068. High-detail GLBs are dense (about 8 MB for this chair); use balanced for
many-item rooms until decimated LODs are validated.

Initial standalone runs with FP32 weights plus autocast peaked around 2,750 MiB,
but the full studio exposed an allocation failure with resident floorplan models.
The final implementation uses FP16 large-model weights as well. See the accompanying
results JSON for final end-to-end measurements and validation. Device memory is
sampled, not continuously traced; unrelated GPU processes can change availability.

The next quality ceiling is the learned prior and missing views, not output pixel
count. Evaluate a diverse set of real chairs/tables/sofas and add measured or multi-view
references before claiming production fidelity. Larger candidates have published
requirements beyond this laptop: Hunyuan3D 2.1 lists 10 GB for shape generation and
TRELLIS.2 lists at least 24 GB. They were researched, not installed or benchmarked.
Sources: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 and
https://github.com/microsoft/TRELLIS.2.

## Verification

For the native stack, run the following from the repository root:

```powershell
.\.venv-furniture-quality\Scripts\python.exe scripts/verify_furniture_gpu.py
```

This submits two consecutive requests through the public API: the transparent
chair reference, then the same photo composited on an opaque white background
to exercise u2netp. It checks CUDA provenance, strict checkpoint loading, the high
preset, valid textured GLBs, metric height, and the existing device-memory guard.
GLBs, input masks, timings, and memory measurements are saved under
`outputs/furniture_quality/rtx4050/`; the user's catalog is untouched.

The Docker API forwards furniture generation to the native CUDA worker on port
8001. Run `scripts/start_local_stack.ps1` to start the worker and Docker together.
The default is high detail. Use `-FurnitureDetail balanced` when starting a new
worker for lighter meshes. The script reuses an already running quality worker.
Logs are in `outputs/furniture_worker/`. The saved model library stays in the
Docker volume. Restart the worker with the stack after restarting Windows.

`python -m pytest tests/ backend/tests/`, `npm test`, `npm run build`, and the actual
studio/API smoke test (`node scripts/verify_furniture_studio.cjs`). GitHub Actions
checks analytic geometry, coordinate/catalog contracts and the frontend without
requiring model weights or a GPU. It does not evaluate learned reconstruction quality.

## Verified RTX 4050 native result (2026-11-13)

The standalone Python 3.11.18 environment runs PyTorch 2.6.0+cu124 on the RTX 4050
Laptop GPU (6141 MiB). The setup script completed successfully, verified the
checkpoint hash, and performed a CUDA tensor operation. The native launcher
started the strict TripoSR worker and routed the public API to it.

Two consecutive public API requests passed with the high preset (384 cubed,
2048 px atlas), FP16 encoder/transformer and FP32 field decoding:

| Input | Generation | Sampled device peak | Faces | GLB size |
| --- | --- | --- | --- | --- |
| Transparent chair reference | 6.67 s | 1583 MiB | 189092 | 6691948 bytes |
| Same photo on opaque white background, u2netp masking | 5.95 s | 1577 MiB | 181362 | 6361076 bytes |

Both outputs loaded successfully with trimesh, contain a 2048 x 2048 embedded
texture, measure 0.85 m high, and report a watertight, one-component surface.
PyTorch peak allocated/reserved memory was 971.5/1044 MiB in both runs. Evidence
is saved locally in `outputs/furniture_quality/rtx4050/verification.json` and the
adjacent GLBs and masks. These are single-run observations, not a latency benchmark.
The browser automation tool stopped before visual verification, so interactive
preview and the browser download button were not verified in this session.

## Previously verified result on the original environment

The actual studio request returned a watertight, one-component chair with 189,116
triangles and a 2048 px texture. Generation took 282.7 seconds; the sampled total
GPU peak was 1,659 MiB, PyTorch peak allocated memory 1,453.6 MiB, and peak reserved
memory 1,564 MiB with the floorplan models resident. Export size: 8,176,096 bytes.
The height is 0.85 m and inferred footprint approximately 0.452 x 0.473 m.

All 104 Python tests, 13 frontend tests, the production frontend build, actual
studio/API generation and four-view GLB rendering passed. GitHub contracts passed
for commit `8afdae7`. The build retains the existing large-JavaScript-chunk warning.

![Same-camera before and after](assets/furniture-quality/comparison.png)

[Four-view comparison](assets/furniture-quality/multiview.png) and
[machine-readable evidence](furniture-quality-results.json).

The final small follow-up moves FP16 casting onto the CPU before GPU transfer to
avoid a transient FP32 weight allocation. It does not change the selected precision
or checkpoint. GPU inference evidence above is from commit `8afdae7`.

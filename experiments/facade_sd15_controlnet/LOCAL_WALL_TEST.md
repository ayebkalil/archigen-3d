# Local house-wall compatibility test

Completed locally on 15 November 2026. The trained step-750 model generated the
wall in **31.58 seconds**, after **24.25 seconds** of loading. Sampled device peak
was **1,177 MiB**, versus the 3,500 MiB target; PyTorch allocation/reservation peaks
were 906.15 / 1,000 MiB. This is one inference with the other project GPU services
temporarily paused, not a concurrent furniture/floorplan/facade benchmark.

The raw image adds a roof and lower story, but both lie outside the wall's active
UV rectangle for this case. The mapped wall looks cleaner than the current
Pix2Pix result. Painted shutters/shadows still extend around some real window
frames. This demonstrates useful progress without establishing reliable alignment
on arbitrary walls or styles.

Automated inspection passed for all three views: seven openings retained, maximum
UV error 4.60e-8, frame-center projection error below 1.81e-7 m, unchanged positions
when switching textures, and no JavaScript/WebGL errors. These measure geometric
placement, not semantic accuracy of the synthesized window pixels. Both normal
services were restored and their health endpoints checked successfully. On Windows
the furniture worker must be relaunched through its isolated venv, even if a
process inspection exposes the base Python executable; the session helper handles
that launcher behavior explicitly.

Machine-readable results and source component hashes are in
`results/local_house_wall.json`. Screenshots and generated files remain under the
application checkout's ignored `outputs/facade_house_test/` directory. Production
weights and defaults were not changed.

This test applies the trained Kaggle checkpoint to the existing sample house's
real exterior wall. It uses the current project's `Viewer3D.jsx` in an isolated
page on port 3001. It does not switch the production facade generator.

The selected wall is `ground_east`: approximately 7.92 m × 2.50 m, six windows and
one door. Geometry, opening dimensions and frame positions remain identical when
switching between production Pix2Pix, the trained candidate and the opening guide.
The sample geometry is a trace; this test does not evaluate floorplan detection.

## Preparation

Use the experiment checkout as the working directory. `PROJECT` below is the
absolute path to the current application checkout; set it appropriately.

1. Obtain `facade-controlnet/export` from Kaggle's saved output. Only FP16
   `controlnet/diffusion_pytorch_model.safetensors`, `controlnet/config.json` and
   `model.json` are needed for inference. Verify the checksum recorded in the
   exported metadata. Keep these separate from production models.
2. In the prepared diffusion environment, download the frozen, pinned SD1.5 base:
   `python -m experiments.facade_sd15_controlnet.prepare_local_base`.
   This caches only required FP16 components (no safety-checker weights), verifies
   weight SHA-256, and uses bounded resumable downloads outside OneDrive.
3. While the normal API is running, prepare inputs and a real production baseline:
   `python -m experiments.facade_sd15_controlnet.prepare_house_wall --project PROJECT`.

## Generate and restore services

```powershell
python -m experiments.facade_sd15_controlnet.local_wall_session --project PROJECT --model PATH_TO_EXPORT
```

The parent uses the project's `.venv-facade-diffusion/Scripts/python.exe` for
generation. It verifies that listeners on ports 8000/8001 are project Uvicorn
processes, captures launch settings in memory, pauses them to release GPU/RAM and
restores them in `finally`. Other apps are not stopped. First download all weights
so generation runs offline. A failed RAM/VRAM preflight does not lower the guard.

The sampler polls device memory at roughly 0.5 s intervals, including model loading.
PyTorch allocator peaks, sampled device peak and load/generation times are saved
separately. Sampling cannot certify the instantaneous device-wide peak. Results
over the 3,500 MiB sampled target are rejected. No laptop numbers are inferred from
Kaggle's T4 timings.

## View and inspect

```powershell
python -m experiments.facade_sd15_controlnet.wall_test_viewer --project PROJECT
```

Start Vite in the application checkout's `frontend` folder on port 3001, then open
`http://127.0.0.1:3001/tmp/facade-wall-test/`. All staged files are under the ignored
`frontend/tmp` folder. The existing application on port 3000 is unchanged.

Automated inspection and screenshots:

```powershell
node experiments/facade_sd15_controlnet/render_wall_test.cjs PROJECT
```

Files are saved in `PROJECT/outputs/facade_house_test/`. Numeric checks validate
normalized UVs, physical frame positions and the seven retained 3D openings.
They **do not** establish that the generated pixels depict correctly aligned
windows. Inspect the raw candidate and its front/oblique viewer screenshots to
assess extra painted openings, frame seams, baked shadows and perspective.

The candidate uses the exported step-750 weights with unchanged default prompt,
seed 42, guidance 6.0, ControlNet scale 1.1 and 25 inference steps. It receives the
canonical semantic sketch converted to the same boundary conditioning as training.
The baseline is the running API's actual output, including its production finishing;
this viewer comparison differs from the raw-Pix2Pix Kaggle benchmark protocol.

# V2 on the actual two-storey house — 15 November 2026

Preview: <http://127.0.0.1:3001/tmp/facade-multiface-v2/>

All **15 exterior panels** are applied together. Use **Inspect side** and
**Face this wall**, the two corner overview buttons, and the viewer's floor
selector. Compare **V2 diffusion**, **Current facade** (the actual production
Pix2Pix 256 response), **Plain finish** and **Opening guide**. Every mode holds
the same 30 canonical physical openings fixed. Raw generated images can be
opened separately. The current studio at port 3000 is unchanged.

## Execution and provenance

- User-downloaded ZIP SHA256 verified:
  `e148df15a758a703e7a5fa8d28e4809371b96b8c5b18e1e043398efcfb095afb`.
- Selected V2 checkpoint: **2000**. Both ControlNet and LoRA hashes verified.
- Full training and paired validation histories/artifacts imported into the
  existing local facade MLflow database. Local training run
  `c9141e312f6f4a3cbb3cd81d5aac754b`; evaluation
  `e7e6f18a9eaa4090a8de8f7f1721bb30`.
- Fifteen textures generated **locally on the RTX 3050**, at 512×512 with
  25 diffusion steps, FP16 and sequential CPU offload. Identical neutral prompt
  and seed 42 used for every panel; no material tinting or photo retouching.
- Mean per panel: **34.3 seconds**. Whole batch including model load:
  **584.9 seconds**. Maximum torch allocation: **910.40 MiB**. Device-wide
  sampled peak: **1081 MiB**, with 0.5-second samples; this does not guarantee
  the instantaneous peak or concurrent-workload behavior.
- Only recognized project GPU services on 8000/8001 were paused for generation.
  Both restored and verified healthy afterward, including the isolated
  furniture runtime. Frontend services remained running.
- Local house-test MLflow run: `fdef24bd43de4333a2d5239a0f3d5413`.

This uses the bundled reference house and the current production opening fitter,
including registered upper-floor placement. It is not evidence of floorplan
segmentation performance on unseen uploaded drawings.

## Renderer verification

Headless browser used the **actual production Viewer3D** component, without
editing its geometry implementation. Checks across all four texture modes:

- 15 exterior cladding meshes and 30 physical openings retained.
- Opening positions and sizes identical across modes; no unresolved openings.
- Maximum planar UV error: **9.48e-8**.
- No JavaScript/WebGL errors.
- Individual screenshots captured for all 15 panels and both overview corners.

These checks establish renderer consistency. They do **not** establish pixel
alignment between neural painted openings and the physical window/door frames.

## Visual outcome: not accepted for automatic whole-house facades

The model generates more detailed individual facade photographs, but the whole
house exposes material changes across adjacent panels, vegetation/contextual
details, and baked lighting. The blank `ground_south_extension` panel receives
a texture depicting glass openings absent from its semantic guide. Its mesh
correctly remains solid, so the failure comes from generated imagery.

All raw results remain visible in the preview. No rejected panel was secretly
replaced or recolored. Improved validation LPIPS therefore does not establish
that V2 is ready to replace the production facade on an entire house. Shared
material conditioning and stronger preservation of the opening mask are the
next constraints to address.

## Artifacts and reproduction

Summary: [results/local-house-multiface.json](results/local-house-multiface.json).
Generated images, probes, screenshots and browser checks are under the ignored
`outputs/house-multiface/` folder in this experiment worktree. The staged preview
is under the primary project's ignored `frontend/tmp/facade-multiface-v2/`.

```powershell
python -m experiments.facade_semantic_v2.house_multiface prepare --project PROJECT --output outputs/house-multiface
python -m experiments.facade_semantic_v2.local_house_session --project PROJECT --model EXPORT_FOLDER --output outputs/house-multiface
python -m experiments.facade_semantic_v2.house_viewer --project PROJECT --output outputs/house-multiface
# Serve PROJECT/frontend with Vite on port 3001, then:
node experiments/facade_semantic_v2/render_house.cjs PROJECT OUTPUT_FOLDER
```

An unused private Kaggle fallback notebook was prepared while the large model
download failed. The reported images and performance above come from the local
run after the user supplied the completed ZIP.

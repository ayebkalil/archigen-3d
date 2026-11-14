# Kaggle ControlNet experiment: completed validation review

Reviewed 14 November 2026. **Decision: keep experimental; do not replace production.**

The notebook completed training, evaluation and packaging successfully. The
candidate produces visibly sharper, more photographic facades in the reviewed
comparisons. It also changes requested architectural details and introduces
unwanted scene content. This is progress in appearance, not an overall win for
projecting textures onto fixed 3D walls and openings.

## Provenance

- [Training notebook, saved version 3](https://www.kaggle.com/code/khalilayeb/archigen-facade-sd15-controlnet-mlflow).
- Source commit: `f6a9bd4e7ab35cdf05e77bc7f0f3b47389822ff0`.
- Training run: `68de00bb44c242a4b3ffa9cf1d7c9cf7`, status `FINISHED`.
- Training: 1,500 effective optimizer steps; batch 1, accumulation 4, FP16,
  checkpointing, AdamW8bit, one Tesla T4.
- Training metric elapsed at step 1,500: 4,010.24 seconds. This excludes setup
  and the final validation/export following that step.
- Peak PyTorch allocated training memory: 6,373.15 MiB. This is not laptop
  inference memory or the true peak across all device allocations.
- Dataset: locked CMP split, 400 training / 100 validation / 106 test.
- Manifest SHA-256: `ed32d4cd052e2ac2b3314d962faacbb4dfe1adbfd7470796dc9346f305ae42f8`.

## Paired evaluation on 100 validation images

Both models used the same original targets at 512 pixels. The baseline is the
production EMA Pix2Pix ONNX model's raw 256-pixel output, resized with bicubic.
Production enhancement/material finishing was not included. Diffusion used
25 steps, guidance 6.0, ControlNet scale 1.1, fixed seed per sorted image ID and
the generic neutral architectural prompt.

| Metric | Pix2Pix baseline | ControlNet candidate | Interpretation |
| --- | ---: | ---: | --- |
| LPIPS AlexNet, lower is better | 0.59899 | 0.56074 | Candidate lower by 6.38% |
| SSIM, higher is better | 0.27431 | 0.19812 | Candidate lower |
| PSNR, higher is better | 13.00 dB | 10.40 dB | Candidate lower by 2.60 dB |
| Opening-region edge MAE, lower is better | 0.27563 | 0.31932 | Candidate error higher by 15.85% |
| Diffusion generation time | Not benchmarked here | 27.09 s/image | T4 with sequential CPU offload; not laptop latency |

The candidate had lower LPIPS on 72 of 100 images. The paired bootstrap 95%
interval for candidate minus baseline LPIPS was [-0.05333, -0.02451]. This describes
the observed validation comparison; it is not an independent held-out test result.

Opening-edge MAE is a Canny edge proxy, not window/door IoU or a measured geometric
displacement. Richer texture can also change this metric. Visible layout deviations
in the comparison images support the concern independently. Lower SSIM/PSNR mean
less resemblance to the paired target, not necessarily lower photographic realism.

These LPIPS values must not be compared directly to older registry numbers without
matching image resolution, preprocessing, metric backbone and split.

## Checkpoint selection

The exported candidate is **step 750**, selected by lowest LPIPS on the same four
fixed validation images. All 1,500 training steps completed; export deliberately
uses the selected checkpoint rather than the last checkpoint.

| Step | Four-image validation LPIPS |
| ---: | ---: |
| 250 | 0.58472 |
| 500 | 0.54872 |
| 750 | 0.54417 |
| 1,000 | 0.54801 |
| 1,250 | 0.55008 |
| 1,500 | 0.54788 |

Four examples provide a noisy model-selection signal. Later steps did not improve
that signal; this alone does not establish overfitting. The 100-image evaluation
includes those four selection images. The 106-image test split remains untouched.
No unfine-tuned ControlNet baseline was evaluated, so the fine-tuning contribution
cannot be separated from the pretrained diffusion prior with these results.

## Visual findings

Downloaded comparison panels are under
`outputs/kaggle-results-v3/facade-controlnet/evaluation/comparison_000.png` through
`comparison_007.png`. Panels are semantic edges, Pix2Pix, candidate, original target.

- Samples 000/002: crisp masonry and window frames, but rectangular openings can
  become arched and additional architectural ornament/balconies appear.
- Sample 001: sharper high-rise detail, with conspicuous distorted repeated bands.
- Samples 003/004/006/007: changed roofs, balconies, facade colors or opening
  geometry, and sometimes street/vehicle/vegetation content.
- Black triangular border artifacts appear in multiple outputs, unsuitable for
  direct application as a complete wall texture.

## Artifacts and local MLflow

Kaggle produced `archigen-facade-model-and-mlflow.zip` and the FP16 ControlNet export.
The archive and weight files themselves were not downloaded or profiled locally
for this review. The following hashes were read from the completed run's log and
export metadata; they are not local verification of downloaded weight bytes:

- Archive SHA-256: `3ca2eb3fbe1d9d231bdadc008fbe897ac1d7721f92f07e540655f4da41964360`.
- Weights SHA-256: `d55c08f221c658e11743a248ebc4154ec16ea715ef84cf8129e10ae69fa782f2`.

All metric histories and the downloaded comparison/report artifacts were imported
into the separate local SQLite store `outputs/facade-experiment-mlflow.db`:

- Local training run: `12b388c3036544e6ad7e83c430225486`.
- Local evaluation run: `ddbb8fdf2b1649918d90c14db499d991`.

Training validation images are not all downloaded, so that local run is marked as
a partial artifact import; importing the full bundle later can add the remainder.
Production models, the running backend and the original project's MLflow database
were not changed by this review.

Aggregate metrics, checkpoint history and per-image values are committed under
`results/`. Photographs, generated comparison PNGs and weights stay outside Git.

## Next experiment

First reuse these weights in a small validation-only inference sweep: stronger
geometric conditioning, lower text guidance and a flat wall/elevation prompt. Do
not spend another full training run merely increasing epochs.

If layout deviations remain, test class-aware semantic conditioning and explicit
wall/opening masks. The present classless boundary map does not distinguish a
window outline from decoration. Cropping/masking training targets to the intended
facade surface should be tested against unwanted sky, streets and black borders.
These are hypotheses to evaluate, not demonstrated fixes.

Compare that controlled experiment with the unfine-tuned pretrained ControlNet
and the production baseline. Use a larger validation selection set and preserve
the held-out test. Dataset expansion is a separate experimental variable. Promote
only after geometry review and local RTX 3050 memory/latency profiling pass.

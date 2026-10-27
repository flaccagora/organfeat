# S3: training ablations (arm A multimodal recipe: λ 0, μ 0.5, encoder lr 1e-4, 10 k steps)

Each ablation is selected on val in the primary setting (recall, then inlier ratio) among steps 8 k /
10 k. The main runs chose among 6 / 8 / 10 k; the proxy curves had flattened by 6 k. Each is scored
once on test.

## Val (primary setting): "img" with images / ":geo" geometry-only inference
| run | step | success img / :geo | inlier ratio@10 img / :geo | point recall@10 img / :geo |
|---|---|---|---|---|
| main (s2_mm_lam0_lr1e-4) | 10 k | 1.000 / 0.997 | 0.349 / 0.345 | 0.390 / 0.381 |
| no modality dropout | 8 k | 1.000 / 0.997 | 0.353 / 0.339 | 0.395 / 0.373 |
| 50 % of train livers (204) | 10 k | 1.000 / 1.000 | 0.357 / 0.349 | 0.392 / 0.382 |
| 25 % of train livers (102) | 10 k | 0.997 / 0.994 | 0.348 / 0.343 | 0.387 / 0.376 |
| scratch init (no Sonata pretraining) | 10 k | 0.997 / 0.991 | 0.268 / 0.264 | 0.294 / 0.285 |

## Reading (val)
- **Sonata pretraining matters for the descriptor task** (+0.08 inlier ratio over scratch). This
  differs from defreg's non-rigid registration task (defreg results 014: worth ~0.1 mm).
- **Training-data size saturates by ~100 livers.** 25 / 50 / 100 % are within ±0.01.
- **Modality dropout makes no measurable difference on synthetic data.** Without dropout,
  geometry-only inference is ~0.005 lower, which is within noise. The CT side always trains the
  missing-image path, so the model learns it either way.
- **The cross-modal consistency term:** see docs/s2_findings.md (λ = 1 is worse).

## Test (primary setting, 46 livers; scored once)
| run | success img / :geo | inlier ratio@10 img / :geo | point recall@10 img / :geo | Δ inlier ratio vs main (Wilcoxon) |
|---|---|---|---|---|
| main (A_mm) | 0.996 / 0.995 | 0.359 / 0.355 | 0.396 / 0.388 | – |
| no modality dropout | 0.997 / 0.995 | 0.361 / 0.348 | 0.399 / 0.380 | +0.002 (p 0.3) |
| 50 % train livers | 0.999 / 0.996 | 0.363 / 0.359 | 0.397 / 0.390 | +0.004 (p 0.06) |
| 25 % train livers | 0.996 / 0.991 | 0.356 / 0.351 | 0.393 / 0.385 | −0.003 (p 0.39) |
| scratch init | 0.982 / 0.984 | 0.273 / 0.268 | 0.293 / 0.287 | **−0.086 (p 3e-14)** |

The test confirms val:
- Only the pretrained Sonata initialisation matters.
- Data size (beyond ~100 livers) and modality dropout make no detectable difference on synthetic
  data.
- Without dropout, geometry-only inference is ~0.01 lower (0.348 vs 0.355). This is the only hint
  that dropout matters for the missing-modality path.

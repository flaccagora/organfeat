# S2: multimodal descriptor (interim, 2026-10-26)

All numbers are val (20 livers × 16 views) in the primary setting (σ = 1 mm, 10 % holes), averaged per
liver. "img" is inference with the partial view's lifted DINOv2 channel; ":geo" is the same checkpoint
with the visual channel replaced by the missing vector, i.e. the CT path.

## Arm A (Sonata), encoder lr 2e-5
| run / checkpoint | success | inlier ratio@10 | point recall@10 | coverage AUC |
|---|---|---|---|---|
| s1_geo_lr2e-5 8 k / 10 k (geometry-only training) | 0.991 / 0.991 | 0.277 / 0.279 | 0.295 / 0.298 | 0.270 / 0.271 |
| s2_mm_lam0 8 k, img | 1.000 | **0.289** | **0.309** | **0.286** |
| s2_mm_lam0 8 k, :geo | 0.994 | 0.275 | 0.293 | 0.267 |
| s2_mm_lam0 10 k, img / :geo | 0.997 / 0.988 | 0.289 / 0.279 | 0.311 / 0.296 | 0.284 / 0.271 |
| s2_mm_lam1 8 k, img / :geo | 0.997 / 0.994 | 0.276 / 0.267 | 0.304 / 0.293 | 0.276 / 0.262 |
| s2_mm_lam1 10 k, img / :geo | 0.997 / 0.994 | 0.276 / 0.270 | 0.307 / 0.296 | 0.277 / 0.263 |

Images at inference vs geometry-only inference of the same checkpoint (inlier ratio@10; paired
Wilcoxon over livers):

| run | step | Δ (img − :geo) | livers with img > :geo | p |
|---|---|---|---|---|
| λ = 0 | 6 k | +0.009 | 15 / 20 | 0.001 |
| λ = 0 | 8 k | +0.015 | 17 / 20 | 0.0002 |
| λ = 0 | 10 k | +0.011 | 17 / 20 | 0.0001 |
| λ = 1 | 6 k | +0.006 | 14 / 20 | 0.04 |
| λ = 1 | 8 k | +0.009 | 19 / 20 | < 0.0001 |
| λ = 1 | 10 k | +0.006 | 15 / 20 | 0.005 |

## Reading (interim)
- **Images help, by a small amount.** The lifted image channel raises the inlier ratio by
  +0.006–0.015 (about +3–5 % relative). The effect is consistent across livers.
- **Registration recall is at ceiling** for every variant (0.99–1.00), so the effect only shows in
  the correspondence metrics.
- **Multimodal training alone does not help the geometry descriptor.** Geometry-only inference after
  multimodal training (0.275 at 8 k) equals geometry-only training (0.277).
- **The cross-modal consistency term (λ = 1) does not help.** λ = 0 is better.
- **Pending:**
  - arm A multimodal runs at encoder lr 1e-4. Geometry-only at lr 1e-4 is much stronger (0.341),
    so this is the decisive comparison.
  - arm B (PPF) multimodal runs. On the training proxy, the PPF image gain is relatively larger.

## Arm B (PPF), lr 1e-3
| run / checkpoint | success img / :geo | inlier ratio img / :geo |
|---|---|---|
| s1_ppf 8 k (geometry-only training) | 0.994 | 0.128 |
| s2_ppf_lam0 6 k | 0.994 / 0.991 | 0.152 / 0.129 |
| **s2_ppf_lam0 8 k (selected)** | **1.000** / 0.984 | **0.157** / 0.132 |
| s2_ppf_lam0 10 k | 0.997 / 0.994 | 0.161 / 0.135 |
| s2_ppf_lam1 6 / 8 / 10 k | 0.991 / 0.991 / 0.997 (img) | 0.119 / 0.124 / 0.127 (img); 0.102 / 0.106 / 0.108 (:geo) |

- **Images help the local descriptor much more.** Images add +0.023–0.026 inlier ratio to PPF
  (about +19 % relative), against +3–5 % for Sonata. A strictly local descriptor cannot tell smooth
  patches apart and gains the most from appearance.
- **λ = 0 is again better than λ = 1.**
- **Selected for arm B** (recall, then inlier ratio): `outputs/train/s2_ppf_lam0/step_008000.pt`.

## Arm A (Sonata), encoder lr 1e-4: the decisive comparison
| run / checkpoint | success | inlier ratio@10 | point recall@10 | coverage AUC |
|---|---|---|---|---|
| s1_geo_lr1e-4 8 k (geometry-only, selected in S1) | 1.000 | 0.341 | 0.377 | 0.341 |
| s1_geo_lr1e-4 10 k | 0.997 | 0.345 | 0.381 | 0.347 |
| s2_mm_lam0_lr1e-4 8 k img / :geo | 1.000 / 1.000 | 0.347 / 0.341 | 0.386 / 0.376 | 0.351 / 0.343 |
| **s2_mm_lam0_lr1e-4 10 k img (selected)** / :geo | **1.000** / 0.997 | **0.349** / 0.345 | **0.390** / 0.381 | **0.353** / 0.348 |
| s2_mm_lam1_lr1e-4 8 k img / :geo | 1.000 / 1.000 | 0.335 / 0.331 | 0.388 / 0.381 | 0.340 / 0.334 |
| s2_mm_lam1_lr1e-4 10 k img / :geo | 1.000 / 0.997 | 0.339 / 0.334 | 0.393 / 0.385 | 0.344 / 0.337 |

Paired inlier-ratio differences over livers (Wilcoxon):

| run | step | img − :geo | img − geometry-only training at the same step |
|---|---|---|---|
| λ = 0 | 6 k | +0.005 (13 / 20, p 0.02) | +0.006 (14 / 20, p 0.06) |
| λ = 0 | 8 k | +0.007 (18 / 20, p 0.001) | +0.006 (14 / 20, p 0.008) |
| λ = 0 | 10 k | +0.004 (14 / 20, p 0.06) | +0.004 (12 / 20, p 0.17) |
| λ = 1 | 6 / 8 / 10 k | +0.004–0.006 (p ≤ 0.01) | +0.001 / −0.007 / −0.007 (worse) |

## S2 selection and reading
- **Arm A multimodal:** `outputs/train/s2_mm_lam0_lr1e-4/step_010000.pt`. Recall is at ceiling, so
  it is chosen on the best inlier ratio (0.349).
- **Arm B multimodal:** `outputs/train/s2_ppf_lam0/step_008000.pt`.
- **Images help the strong full-context encoder only marginally.** For arm A the gain over the
  geometry-only model is +0.004–0.007 inlier ratio (about +1–2 %), consistent but not always
  significant on 20 val livers.
- **Images help the strictly local descriptor clearly** (arm B: about +19 %).
- **The cross-modal consistency term hurts at lr 1e-4** (λ = 1 is below geometry-only).
- **The noisy primary setting saturates registration recall for every learned variant**, so all
  differences show in the secondary correspondence metrics. Test and stress tests (σ 2 mm, held-out
  textures, flat albedo, tools, deformation, low overlap) decide whether the effect generalises and
  where it matters.

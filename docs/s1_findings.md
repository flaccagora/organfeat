# S1: geometry-only learned descriptor, diagnosis and gate (2026-10-26)

## First attempt: collapse, like defreg P3
The first config used head lr 1e-3, bf16 autocast, encoder lr 2e-5 or 1e-4, and 10 k steps. Both runs
learned for a few hundred steps (train top-1 within 10 mm rose to ~3 %), then collapsed after warmup:
the loss rose and accuracy fell to chance. The 1e-4 run collapsed at step ~300, the 2e-5 run started
degrading at step ~600–900. This is the pattern of defreg results 014 (P3: "collapse after warmup").
Both runs were stopped.

## Diagnosis (2 k steps each, fp32, head lr 1e-4, encoder lr 2e-5)
Following the protocol's S1 gate, the receptive field was diagnosed first.

| run | change | val σ 1 mm (primary): success [95 % CI] | inlier ratio@10 | val noise-free: success | inlier ratio@10 |
|---|---|---|---|---|---|
| **diag_fp32** | none (full Sonata context) | **0.953 [0.922, 0.981]** | **0.156** | 0.975 | 0.230 |
| diag_crop | 50 % random CT crops (r 60–150 mm) | 0.941 [0.922, 0.959] | 0.148 | 0.984 | 0.221 |
| diag_local | 3 encoder stages, attention patch 64 | 0.925 [0.897, 0.950] | 0.114 | 0.994 | 0.250 |
| diag_local_crop | local + crops | 0.953 [0.931, 0.972] | 0.104 | 0.988 | 0.226 |
| diag_ppf | strictly local PPF-PointNet (10 / 25 mm), from scratch, lr 1e-3 | **0.959** | 0.082 | **1.000** | **0.579** |
| FPFH r20 | handcrafted | 0.631 [0.566, 0.694] | 0.034 | 1.000 | 0.267 |
| frozen Sonata | – | 0.172 | 0.012 | 0.169 | 0.020 |

Val has 20 livers × 16 views. Every learned run beats FPFH r20 in the primary setting by 0.29–0.32
(paired Wilcoxon over livers, p < 0.001). Noise-free, the Sonata runs are 0.6–2.5 points below
FPFH's ceiling, and PPF reaches it (1.000).

## Reading
- **The cause was optimisation, not the receptive field.** With a 10× smaller head lr and fp32,
  the same full-context encoder learns steadily, with no collapse in 2 k steps. Under bf16, spconv
  runs fp16 kernels. This run did not separate the two factors (head lr vs precision); both are
  fixed in the recipe.
- **Context buys robustness to depth noise, but locality buys precision.**
  - *Under depth noise:* the full-context encoder has the best inlier ratio (0.156); local attention
    gives 0.114 and the strictly local PPF network 0.082.
  - *Noise-free:* PPF is by far the most precise (inlier ratio 0.58 vs 0.23).
  - *Registration recall:* the RANSAC bound saturates at ~3–5 % inliers, so in the primary setting
    recall is the same (PPF 0.959, Sonata 0.953).
  - *Why:* local descriptors depend on per-point normals, which depth noise corrupts. FPFH shows
    the same weakness: 1.00 → 0.63 at σ = 1 mm.
  - *Consequence:* the learned PPF network keeps 0.96 recall where FPFH falls to 0.63, so learning
    on invariant local features recovers most of the noise robustness.
- **CT crops** (the proposal's "overlapping crops") neither help nor hurt measurably at 2 k steps.
- **The S1 gate is passed** (on val, 2 k-step diagnostics).

## S1 recipe (selection on val, primary setting: recall, then inlier ratio)
- The diag_fp32 configuration: full-context Sonata, fp32, head lr 1e-4, encoder lr 2e-5, no crops.
- It is now the default in `configs/train/base.yaml`.
- **Full runs:** `scripts/train/s1_geo.sh` (10 k steps, encoder lr {2e-5, 1e-4}) and the S2
  multimodal runs `scripts/train/s2_mm.sh` (λ ∈ {0, 1}, μ = 0.5).
- **Arm B (PPF), added with the user:** `s1_ppf` (geometry-only) and `s2_ppf_lam{0,1}` (multimodal),
  10 k steps, lr 1e-3.

## S1 full runs and selection (val, primary setting; 2026-10-26)
| run, checkpoint | success | inlier ratio@10 | point recall@10 | coverage AUC |
|---|---|---|---|---|
| s1_geo_lr1e-4, step 6 k | 0.997 | 0.324 | 0.356 | 0.321 |
| **s1_geo_lr1e-4, step 8 k (selected)** | **1.000** | 0.341 | 0.377 | 0.341 |
| s1_geo_lr1e-4, step 10 k | 0.997 | 0.345 | 0.381 | 0.347 |
| s1_geo_lr2e-5, step 6 / 8 / 10 k | 0.984 / 0.991 / 0.991 | 0.266 / 0.277 / 0.279 | 0.278 / 0.295 / 0.298 | 0.258 / 0.270 / 0.271 |
| FPFH r20 | 0.631 | 0.034 | 0.013 | 0.035 |

- **Selected** (recall, then inlier ratio): `outputs/train/s1_geo_lr1e-4/step_008000.pt`.
- **Ceiling.** The learned geometry-only descriptor reaches the registration-recall ceiling in the
  primary setting too. Whether images add anything is therefore judged on the pre-registered
  secondary metrics (inlier ratio, point recall, coverage) and on the stress tests. The primary
  metric is not amended again after seeing learned results.

### Arm B (PPF) geometry-only, val primary setting
| checkpoint | success | inlier ratio@10 |
|---|---|---|
| s1_ppf 6 k | 0.991 | 0.128 |
| **s1_ppf 8 k (selected)** | **0.994** | 0.128 |
| s1_ppf 10 k | 0.988 | 0.133 |

Selected for arm B: `outputs/train/s1_ppf/step_008000.pt`. It is at the recall ceiling too, but its
inlier ratio under noise is lower than arm A's (0.128 vs 0.341).

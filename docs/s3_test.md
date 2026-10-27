# S3: main test (scored once, 2026-10-26)

46 test livers × 16 views = 736 pairs per descriptor. Statistics are per liver, with 95 % bootstrap
CIs over livers and paired Wilcoxon tests. `scripts/bench/test_report.py --set test_noise1_holes0.1` /
`--set test`. Descriptors (`scripts/bench/final_descs.sh`):
- **A_geo / A_mm:** arm A (Sonata, fine-tuned), geometry-only / multimodal training (λ = 0).
- **A_mm:geo:** the multimodal checkpoint with geometry-only inference.
- **B_\*:** the same three variants for arm B (local PPF-PointNet).

## Primary setting (σ = 1 mm at 100 mm, 10 % holes)
| descriptor | registration recall [95 % CI] | inlier ratio@10 [95 % CI] | point recall@10 | coverage AUC | descriptor ms (both clouds) | RANSAC ms | peak GPU MB |
|---|---|---|---|---|---|---|---|
| A_geo | 0.996 [0.990, 1.000] | 0.349 [0.340, 0.359] | 0.383 | 0.349 | 214 | 75 | 5788 |
| **A_mm** | 0.996 [0.990, 1.000] | **0.359** [0.350, 0.368] | **0.396** | 0.361 | 297 | 75 | 6950 |
| A_mm:geo | 0.995 [0.989, 0.999] | 0.355 [0.346, 0.365] | 0.388 | 0.356 | 286 | 76 | 6950 |
| B_geo | 0.991 [0.984, 0.996] | 0.147 [0.137, 0.156] | 0.109 | 0.185 | 163 | 118 | 5375 |
| **B_mm** | **0.999** [0.996, 1.000] | 0.176 [0.166, 0.186] | 0.134 | 0.218 | 249 | 94 | 6536 |
| B_mm:geo | 0.988 [0.980, 0.995] | 0.150 [0.140, 0.159] | 0.108 | 0.188 | 223 | 107 | 6536 |
| FPFH r20 | 0.670 [0.615, 0.723] | 0.038 [0.035, 0.042] | 0.016 | 0.039 | 115 | 218 | 530 |
| FPFH r20 ‖ DINO (CT: zeros) | 0.666 [0.617, 0.715] | 0.038 | 0.016 | 0.044 | 313 | 215 | 1700 |
| frozen Sonata | 0.139 [0.115, 0.162] | 0.013 | 0.010 | 0.013 | 221 | 87 | 1052 |
| frozen Sonata PCA-32 | 0.097 [0.077, 0.115] | 0.012 | 0.009 | 0.011 | 189 | 84 | 946 |

Paired differences over livers (count of livers better / worse, Wilcoxon p):

| comparison | registration recall | inlier ratio@10 | point recall@10 |
|---|---|---|---|
| **A_mm − A_geo** (primary comparison, arm A) | +0.000 (1 / 1, p 1) | **+0.010** (36 / 10, p 2e-5) | **+0.013** (45 / 1, p 6e-14) |
| A_mm − A_mm:geo (images at inference) | +0.001 (p 0.56) | +0.004 (30 / 16, p 0.02) | +0.008 (45 / 1, p 2e-13) |
| **B_mm − B_geo** (primary comparison, arm B) | **+0.008** (7 / 1, p 0.03) | **+0.029** (46 / 0, p 3e-14) | **+0.025** (46 / 0, p 3e-14) |
| B_mm − B_mm:geo | +0.011 (7 / 0, p 0.01) | +0.026 (46 / 0) | +0.026 (46 / 0) |
| A_geo − FPFH r20 | **+0.326** (44 / 0, p 7e-9) | +0.311 (46 / 0) | +0.368 (46 / 0) |
| A_mm − FPFH r20 | +0.326 (44 / 0, p 7e-9) | +0.321 (46 / 0) | +0.380 (46 / 0) |
| B_geo − FPFH r20 | +0.321 (44 / 0, p 7e-9) | +0.108 (46 / 0) | +0.094 (46 / 0) |
| B_mm − FPFH r20 | +0.329 (44 / 0, p 7e-9) | +0.138 (46 / 0) | +0.118 (46 / 0) |
| A_mm − B_mm | −0.003 (0 / 2, p 0.16) | +0.183 (46 / 0) | +0.262 (46 / 0) |

## Noise-free (secondary)
| descriptor | registration recall | inlier ratio@10 | point recall@10 | coverage AUC |
|---|---|---|---|---|
| A_geo / A_mm / A_mm:geo | 0.997 / 0.996 / 0.997 | 0.438 / 0.440 / 0.440 | 0.453 / 0.458 / 0.457 | 0.437 / 0.441 / 0.440 |
| B_geo / B_mm / B_mm:geo | 1.000 / 1.000 / 1.000 | 0.784 / **0.800** / 0.785 | 0.724 / **0.745** / 0.729 | 0.900 / 0.903 / 0.901 |
| FPFH r20 | 1.000 | 0.292 | 0.256 | 0.348 |
| frozen Sonata / PCA-32 | 0.125 / 0.140 | 0.016 / 0.017 | 0.012 / 0.010 | 0.015 / 0.014 |

- **Images:** A_mm − A_geo gives inlier ratio +0.003 (p 0.12) and point recall +0.005 (p 4e-6).
  B_mm − B_geo gives inlier ratio +0.016 and point recall +0.021 (46 / 0).
- **Learned vs FPFH:** every learned descriptor beats FPFH on inlier ratio (46 / 0).

## Reading
1. **Learned descriptors are far better than FPFH and frozen Sonata.** The primary endpoint is
   registration recall at σ = 1 mm: learned 0.99–1.00, FPFH 0.67, frozen Sonata 0.14. Frozen
   scene-pretrained features (Sonata) carry almost no liver correspondence information.
2. **Primary comparison (images vs no images): a tie on the primary endpoint for arm A.** Both arms
   are at the recall ceiling. Arm B shows a small significant recall gain (+0.008).
3. **On the pre-registered secondary correspondence metrics, images help consistently:**
   - **arm A**, full-context Sonata: +3 % relative inlier ratio and point recall under noise; about
     nothing noise-free.
   - **arm B**, strictly local PPF: +20 % relative under noise, +2 % noise-free, on all 46 test livers.

   The visual channel mostly does what local geometry cannot: it disambiguates smooth patches. A
   global-context encoder already resolves much of that ambiguity from shape.
4. **Most of the image gain is inference-time evidence.** Multimodal training with geometry-only
   inference (`:geo`) is between geometry-only training and multimodal inference; arm B gains
   nothing from training alone.
5. **The architecture trade-off persists on test.** Under depth noise Sonata gives more correct
   matches; noise-free PPF is about twice as precise (0.80 vs 0.44 inlier ratio). Registration
   recall is the same.
6. **Cost.** Learned descriptors cost about 160–300 ms per pair on an A100 (including DINOv2 ViT-L
   for the multimodal ones) and 5–7 GB of GPU memory. FPFH costs 115 ms. Better matches make RANSAC
   2–3× faster (75–118 ms vs 218 ms).

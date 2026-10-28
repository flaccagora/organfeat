# R4: lifting baselines (post hoc, 2026-10-28)

Pre-registered in docs/protocol.md (2026-10-28). `scripts/bench/run_bench.py --set {test,test_tex_flat}
--sigma100 1 --holes 0.1`. The descriptors are in `ofeat/eval/descriptors.py` (`dinolift`,
`<geom>+dinolift`). CT features are cached in `outputs/cache/dinolift/`.

**`dinolift`.**
- *Partial:* DINOv2 PCA-64 lifted from its frame (as in training).
- *CT:* the untextured mesh rendered with flat albedo from 32 views (training camera model), lifted
  per view and averaged per point.
- No training. Both sides carry image features, so this is the "paint the CT" alternative to our
  missing-modality design.

**Results.** Synthetic test, primary setting (σ 1 mm at 100 mm, 10 % holes), 46 livers × 16 views.
Per-liver means; flat-albedo partials in parentheses.

| descriptor | success | inlier ratio @10 |
|---|---|---|
| FPFH | 0.670 (0.670) | 0.038 (0.038) |
| FPFH + DINO, zeros on the CT (S3) | 0.666 | 0.038 |
| dinolift | 0.834 (0.842) | 0.127 (0.135) |
| FPFH + dinolift | 0.825 (0.838) | 0.111 (0.120) |
| A_geo | 0.996 (0.996) | 0.349 (0.352) |
| A_mm (images on the partial only) | 0.996 (1.000) | 0.359 (0.358) |
| **A_geo + dinolift** | **0.999 (0.999)** | **0.469 (0.486)** |
| B_geo | 0.990 (0.992) | 0.147 (0.146) |
| B_mm (images on the partial only) | 0.999 (0.996) | 0.176 (0.178) |
| **B_geo + dinolift** | 0.993 (0.997) | **0.198 (0.210)** |

**Paired over livers** (inlier ratio, Wilcoxon):
- **Our descriptors vs dinolift alone:** A_mm 46 / 46 better, B_mm 44 / 46 (40 / 46 flat), p ≤ 3e-11.
- **A_geo + dinolift vs A_mm (img):** +0.110 (+0.128), better on **46 / 46** livers, p 3e-14.
- **B_geo + dinolift vs B_mm (img):** +0.022 (+0.032), better on **38 / 46** (40 / 46) livers,
  p ≤ 2e-6.

## Reading
- **Lifting DINOv2 onto a rendered, untextured CT is a strong training-free signal.** It reaches
  0.83 success against FPFH's 0.67, and does as well on flat-albedo partials. This again says the
  DINOv2 channel carries shading and shape, not texture.
- **Our learned missing-modality descriptors beat lifting alone,** but **lose to geometry +
  lifting**:
  - Concatenating our geometry-only descriptor with both-sides lifted DINOv2 beats our image-trained
    models on inlier ratio for both arms (arm A by +31 %, arm B by +13 %).
  - On synthetic data, "render the CT and lift on both sides" uses images better than "image on the
    partial, learned missing token on the CT".
- **Caveat: no domain gap here.** The synthetic partials and the CT renders come from the same
  renderer. On real data the partial-side features would come from real frames and the CT side from
  renders, and that gap is untested. DePoLL fails for every method (docs/r3a_depoll_whole.md), and
  no other real dataset with CT ground truth and RGB is usable (docs/r3b_datasets.md).
- **For the paper:** the missing-modality design is not the best way to use images in our own
  benchmark. A method paper would have to adopt geometry + both-sides lifting, or show the
  missing-modality model wins under a real domain gap. Neither is available now.

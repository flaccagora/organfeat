# S4: locked P2I-LReg evaluation (run once, 2026-10-26)

`scripts/eval/p2i_eval.py --split locked --run final`, OFEAT_UNLOCK_EVAL=p2i_lreg. Output:
`outputs/eval/p2i/locked_final.parquet`. 17 locked cases, 281 real keyframes, none skipped. The
audit and the projection rule are in docs/p2i_audit.md.

## Protocol
- **Partial side.** The provided reconstruction (Depth-Anything relative depth, scaled by the
  authors) at the liver-mask pixels, minus specular pixels, at 3 mm voxels.
- **CT side.** The preoperative model, in mm.
- **Pipeline.** The fixed one: mutual NN → RANSAC → ICP.
- **Metrics.** Dice and mean boundary distance (px) of the posed CT model, projected with each
  frame's fitted affine camera, against the annotated liver mask. The ceiling is the cloud's own
  projection. Chamfer distance (partial → model) is reported for consistency only.

## Results (per-case means; 17 cases)
| descriptor | Dice | boundary distance (px) | Dice ceiling | Chamfer (mm) |
|---|---|---|---|---|
| FPFH r20 | 0.536 | 79.7 | 0.881 | 6.19 |
| frozen Sonata | 0.517 | 91.8 | 0.881 | 8.50 |
| A_geo | 0.541 | 81.0 | 0.881 | 6.03 |
| A_mm (img) | 0.544 | 78.9 | 0.881 | 6.23 |
| A_mm :geo | 0.544 | 78.7 | 0.881 | 6.06 |
| B_geo | 0.538 | 79.7 | 0.881 | 6.17 |
| B_mm (img) | 0.538 | 80.5 | 0.881 | 6.30 |
| B_mm :geo | 0.534 | 80.6 | 0.881 | 6.19 |

The same pattern holds by case origin:
- the authors' own 13 cases: Dice 0.53 (frozen Sonata 0.50)
- the 4 locked P2ILF-derived cases: Dice 0.56–0.58

Paired Dice differences over the 17 cases (Wilcoxon):

| comparison | Δ Dice | cases better / worse | p |
|---|---|---|---|
| A_mm − A_geo | +0.003 | 12 / 5 | 0.24 |
| B_mm − B_geo | +0.000 | 8 / 9 | 0.61 |
| A_geo − FPFH | +0.005 | 10 / 7 | 0.22 |
| B_mm − FPFH | +0.002 | 9 / 8 | 0.52 |
| A_mm − FPFH | +0.009 | 12 / 5 | 0.05 |

## Null references (post hoc; no descriptor or setting changed)
- **Random pose** (`scripts/sanity/p2i_reference.py`): the CT model centred on the partial cloud,
  uniformly random rotation, 10 draws per frame, projected the same way. Result: Dice **0.498**,
  boundary distance 95.4 px (`outputs/eval/p2i/locked_random_pose_reference.parquet`).
  - every learned descriptor and FPFH beat it by +0.036–0.047 Dice (16–17 / 17 cases, p ≤ 4e-4)
  - frozen Sonata beats it by +0.019 (13 / 4, p 0.008)

  So the full pipeline sits above a random pose: ~0.54 Dice vs 0.50, against a 0.88 ceiling, i.e.
  about 10 % of the gap closed.
- **Random descriptors through the full pipeline** (i.i.d. Gaussian descriptors → mutual NN →
  RANSAC → ICP; `S4_p2i_null`, `outputs/eval/p2i/locked_null.parquet`). Result: Dice **0.532**,
  boundary distance 80.5 px, Chamfer 6.44 mm (own cases: 0.522).

  Paired against this null over the 17 cases (Wilcoxon; 8 comparisons, uncorrected p):

  | descriptor | Δ Dice | cases better / worse | p | Δ boundary distance (px) |
  |---|---|---|---|---|
  | FPFH r20 | +0.003 | 8 / 9 | 0.78 | −0.7 |
  | A_geo | +0.008 | 11 / 6 | 0.33 | +0.5 |
  | A_mm (img) | +0.012 | 10 / 7 | 0.12 | −1.6 |
  | A_mm :geo | +0.012 | 13 / 4 | 0.035 | −1.8 |
  | B_geo | +0.006 | 9 / 8 | 0.55 | −0.7 |
  | B_mm (img) | +0.006 | 11 / 6 | 0.33 | 0.0 |
  | B_mm :geo | +0.002 | 8 / 9 | 0.68 | +0.1 |
  | frozen Sonata | **−0.016** | 3 / 14 | 0.0004 | **+11.3** |

## Reading
- **On P2I-LReg the reprojection metric does not discriminate descriptors.** Random descriptors
  through the same pipeline reach 0.532 Dice; no learned descriptor or FPFH beats them reliably
  (best +0.012, p 0.035 uncorrected, not significant after correcting for 8 comparisons). The
  improvement over a random pose (0.498 → 0.53) comes from RANSAC + ICP on any matches, not from
  the descriptors.
- **Frozen Sonata is worse than random.** Structured but wrong matches drive RANSAC to confidently
  wrong poses, whereas random matches leave ICP close to the centroid alignment.
- **Why the metric is insensitive** (audit, docs/p2i_audit.md):
  - The provided intraoperative clouds come from monocular relative depth with an affine-like
    camera; they are not metric surfaces.
  - The masks cover large, smooth, mostly convex anterior surfaces, so a roughly centred liver
    silhouette already overlaps them.
  - The fitted cameras have a 12–49 px error floor (ceiling 0.88).
- **Consequence.** No conclusion about images or architectures can be drawn from P2I-LReg. The
  real-data evidence for those questions comes from StereoMIS (docs/s3_real_consistency.md):
  images help, PPF transfers, Sonata does not.

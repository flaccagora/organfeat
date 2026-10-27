# External benchmark: deformed IRCAD (Neri et al. 2025), post hoc (2026-10-27)

Run at the user's request after S4, to place the final descriptors next to published complete-to-partial
liver registration results. Nothing is selected or tuned on it. Descriptors, `configs/match.yaml` and
`configs/points.yaml` are unchanged (docs/protocol.md, Deviations, 2026-10-27).

- **Paper:** Neri, Penza, Haouchine, Mattos, *Benchmarking complete-to-partial point cloud
  registration techniques for laparoscopic surgery*, Front. Robot. AI 2025,
  doi:10.3389/frobt.2025.1702360, Table 2.
- **Run:** `scripts/eval/ircad_eval.py --run final`.
- **Output:** `outputs/eval/ircad/final.parquet`.
- **Tables:** `scripts/bench/ext_report.py --ircad outputs/eval/ircad/final.parquet`.

## Data
The paper's test sets are not released. We regenerated them with the authors' generator, copied with
minimal marked changes into `scripts/external/gen_deformed_ircad.py` (repo commit 2d297f5, raw copy in
`data/raw/deformed_ircad/`).

**Sets.** Livers 1–2 of 3D-IRCADb-01, 50 pairs each, at crop ratios 5 / 10 / 15 / 25 %
(175 / 350 / 525 / 875 partial points). The source is a 3,500-point subsample of the liver mesh.

**Each pair:**
- an ARAP displacement of one lobe, up to ±0.25 normalised units (~25 mm) in x and z
- a rigid transform: each Euler angle up to 45°, translation up to ±0.5 units
- a visibility crop from a virtual camera
- jitter σ 0.01 (~1 mm)

Coordinates are min-max normalised to [−1, 1] per axis.

**Reproducibility.** The seeds (7 + idx) fix the deformation, pose and camera of every pair. Two
draws were unseeded in the original, so the paper's exact samples cannot be reproduced:
- **Liver 1's point subsample.** We seed it (`np.random.seed(0)`).
- **The 80 hidden-region TRE landmarks** (farthest-point sampling). We seed them with idx.

**Validation of the regenerated sets:**
- **Liver 2:** identical to an earlier independent run of the same generator (defreg's 15 % file),
  in `gt_T`, source and partial.
- **Liver 1:** the poses are identical too.
- **GT residual TRE** (the TRE left by the true rigid transform, i.e. the deformation alone):

  | crop | here (mm) | paper (mm) |
  |---|---|---|
  | 15 % | 5.38 ± 4.92 | 5.37 ± 4.82 |
  | 25 % | 5.07 ± 4.72 | 5.01 ± 4.40 |
  | 5 % | 5.67 ± 5.05 | 6.45 ± 5.62 |
  | 10 % | 5.52 ± 4.95 | 6.17 ± 5.18 |

  The paper's 5 % and 10 % sets therefore differ from ours: across crop ratios only the hidden
  region changes here, never the deformation. **Treat 5 % and 10 % as approximate comparisons.**

## Protocol
- **Units.** The paper reports mm without saying how it converts. Per-axis de-normalisation
  (× (max − min) / 2) of target-frame differences reproduces its GT residual. ×100 does not
  (6.16 mm at 15 %). We use per-axis de-normalisation for TRE and MAE(t).
- **Metrics:**
  - **TRE:** the mean over the 80 landmarks. The ± is the SD pooled over all landmarks of all
    pairs, matching the paper's GT row.
  - **MAE(R):** the mean absolute error of the scipy "xyz" Euler angles, in degrees (the
    RPM-Net / OMNet convention).
  - **MAE(t):** the mean absolute translation error per component, in mm.
  - All are against `gt_T` (src → tgt), so even a perfect rigid fit of the visible part keeps the
    deformation in TRE and MAE.
- **Inputs to the fixed pipeline:**
  - Normalised coordinates × the liver's mean half-extent (~90 mm/unit), so the frozen mm thresholds
    apply. Rigid registration stays exact in that space.
  - **CT side:** the 3,500 source points with the mesh's vertex normals.
  - **Partial:** kNN-PCA normals facing the generator's camera. A real system knows the camera
    position, because intraoperative clouds are in the camera frame. The normals agree with the mesh
    normals at a median cosine of 0.98.
  - Clouds as given: no 3 mm voxelisation. They are already sparser, about 5 mm spacing.
- **Initial pose:**
  - `given`: the generator's poses, which is the paper's setting.
  - `so3`: an independent random rigid transform per cloud, which is this project's setting.
- **Also reported:**
  - `success`: this project's criterion, pose RMSE < 10 mm against the oracle rigid fit of the
    visible points.
  - `IR@10`: the inlier ratio at 10 mm.
  - `random`: i.i.d. descriptors through the same pipeline, as a null.
- **Descriptors:** geometry only, since there are no images. The multimodal checkpoints run as `:geo`.

## Results
Each cell gives TRE (mm, mean ± pooled SD) / MAE(R) (°) / MAE(t) (mm), with our `success` rate in
parentheses. Italic rows are the paper's Table 2. 100 pairs per crop ratio.

| init = given | 5 % | 10 % | 15 % | 25 % |
|---|---|---|---|---|
| *OMNet* | 23.49±14.14 / 9.00 / 7.20 | 15.23±8.36 / 5.77 / 5.29 | 11.72±6.45 / 4.37 / 3.82 | 10.45±4.81 / 3.86 / 3.73 |
| *Lepard* | 27.03±39.44 / 10.94 / 10.58 | 9.82±8.59 / 3.91 / 4.86 | 7.45±5.49 / 3.21 / 3.64 | 7.41±6.69 / 3.27 / 3.04 |
| *LiverMatch* | 18.69±28.58 / 7.47 / 8.41 | 7.91±5.90 / 3.64 / 4.68 | 6.39±4.54 / 3.08 / 3.73 | 4.81±3.21 / 2.78 / 2.78 |
| *Neri et al.* | 22.36±12.04 / 6.60 / 7.72 | 13.01±6.07 / 3.79 / 4.25 | 8.49±2.62 / 2.85 / 2.60 | 6.78±2.08 / 2.33 / 2.86 |
| *GT (residual)* | 6.45±5.62 / 0 / 0 | 6.17±5.18 / 0 / 0 | 5.37±4.82 / 0 / 0 | 5.01±4.40 / 0 / 0 |
| ours: GT (residual) | 5.67±5.05 / 0.00 / 0.00 | 5.52±4.95 / 0.00 / 0.00 | 5.38±4.92 / 0.00 / 0.00 | 5.07±4.72 / 0.00 / 0.00 |
| fpfh_r20 | 109.16±65.25 / 65.30 / 28.50 (0.09) | 63.04±67.72 / 38.92 / 14.26 (0.45) | 23.71±43.52 / 13.91 / 5.90 (0.79) | 6.50±16.21 / 4.69 / 2.95 (0.98) |
| sonata | 76.89±66.72 / 32.17 / 28.92 (0.26) | 66.62±62.08 / 31.64 / 22.31 (0.34) | 56.15±62.82 / 24.88 / 20.80 (0.49) | 34.15±50.47 / 17.34 / 10.98 (0.63) |
| A_geo | 61.90±72.06 / 30.74 / 21.90 (0.45) | 7.20±9.52 / 3.56 / 4.01 (0.98) | 5.70±4.78 / 3.00 / 3.62 (0.99) | 4.54±3.60 / 2.73 / 2.84 (1.00) |
| A_mm:geo | 58.76±67.33 / 30.22 / 18.42 (0.43) | 6.88±6.74 / 3.43 / 4.05 (0.97) | 6.24±7.85 / 3.24 / 3.66 (0.98) | 4.51±3.62 / 2.72 / 2.85 (1.00) |
| B_geo | 55.33±70.40 / 29.58 / 16.85 (0.58) | 8.75±16.05 / 4.38 / 4.54 (0.96) | 5.56±4.57 / 2.95 / 3.55 (1.00) | 4.55±3.76 / 2.77 / 2.87 (1.00) |
| B_mm:geo | 55.33±67.95 / 35.11 / 14.43 (0.51) | 7.25±8.58 / 3.60 / 4.14 (0.99) | 5.55±4.57 / 2.95 / 3.55 (1.00) | 4.45±3.55 / 2.72 / 2.82 (1.00) |
| random | 130.09±65.08 / 71.60 / 44.07 (0.01) | 113.52±64.72 / 69.45 / 36.47 (0.09) | 114.25±62.57 / 68.76 / 34.68 (0.07) | 104.40±64.56 / 66.58 / 27.43 (0.14) |

Same, with an independent random rigid pose per cloud (this project's setting; no published rows):

| init = so3 | 5 % | 10 % | 15 % | 25 % |
|---|---|---|---|---|
| ours: GT (residual) | 5.67±5.05 / 0.00 / 0.00 | 5.52±4.95 / 0.00 / 0.00 | 5.38±4.92 / 0.00 / 0.00 | 5.07±4.72 / 0.00 / 0.00 |
| fpfh_r20 | 111.50±64.66 / 66.57 / 31.54 (0.06) | 69.59±70.49 / 45.26 / 15.94 (0.42) | 25.36±44.97 / 15.38 / 7.38 (0.76) | 8.33±21.70 / 5.63 / 3.31 (0.96) |
| sonata | 123.15±62.61 / 70.40 / 35.42 (0.02) | 119.79±58.52 / 69.38 / 33.69 (0.02) | 111.54±55.96 / 69.27 / 29.55 (0.05) | 105.89±56.79 / 68.85 / 27.51 (0.08) |
| A_geo | 71.27±75.50 / 38.82 / 23.83 (0.40) | 7.73±11.75 / 4.38 / 4.02 (0.95) | 5.63±4.63 / 2.96 / 3.59 (1.00) | 4.39±3.46 / 2.67 / 2.80 (1.00) |
| A_mm:geo | 60.05±70.95 / 32.14 / 21.11 (0.47) | 11.48±29.25 / 5.99 / 5.66 (0.93) | 7.66±16.67 / 5.44 / 3.83 (0.98) | 4.58±3.75 / 2.70 / 2.91 (1.00) |
| B_geo | 59.93±70.30 / 34.26 / 17.92 (0.52) | 8.38±13.49 / 4.38 / 4.19 (0.96) | 5.56±4.57 / 2.96 / 3.55 (1.00) | 4.52±3.71 / 2.75 / 2.86 (1.00) |
| B_mm:geo | 51.68±66.70 / 31.29 / 14.03 (0.54) | 8.70±16.17 / 4.66 / 4.61 (0.98) | 5.57±4.58 / 2.96 / 3.56 (1.00) | 4.46±3.57 / 2.72 / 2.83 (1.00) |
| random | 127.32±60.83 / 71.02 / 42.65 (0.00) | 121.50±65.79 / 71.38 / 42.03 (0.05) | 109.82±64.64 / 65.00 / 31.70 (0.11) | 93.51±70.11 / 61.38 / 22.86 (0.23) |

**Paired comparisons** (per-pair TRE, Wilcoxon, `given`):

| crop | B_geo vs FPFH: better / worse, p | A_geo vs FPFH: better / worse, p |
|---|---|---|
| 5 % | 75 / 25, p 2e-10 | 75 / 25, p 3e-8 |
| 10 % | 80 / 20, p 2e-12 | 79 / 21, p 3e-12 |
| 15 % | 62 / 38, p 2e-4 | 56 / 44, p 0.003 |
| 25 % | 48 / 52, p 0.55 | 55 / 45, p 0.99 |

- Image-trained checkpoints run geometry-only vs geometry-only training (A_mm:geo vs A_geo,
  B_mm:geo vs B_geo): p ≥ 0.56 at every crop.
- Arm A vs arm B: p ≥ 0.09.

## Reading
- **At 15 % and 25 % visibility the learned descriptors reach the deformation floor.**
  - TRE is 5.6–6.2 / 4.5 mm against a GT residual of 5.4 / 5.1 mm. Success is 0.98–1.00.
  - That is below every published method: LiverMatch 6.39 / 4.81 mm, Neri et al. 8.49 / 6.78 mm.
  - MAE(R) and MAE(t) (3.0° / 3.6 mm and 2.7° / 2.8 mm) are on par with LiverMatch. Neri et al.
    is slightly lower on rotation.
  - A rigid fit of the visible part beats the true rigid part on the hidden landmarks in 56–62 %
    of pairs, so TRE against `gt_T` saturates here. Sub-millimetre differences between methods
    mean nothing.
- **At 10 %: on par or better, but approximate.**
  - The learned descriptors reach 6.9–8.8 mm (success 0.96–0.99), against LiverMatch 7.91 and
    Lepard 9.82.
  - Our 10 % set has a 0.65 mm lower GT residual than the paper's, so the comparison is approximate.
- **At 5 % the pipeline fails on about half the pairs.**
  - Success is 0.43–0.58, with mean TRE 55–62 mm and median 12–49 mm. That is worse than every
    published learned method (18.7–27.0 mm mean).
  - These partials have 175 points, fewer than this project's own `min_partial` (256). Our protocol
    never evaluates views that small, and the descriptors were never trained on them.
- **Learned vs FPFH.** Learned clearly beats FPFH where geometry is scarce (5–15 %). They tie at 25 %.
  FPFH alone is close to LiverMatch only at 25 % (6.5 mm, success 0.98).
- **Images during training change nothing at test time without images.** This matches the
  synthetic finding (docs/s3_ablations.md).
- **Initial pose.**
  - The learned descriptors and FPFH do not care: the pipeline is global, and `given` ≈ `so3`.
  - Frozen Sonata collapses under random poses (success 0.02–0.08 vs 0.26–0.63). Its `given`
    numbers come from the generator's limited rotation, not from invariance.
- **Null.** Random descriptors fail at every crop (TRE > 90 mm), so on these sparse clouds RANSAC +
  ICP does not succeed by geometric search alone. Contrast P2I-LReg (docs/s4_p2i.md,
  docs/ext_p2i_syn.md).

## Improving 5 % visibility (post hoc extension, 2026-10-27)
At the user's request ("can we improve 5 %"). Diagnosis and selection were done on a **dev set**: the
paper's training livers 3–20, seeds 1000+, 180 pairs at 5 % and 10 %
(`scripts/sanity/ircad_small_patch.py`, `outputs/eval/ircad_dev/`). The selected variant was then
scored **once** on the test pairs. `ofeat/match.py` is unchanged.

**Diagnosis (dev):**
- **RANSAC scoring is not the bottleneck.** Running all 100k iterations (confidence 1.0), or ranking
  hypotheses by descriptor consensus instead of geometric overlap, moves success by ±5 points. In
  failures the chosen pose overlaps *less* than the true one (0.77–0.88 vs 1.00), so the right
  hypothesis is never generated.
- **The descriptors give only 0–3 correct matches** in failing pairs; arm A gives ~12 mutual matches
  in total.
- **Not density, size or deformation.** Spacing is 4 mm (training: 3 mm), patches are ~90 mm across,
  and a rigid fit of the visible patch leaves 2–3 mm (~1.7 mm of it jitter).
- **Failures are mostly flips.** 61–92 % of failed poses rotate more than 75°, outside the
  generator's range (≤ 71°). The published learned methods are trained on that range, so they
  implicitly never leave it.

**Selected variant** (`ofeat/match_ext.py`, `ircad_eval.py --pipeline knn3_prior`):
- each partial point's 3 nearest CT descriptors as candidate matches (instead of mutual NN)
- torch RANSAC with the frozen sampling, checks and 7.5 mm threshold, ranked by geometric overlap
- a **rotation prior ≤ 75°**, set from the generator's range, not tuned
- then the frozen ICP

It was the best variant for every descriptor at both dev crops (5 %: arm A 0.61 → 0.79, arm B
0.42 → 0.71). Random descriptors plus the prior stay at 0.04, so the prior alone does not do the work.
It applies to the `given` setting only, because the prior is centred on the generator's poses.

**Test (one run):**

| 5 % | frozen: TRE mean (median) / success | knn3 + prior: TRE mean (median) / success | MAE(R) (°), frozen → prior |
|---|---|---|---|
| *LiverMatch* | 18.69 | | 7.47 |
| *Neri et al.* | 22.36 | | 6.60 |
| *OMNet / Lepard* | 23.49 / 27.03 | | 9.00 / 10.94 |
| A_geo | 61.9 (48.1) / 0.45 | 30.8 (12.5) / 0.61 | 30.7 → 13.3 |
| A_mm:geo | 58.8 (48.9) / 0.43 | 27.9 (11.0) / 0.67 | 30.2 → 11.1 |
| B_geo | 55.3 (12.3) / 0.58 | 26.5 (8.7) / 0.74 | 29.6 → 11.9 |
| B_mm:geo | 55.3 (17.4) / 0.51 | **24.5 (8.1) / 0.75** | 35.1 → 10.6 |
| FPFH | 109.2 (116.0) / 0.09 | 99.2 (103.0) / 0.19 | 65.3 → 40.4 |
| random | 130.1 (126.6) / 0.01 | 101.5 (94.0) / 0.03 | 71.6 → 40.2 |

- **At 5 %, mean TRE halves** (55–62 → 24.5–30.8 mm) and success rises to 0.61–0.75. The medians
  drop to 8–12.5 mm.
- **Against published methods:** comparable to Neri et al. (22.4), OMNet (23.5) and Lepard (27.0),
  still behind LiverMatch (18.7). Rotation error (10.6–13.3°) remains worse than all four (6.6–10.9°).
- **The mean is driven by the remaining ~25–40 % of failures.** Most pairs land near the floor.
- **No regression at 10–25 %:**
  - Learned descriptors stay at success 0.99–1.00, with mean TRE equal or slightly lower (10 %:
    6.6–8.0 mm; 15 %: 5.6; 25 %: 4.4–4.5).
  - FPFH gains most: 15 % 23.7 → 9.0 mm, 25 % 6.5 → 4.4 mm.
- **The prior also lifts the random null at large crops** (success 0.40 at 25 %). There, overlap search
  within ±75° can find the pose without descriptors. At 5 % it does not (0.03).
- **Caveats:**
  - The prior is benchmark knowledge, though clinically plausible (rough liver orientation in the
    laparoscope frame).
  - The 5 % test set's GT residual differs from the paper's (5.67 vs 6.45 mm).
  - No significance tests: 100 pairs from 2 livers.
  - The `so3` results above remain prior-free.

Full per-crop tables: `scripts/bench/ext_report.py --ircad outputs/eval/ircad/final.parquet
--ircad-prior outputs/eval/ircad/knn3_prior.parquet`.

## Caveats
- **Post hoc, on regenerated data.** Only the 15 % and 25 % sets are validated against the paper,
  through the GT residual. 5 % and 10 % differ.
- **Small sample.** 2 livers × 50 pairs, so pairs are not independent and no CIs are given.
- **The mm conversion is inferred.** It is per-axis, the only convention that reproduces the
  paper's GT row.
- **Inputs differ from the published methods:**
  - Our CT side carries mesh normals, and our partial normals use the generator's camera position.
    The published methods take points only.
  - Our clouds are scaled from normalised units to ~mm, so the frozen mm thresholds apply. The
    learned models were trained on real-proportioned livers; here they see per-axis-normalised ones.
- **Rigid only, as in the paper.** No method can go below the deformation floor.

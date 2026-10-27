# External benchmark: P2I-LReg synthetic test split vs Self-P2IR, post hoc (2026-10-27)

Run at the user's request after S4, to compare with the rigid-registration table of the paper that
released P2I-LReg. Nothing is selected or tuned on it. Descriptors, `configs/match.yaml` and
`configs/points.yaml` are unchanged (docs/protocol.md, Deviations, 2026-10-27).

- **Paper:** Self-P2IR, arXiv 2504.15152, Table I (RANSAC-50k rows) and Table IV.
- **Run:** `scripts/eval/p2i_syn_eval.py`, three case groups (`--run final_g1/g2/g3`), with
  `OFEAT_UNLOCK_EVAL=p2i_lreg`.
- **Output:** `outputs/eval/p2i_syn/final_g{1,2,3}.parquet`.
- **Tables:** `scripts/bench/ext_report.py --p2i-syn outputs/eval/p2i_syn/final_g*.parquet`.

## Data
The Kaggle release ships, per case, synthetic renders of the preoperative model (`<case>/syn`):
- the visible liver points in the camera frame, mm, dense and noise-free
- the exact camera-to-model pose (`camPose.yml`)
- label maps, but **no RGB**
- the authors' split lists (`train_syn.txt`, `test_syn.txt`)

We use every `test_syn.txt` frame: **2,800 frames over all 21 cases** (100 or 150 per case, frames
01900 onward). 7 are skipped: 6 render at most one liver point (liver out of view), 1 renders 218 points
(< `min_partial` 256). That leaves **2,793 scored**.

**Check:** moving the points by camPose puts them on the model (median 0.46 mm, dev case 15), so the
ground truth is exact.

**Locked cases.** 17 of the 21 cases are locked, so the run is unlocked. These synthetic frames are
not the locked real frames: those were evaluated once in docs/s4_p2i.md and are not rerun.

## Protocol
- **Partial:** the rendered points at 3 mm voxels (cap 8,192; median 3,000 points, 10–90 %:
  2,150–4,140), with kNN-PCA normals facing the camera.
- **CT:** the preoperative model, sampled as `configs/points.yaml` (9,500–18,900 points).
- **Pose:** each side gets its own random rigid transform. This project's setting; the paper does
  not state its initial poses.
- **Pipeline:** the fixed one (mutual NN → RANSAC 100k → ICP).
- **Metric mapping** to Self-P2IR (τ1 = 1 cm, τ2 = 5 %, ρ = 10 mm):
  - **IR:** the inlier ratio at 10 mm. Self-P2IR computes it on its top-k scored correspondences
    (k = 250–2000), so we report the top-1000 and top-250 mutual matches by ratio-test confidence,
    plus all mutual matches.
  - **FMR:** IR > 5 %. Here it uses IR over all mutual matches (`fmr` of `pair_metrics`).
  - **RR:** the fraction with pose RMSE < 10 mm. Table I's RR is at ρ = 10 mm: Table IV's
    λ_tran = 1 row equals Table I's RANSAC-50k / 1000 row. Also reported at 5 and 2 mm (Table IV).
  - **RRE (°) / RTE (mm):** means over the RR@10 successes, the usual convention.
- **Not comparable:**
  - The split differs. The paper says it took a random 20 % for testing; the released lists hold
    ~6 % (the last frames).
  - The paper does not say which cases its synthetic evaluation covers.
  - Its correspondences come from its own matcher and top-k sampling, not mutual NN.

## Results
2,793 scored frames, 21 cases. Italic rows are Self-P2IR Table I (RANSAC-50k, with 1,000 or 250
samples). Ours run through the fixed pipeline. Our IR (top-k) is the top-1,000 mutual matches by
ratio-test confidence. RRE and RTE are means over RR@10 successes.

| method | FMR | IR (top-k) | RR@10 | RRE (°) | RTE (mm) |
|---|---|---|---|---|---|
| *Predator (1000)* | 95.3 | 73.9 | 89.0 | 0.288 | 2.16 |
| *GeoTrans (1000)* | 96.9 | 94.7 | 93.5 | 0.224 | 1.44 |
| *Self-P2I (1000)* | 96.8 | 95.0 | 99.0 | 0.190 | 1.21 |
| *Self-P2I (250)* | 96.8 | 98.1 | 98.8 | 0.194 | 1.24 |
| fpfh_r20 | 99.5 | 19.3 | 99.9 | 0.133 | 0.40 |
| sonata | 10.0 | 1.3 | 16.6 | 0.149 | 0.44 |
| A_geo | 98.8 | 38.4 | 98.6 | 0.130 | 0.40 |
| A_mm:geo | 98.4 | 38.2 | 97.9 | 0.127 | 0.40 |
| B_geo | 100.0 | 72.3 | 100.0 | 0.134 | 0.41 |
| B_mm:geo | 100.0 | 70.9 | 100.0 | 0.134 | 0.41 |
| random | 0.0 | 0.4 | 29.2 | 0.117 | 0.41 |

Our extra columns:
- **IR top-250 / IR all:** the inlier ratio on the 250 most confident mutual matches, and on all of
  them. Median mutual matches per frame: arm A 73, arm B 1,030, FPFH 1,231, frozen Sonata 20. So
  arm A's top-k equals all.
- **RR@5 / RR@2:** registration recall at 5 and 2 mm.
- **RANSAC only:** RR@10, RRE and RTE before ICP. Self-P2IR reports no ICP.
- **Worst-case case:** the lowest per-case RR@10.

| method | IR top-250 | IR all | RR@5 | RR@2 | RR@10 RANSAC only | RRE / RTE RANSAC only | worst case RR@10 |
|---|---|---|---|---|---|---|---|
| fpfh_r20 | 23.5 | 18.4 | 99.8 | 99.2 | 99.0 | 2.180 / 1.94 | 98.7 |
| sonata | 1.3 | 1.3 | 16.6 | 16.5 | 1.1 | 4.690 / 5.14 | 9.3 |
| A_geo | 38.4 | 38.4 | 98.5 | 98.0 | 92.6 | 3.648 / 4.27 | 86.9 |
| A_mm:geo | 38.2 | 38.2 | 97.8 | 97.4 | 90.9 | 3.647 / 4.33 | 79.8 |
| B_geo | 88.4 | 70.5 | 99.9 | 99.3 | 99.8 | 1.445 / 1.24 | 99.3 |
| B_mm:geo | 87.1 | 69.3 | 99.9 | 99.3 | 99.8 | 1.485 / 1.26 | 99.3 |
| random | 0.4 | 0.4 | 29.2 | 29.1 | 0.5 | 7.201 / 5.38 | 17.0 |

By case group (RR@10 % / IR top-1000 %):
- dev cases (4): fpfh_r20 RR 99.8 / IR 17.0, sonata RR 14.3 / IR 1.3, A_geo RR 98.8 / IR 39.8, A_mm:geo RR 98.0 / IR 39.7, B_geo RR 100.0 / IR 71.8, B_mm:geo RR 100.0 / IR 70.1, random RR 29.5 / IR 0.4
- locked cases (17): fpfh_r20 RR 99.9 / IR 19.7, sonata RR 17.2 / IR 1.3, A_geo RR 98.5 / IR 38.1, A_mm:geo RR 97.8 / IR 37.8, B_geo RR 100.0 / IR 72.4, B_mm:geo RR 100.0 / IR 71.1, random RR 29.2 / IR 0.4

Per-case RR@10 (not tabulated):
- B_geo / B_mm:geo: ≥ 0.99 in every case. FPFH: ≥ 0.987.
- Arm A: 0.96–1.00 except case 21, at 0.87 (A_geo) and 0.80 (A_mm:geo).
- Random: 0.17–0.40.

## Reading
- **Registration recall is saturated.**
  - Arm B (PPF) registers 100.0 % of frames and FPFH 99.9 %, against Self-P2I's 99.0 % and
    GeoTransformer's 93.5 %.
  - Before ICP, arm B is at 99.8 % with RTE 1.24 mm, equal to Self-P2I's 1.21 mm. Its RRE is
    1.4° vs 0.19°: our RANSAC runs on 3 mm voxels with a 7.5 mm threshold.
  - After ICP, every successful method reaches 0.13° / 0.4 mm.
  - These renders are exact, noise-free, dense subsets of the very surface they are matched to, a
    near-ideal case for any geometric descriptor.
- **Even random descriptors register 29 % of frames.** It is the same mechanism as on the real
  frames (docs/s4_p2i.md):
  - Open3D's RANSAC scores hypotheses by geometric overlap.
  - ICP then pulls overlapping poses into place. Among the null's successes, the RANSAC pose was
    off by a median 24° (10–90 %: 11–51°).

  So RR alone overstates descriptor quality here. The inlier ratio is the discriminating number:
  random 0.4 %, frozen Sonata 1.3 %, FPFH 19 %, arm A 38 %, arm B 72 % (top-1000) and 88 % (top-250).
- **Inlier ratio is below Self-P2I's** (95–98 % on its top-k). The protocols differ: Self-P2I scores
  dense predicted correspondences with a learned confidence, while we take mutual nearest neighbours
  of fixed descriptors ranked by a ratio test. Its IR depends on its sampling. Only arm B's top-250
  (88 %) comes near.
- **Arms.** Arm B transfers here as on StereoMIS (docs/s3_real_consistency.md), with RR 100 % in
  every case. Arm A is weaker: RR 98 %, worst case 0.80–0.87, and a median of only 73 mutual
  matches per frame.
  Frozen Sonata fails (17 %, below the random null), as on the synthetic test.
- **No image question.** There is no RGB. The multimodal checkpoints run geometry-only and match
  geometry-only training (B: 100.0 vs 100.0 %, A: 97.9 vs 98.6 %).
- **Same in dev and locked cases.** No sign that the 4 dev cases are easier.

## Caveats
- **Post hoc.**
- **The split differs:** the released `test_syn.txt` (~6 % of frames per case) vs the paper's
  random 20 %. The lists hold 5–6 % per case.
- **Pose setting unknown.** We use random rigid poses per side; the paper does not state its
  initial poses.
- **Not the same pipeline.** Our numbers come from the fixed mutual-NN → RANSAC-100k → ICP pipeline,
  not from Self-P2IR's matcher.
- **Frames are correlated** within a case (consecutive renders).
- **This says nothing about real data.** On the real P2I-LReg frames every method, including random
  descriptors, sits at ~0.54 Dice (docs/s4_p2i.md). The synthetic split cannot detect that gap.

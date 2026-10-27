# Pre-registered protocol (S0, 2026-10-26)

Fixed before any descriptor is trained and before any baseline is scored. Changes after this date are
added under **Deviations** with a date and a reason; nothing above that section is edited silently.

## 1. Question and scope
Do self-supervised per-point descriptors that combine surface geometry with frozen DINOv2 features,
lifted onto visible points, improve **rigid** matching of a partial visible liver surface to its
complete CT surface when the CT side has no appearance? The organ is the liver, matching is rigid, and
the descriptor is the only variable. Mild deformation is a separate stress test, not the core scope.
CT Hounsfield values are never treated as colour. Synthetic colour is a nuisance and is never treated
as observed anatomy.

## 2. Data roles and splits (frozen)
| set | role | split |
|---|---|---|
| TotalSeg livers (human CT) | synthetic training, selection, testing | official split, 408 / 20 / 46 (`configs/splits/totalseg_liver.yaml`; meshes < 300 ml excluded) |
| StereoMIS (porcine) | real RGB/geometry consistency, no CT | P2_0–P2_5 train, P3 test (`configs/splits/stereomis.yaml`) |
| DePoLL (porcine) | **locked** real paired evaluation, run once | all states |
| P2I-LReg (human) | audit, then dev and **locked** subsets | patient split fixed at the audit, before any descriptor runs on it |

- **Synthetic views.** They are rendered per split into `data/processed/views/<split>/` *after* the
  split is fixed. Train livers have 16 groups × 3 views. Val and test livers have 8 groups × 2
  views. Stress-test views are rendered from test livers only.
- **Locked sets.** Locked datasets raise `LockedDatasetError` unless `OFEAT_UNLOCK_EVAL=<name>`.
  The only exception is the dev part of P2I-LReg after the audit, which is used for format and scale
  debugging only and never for selection.
- **P2I-LReg go/no-go.** If P2I-LReg is not downloaded and audited as usable by **2026-11-09**, it
  is dropped. Conclusions are then limited to synthetic matching, StereoMIS consistency and DePoLL,
  with no clinical CT-registration claims.

## 3. Inputs (frozen: `configs/points.yaml`)
- **Sampling.** Both sides get one point per 3 mm voxel. Caps are 8192 points for the partial cloud
  and 32768 for the CT cloud. Views with fewer than 256 visible points are skipped.
- **Partial points.** Pixels that are organ, not tool, not specular (8-bit HSV V > 240 and S < 40,
  dilated 2 px) and have observed depth > 0. They are back-projected at pixel centres.
- **Normals.** k-NN PCA (k = 16) on both sides, oriented outward: towards the camera on the partial
  side, along the mesh normals on the CT side.
- **Geometric channels.** Shape index and log(1 + curvedness · r) at r = 6 and 15 mm, from a
  quadric fit (`ofeat/features/geom.py`).
- **Visual channel (partial side only).**
  - DINOv2 ViT-L/14 with registers, last-layer patch tokens on 644 × 518 frames (46 × 37 tokens).
  - Bilinearly lifted to each point's pixel.
  - Reduced by PCA to 64 dimensions, fitted on train-split renders only.
  - The CT side has no visual channel: the *missing* path.
- **Leakage controls.**
  - Each cloud gets its own uniform random SO(3) rotation plus translation, then is centred.
  - Ground-truth correspondences (exact surface points) are used only in losses and metrics.
  - Textures are redrawn per view group from seeds unrelated to vertex ids. Two texture families
    (stripes, cells) are never used in training.

## 4. Fixed matching pipeline (frozen: `configs/match.yaml`, `ofeat/match.py`)
1. **Matching.** Cosine mutual nearest neighbours of L2-normalised descriptors. The ratio-test
   confidence 1 − d1/d2 is kept for coverage curves only.
2. **RANSAC.** Open3D correspondence RANSAC:
   - n = 3, inlier distance 7.5 mm
   - edge-length checker 0.9 and distance checker 7.5 mm
   - 100 000 iterations, confidence 0.999, seed 0
3. **Refinement.** Point-to-plane ICP (partial → CT), 5 mm, 50 iterations.

Every method uses this pipeline with these parameters. It is never tuned.

## 5. Metrics (`ofeat/eval/metrics.py`)
Each pair has partial points P (source input frame), their true positions G in the CT input frame,
and CT points Q. The oracle rigid transform is T* = kabsch(P, G). It is exact when rigid and
noise-free.

| metric | definition |
|---|---|
| point recall@τ | fraction of partial points whose descriptor nearest neighbour in Q lies within τ of the point's true position (τ = 5, 10, 20 mm) |
| inlier ratio@τ | the same, over mutual matches only |
| feature match recall (FMR) | inlier ratio@10 > 5 % |
| coverage curve, AUC | inlier ratio@10 of the top-q most confident mutual matches, q = 5 %…100 % in 20 steps; AUC = mean |
| RRE, RTE | rotation (deg) and translation (mm) error of T vs T*, after RANSAC and after ICP |
| pose RMSE | RMSE over P of T(P) − T*(P) |
| **registration success** | **pose RMSE after ICP < 10 mm** (primary). Also reported: RRE < 10° and RTE < 10 mm |
| runtime, memory | wall clock per stage (descriptor incl. DINOv2, matching, RANSAC, ICP) on one A100; peak GPU and CPU memory |

- **Primary endpoint** (amended 2026-10-26, see Deviations). Registration recall, i.e. the mean
  success, on the synthetic **test** livers with stereo-like depth noise: σ = 1 mm at 100 mm,
  10 % holes (`run_bench.py --sigma100 1 --holes 0.1`). The noise-free base setting and the
  other noise levels are secondary.
- **Secondary endpoints.** Inlier ratio@10, FMR, point recall@10 and coverage AUC.
- **Aggregation.** Per liver, average over its views; then average over livers. Views of one
  liver are not independent.
- **Comparisons.** Paired Wilcoxon signed-rank tests over test livers, plus 95 % bootstrap CIs
  over livers (10 000 resamples).
- **Primary comparison.** Multimodal descriptor vs geometry-only descriptor, then each vs FPFH.

## 6. Methods
- **Baselines:**
  - FPFH. The radius is chosen from {10, 20, 30} mm on val by registration recall. Chosen: 20 mm
    (val, σ = 1 mm: 0.631 vs 0.628 for 30 mm and 0.556 for 10 mm).
  - Frozen Sonata: raw 1088-d, and PCA-32 fitted on train CT clouds.
  - Simple concatenation: [FPFH ‖ DINO-PCA] and [Sonata-PCA ‖ DINO-PCA]. Each block is
    L2-normalised and weighted equally; the CT side gets a zero visual block.
- **Ours, arm A (pre-registered).** A Sonata (PTv3) encoder, fine-tuned:
  - a zero-initialised visual adapter and a learned *missing* token
  - a 32-d L2-normalised head
  - multi-positive InfoNCE: τ = 0.07, positives within 4 mm, 12 mm safe radius for negatives
  - cross-modal consistency and view-pair terms
  - visual-modality dropout: 0.5 per sample, 0.1 per point
- **Ours, arm B (added in S1, see Deviations).** A strictly local PPF-PointNet:
  - rotation-invariant point-pair features over 10 / 25 mm balls, shared MLP, max + mean pooling
  - the same geometric and visual channels, the same missing vector, loss and dropout
  - trained from scratch (lr 1e-3)
  - selection, testing and ablations follow arm A's rules, per arm
- **Ablations:**
  - visual features: none / training only (geometry-only inference) / training and inference
  - dropout on/off
  - consistency term on/off
  - training livers 25 / 50 / 100 %
  - Sonata initialisation vs scratch
- **RANSAC bound (S0).** Registration success vs injected true-inlier ratio among a fixed number of
  matches, on val livers, with this pipeline. This shows how good any descriptor must be.

## 7. Selection and testing
- **Selection.** Uses synthetic **val** livers only, in the primary setting (σ = 1 mm, 10 %
  holes). The criterion is registration recall, with inlier ratio@10 as tie-break. This covers checkpoints, the FPFH radius and the small grids (learning
  rate; λ for consistency; μ for view pairs). Grids are listed in the training config before the run.
- **Testing.** The test set is scored once per final configuration. Stress tests are scored once,
  after the main test.
- **Base test setting** (amended 2026-10-26). Train texture families and the renderer's nuisance
  distribution, with stereo-like depth noise (σ = 1 mm at 100 mm, 10 % holes) and no deformation.
  The original noise-free setting is reported as a secondary result.
- **Stress tests.** Each varies one factor from the base, on test livers. Since the 2026-10-26
  amendment the base is the primary setting (σ = 1 mm, 10 % holes); the noise-free setting is one
  level of the depth-noise stress (`scripts/bench/s3_stress.sh`). The factors:
  - held-out textures (stripes, cells) and untextured (flat albedo)
  - occlusion: 0 / 1 / 2 / 3 tools
  - visible fraction bins: < 10 %, 10–20 %, 20–30 %, ≥ 30 % (rendered surface-area fraction)
  - depth noise: σ at 100 mm ∈ {0, 0.5, 1, 2} mm, with and without 10 % holes
  - mild deformation: `organ_sim` ARAP, handles displaced 2–8 mm. Success is judged against the
    oracle rigid fit.
- **S1 gate.** If the learned geometry-only descriptor does not beat FPFH (r20) on val registration
  recall in the primary setting, the receptive field and the safe radius are diagnosed before images are added. The
  outcome is reported either way.

## 8. Real data
- **StereoMIS (no CT): RGB/geometry consistency.**
  - Frame pairs within a sequence, at most 600 video frames apart, binned by kinematic camera
    translation: 5–10, 10–20 and 20–40 mm.
  - SGBM depth; liver points come from the frozen DINOv2 liver probe; dataset tool masks are applied.
  - Pseudo ground truth is the relative pose from forward kinematics. Its convention was checked by
    depth warping on P2 only: camera-to-world, metres, best on 58 / 60 moving pairs.
  - Metrics: point recall@10, pair RRE/RTE, and success (RRE < 15° and RTE < 20 mm),
    geometry-only vs geometry + image, on **P3** only. Registration with the pseudo-GT's own
    correspondences (`--oracle`) is reported as the floor.
  - Breathing motion and kinematic error limit this test. It is not CT registration.
- **DePoLL (locked, run once with `OFEAT_UNLOCK_EVAL=depoll`).**
  - Per state: the Poisson mesh of the coloured stereo reconstruction, rendered from 24 virtual views
    (as in correspondences stage 5). Liver points come from the frozen probe. The CT side is the
    reference surface (`surfaceFull.ply`).
  - Ground truth: the 60 marker pairs. The oracle rigid fit is the best rigid fit of the markers
    (reconstruction frame → reference CT frame).
  - Metrics: marker TRE after our T vs the oracle-rigid TRE; RRE and RTE vs the oracle. Success is
    TRE ≤ oracle TRE + 10 mm.
  - The human (training) → porcine (test) shape gap is reported as a limitation.
- **P2I-LReg.**
  - Audit: RGB, masks, intrinsics, cloud↔pixel mapping, scale, patient ids, and overlap with P2ILF.
  - Reconstructed clouds are used as given.
  - Metrics: Dice and mean boundary distance (px) of the CT model projected with our pose vs the
    annotated liver mask. Chamfer distance to the provided cloud is reported as consistency only;
    estimated depth is not dense ground truth.
  - **Projection** (fixed by the audit, `docs/p2i_audit.md`). The provided clouds are not pinhole
    back-projections with CAM_K, so the model is projected with each frame's affine camera, fitted
    to that frame's own cloud-to-pixel correspondences. Each frame's reprojection ceiling (the Dice
    of the cloud itself, projected the same way) is reported next to the results.
  - Results are reported with and without the P2ILF-derived patients.

## Deviations
- 2026-10-26 (S0, after a one-liver pipeline smoke of FPFH and frozen Sonata, before any full scoring
  or training): the point caps were raised from
  4096 / 16384 to 8192 / 32768. With the old caps, 72 of 411 train CT clouds were randomly
  thinned, so CT and partial densities differed on large livers. The caps now only guard memory.
- 2026-10-26 (S0, before any result): three train meshes with status "ok" are 50–60 ml fragments
  (s0067, s0361, s1137), whereas the next smallest is 693 ml. Livers under 300 ml are now excluded,
  leaving 408 / 20 / 46. Val and test are unchanged.
- 2026-10-26 (S0, before any descriptor was run on StereoMIS): StereoMIS pairs are now selected by
  kinematic camera translation (5–10 / 10–20 / 20–40 mm, at most 600 frames apart) instead of fixed
  gaps of {5, 10, 25, 50} frames. Over 6–30 frames the camera barely moves: the median step is
  0.02–0.1 mm per 6 frames, and the pose-convention check could not tell any hypothesis from "no
  motion" (`scripts/sanity/stereomis_poses.py`).
- 2026-10-26 (S1, after val baselines, before any learned descriptor was scored): **primary
  endpoint amended** with the user.
  - **Why.** On noise-free val views, FPFH already registers 100 % of pairs at every radius
    (inlier ratio@10 0.14–0.27). The RANSAC bound shows that 3–5 % true inliers are enough. The
    pre-registered base setting therefore cannot show an improvement.
  - **Val registration recall under depth noise with 10 % holes:**

    | σ at 100 mm | FPFH r10 / r20 / r30 | frozen Sonata |
    |---|---|---|
    | 0.5 mm | 0.77 / 0.90 / 0.87 | 0.14 |
    | 1 mm | 0.56 / 0.63 / 0.63 | 0.17 |
    | 2 mm | 0.25 / 0.29 / 0.22 | 0.12 |

  - **New primary setting.** σ = 1 mm with 10 % holes, a realistic stereo error level with
    headroom in both directions. Selection uses the same setting on val.
- 2026-10-26 (S1, with the user): **a second architecture arm (B: local PPF-PointNet)** was added.
  - **Origin.** It was an S1 receptive-field diagnostic. On val (2 k steps) it matched arm A in the
    primary setting (0.959 vs 0.953 registration recall; inlier ratio 0.082 vs 0.156).
  - **Noise-free** it was far more precise: inlier ratio 0.58 vs 0.23 (recall 1.000 vs 0.975).
  - **Scope.** Arm A stays the pre-registered model. Arm B goes through the same geometry-only /
    multimodal comparisons with its own val selection, so the image question is answered on two
    backbones.
- 2026-10-26 (S1, training recipe selected on val): fp32 and head lr 1e-4 (was bf16 and 1e-3, which
  collapsed). Encoder-lr grid {2e-5, 1e-4}. See docs/s1_findings.md.
- 2026-10-26 (S1, P2I-LReg go/no-go): **P2I-LReg is usable**, so it stays in the study.
  - Audit in `docs/p2i_audit.md`: 21 cases, 344 real frames; cases 01–06 are P2ILF-derived.
  - Split: dev 03 / 05 / 15 / 19, the other 17 locked.
  - The reprojection metric uses fitted affine cameras, because the given clouds are inconsistent
    with CAM_K.
  - Decided before any descriptor was run on the data.
- 2026-10-26 (S1, before any learned descriptor ran on StereoMIS): **StereoMIS success criterion set
  to RRE < 15° and RTE < 20 mm.**
  - *Why:* a mechanics smoke on P2 (training animal, FPFH only) showed that registering with the
    pseudo-GT's own correspondences gives a median 6.7° / 9.6 mm. Only 48 % of those pairs pass
    10° / 10 mm, so the kinematic + breathing + SGBM error floor is ~5–10 mm.
  - *Also seen:* on small, near-planar patches FPFH often lands on in-plane ~180° flips.
- 2026-10-26 (S3, **after** seeing the first P3 StereoMIS run): **StereoMIS consistency is also
  reported on P2**, as a second real animal. No descriptor trains or selects on StereoMIS; P2 only
  trained the frozen liver probe (Project 10) and was used for the FPFH-only mechanics smoke.
  - *Why:* P3 yielded only 12 usable pairs out of the 90 requested. The probe has low recall on P3,
    so few pairs pass the point-count and overlap checks.
  - *P3 pseudo-GT is weak:* the oracle has a median of 15° / 6.5 mm and passes only 50 %.
  - *Reporting:* P3 is reported as pre-registered, P2 separately, and both as underpowered
    consistency checks.
  - *Diagnostics added:* the number of mutual matches and how many fall in the overlap. In the
    first P3 run arm A had an inlier ratio of 0 in every bin while PPF did not.

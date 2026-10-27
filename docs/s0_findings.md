# S0 findings (2026-10-26)

## Done
- **Repo.** `/leonardo_work/IscrC_FLAC/organfeat` (package `ofeat`) with its own venv, container,
  wheels and data dir. Code copied from correspondences (and through it from defreg) carries origin
  notes; nothing imports either repo.
- **Data copied.**
  - TotalSeg liver meshes (531 files; 477 "ok", of which 474 are ≥ 300 ml)
  - Sonata weights
  - the DINOv2 liver probe
  - DePoLL without videos or DICOM (111 MB)
  - StereoMIS liver sequences P2_0–P2_5 and P3: frames, SGBM depth, tool masks, calibration,
    kinematics (5.2 GB)
- **Protocol.** `docs/protocol.md`, plus `configs/match.yaml` and `configs/points.yaml`.
- **Fixed pipeline and metrics.** `ofeat/match.py`, `ofeat/eval/metrics.py`, with unit tests (90 %
  outlier recovery, exact metrics on toy cases).
- **Synthetic caches.** In `data/processed/views/`:
  - train: 408 livers × 48 views (16 groups of 3)
  - val: 20 × 16
  - test: 46 × 16
  - 7 stress sets on test livers, paired cameras: held-out textures, flat, 0/1/2/3 tools, ARAP 2–8 mm
- **Ground-truth sanity** (`scripts/sanity/check_sets.py`). The oracle-rigid residual is 0.00 mm on
  every rigid set. On `test_deform` it has median 1.8 mm and max 3.9 mm. Median partial cloud is
  ~2.7–3.3 k points, CT ~14 k, visible surface fraction ~0.2.
- **CT cache.** 4 samplings per train liver (`data/processed/ct_cache`).

## Issues found and fixed before any full scoring
1. **Background artefacts.** Cubic upsampling in the tissue background overshot [0, 1] and left
   off-colour blobs (background pixels only). Fixed, and every cache was regenerated.
2. **Point caps** (protocol deviation 1). 72 train CT clouds hit the 16 k cap, so the caps were raised.
3. **Liver fragments** (protocol deviation 2). Three "ok" meshes are 50–60 ml fragments; meshes
   < 300 ml are excluded.
4. **Non-deterministic PTv3.** PTv3 shuffles its serialization order from the global RNG. All
   inference is now seeded (extractor, frozen Sonata, training proxy).
5. **CUDA in data workers.** Normal estimation tried to use CUDA inside forked DataLoader workers. It
   now runs on CPU there.

6. **StereoMIS kinematic poses.** These are camera-to-world, in metres: this hypothesis is best on 58 / 60
   moving pairs, with a median depth residual of 2.8 mm vs 13.6 mm for identity. The camera is mostly
   static (median 0.02–0.1 mm per 6 frames), so pairs are selected by camera motion, not by frame gap.

## RANSAC bound (val, descriptor-free; `scripts/bench/ransac_bound.py`)
K matches per pair, a fraction ρ of them correct, the rest random; 20 val livers × 4 views.

| K \ ρ | 0.01 | 0.02 | 0.03 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|
| 250 | 0.30 | 0.51 | 1.0 | 1.0 | 1.0 | 1.0 |
| 500 | 0.39 | 1.00 | 0.9 | 1.0 | 1.0 | 1.0 |
| 1000 | 0.60 | 0.80 | 0.9 | 1.0 | 1.0 | 1.0 |

The frozen RANSAC + ICP succeeds once ~3–5 % of mutual matches are correct.

## Baselines on val (20 livers × 16 views; per-liver means)
Registration recall, with inlier ratio@10 in brackets. Depth noise columns have 10 % holes.

| descriptor | noise-free | σ 0.5 mm | **σ 1 mm (primary)** | σ 2 mm |
|---|---|---|---|---|
| FPFH r10 | 1.000 (0.139) | 0.772 (0.036) | 0.556 (0.021) | 0.250 (0.010) |
| **FPFH r20** (selected) | 1.000 (0.267) | 0.897 (0.065) | **0.631 (0.034)** | 0.291 (0.013) |
| FPFH r30 | 1.000 (0.269) | 0.872 (0.066) | 0.628 (0.034) | 0.219 (0.013) |
| frozen Sonata (1088-d) | 0.169 (0.020) | 0.141 (0.015) | 0.172 (0.012) | 0.119 (0.009) |
| frozen Sonata PCA-32 | 0.119 (0.017) | – | 0.097 (0.016) | – |
| FPFH r20 ‖ DINO (CT: zeros) | 1.000 (0.267) | – | 0.641 (0.034) | – |
| FPFH r30 ‖ DINO (CT: zeros) | 1.000 (0.269) | – | 0.625 (0.034) | – |
| Sonata-PCA ‖ DINO (CT: zeros) | 0.144 (0.018) | – | 0.122 (0.014) | – |

- **Concatenation is a no-op.** Concatenating an image block that is zero on the CT side cannot change
  cosine nearest neighbours: it adds a per-source constant. So "geometry ‖ DINO" equals the geometry
  descriptor; FPFH r20 ‖ DINO gives 0.641 vs FPFH r20 0.631. That 1-point gap on identical matches is
  the run-to-run nondeterminism of Open3D's multithreaded RANSAC. Near the RANSAC threshold (1–2 %
  inliers) that noise is larger: Sonata-PCA ‖ DINO gives 0.144 vs Sonata-PCA 0.119. So differences
  under ~3 points between single runs are not interpretable without the per-liver statistics. Images
  can only help if a model learns to map them into the geometry space.
- **Primary endpoint amended** (protocol deviation 4). FPFH is at ceiling noise-free, so the primary
  setting is now σ = 1 mm with 10 % holes, where FPFH r20 scores 0.631.

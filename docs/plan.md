# Self-supervised organ point features for CT surface matching (liver)

## Context
New project, complementing Project 3 (`../defreg`, which registers non-rigidly *after an oracle rigid stage*; a
realistic rigid initialiser is out of its scope). Question: do self-supervised per-point descriptors that combine
surface geometry with frozen DINOv2 features lifted onto visible points improve matching of a **partial visible
liver surface** to its **complete CT surface**, when the CT side has no appearance? The descriptor is the only
variable. Matching (mutual NN → RANSAC → ICP) is frozen before any training. Deliverables (6 months): a
descriptor extractor, a missing-modality benchmark, and a CT-surface matching study. A negative result (images
do not help, or smooth patches stay ambiguous) is a valid outcome.

**Decisions (with user):**
- A new standalone repo, `/leonardo_work/IscrC_FLAC/organfeat` (package `ofeat`). Code is copied from
  `correspondences`/`defreg` with origin notes; there is no import or path dependency.
- P2I-LReg comes from Kaggle (`kaggle datasets download junezhou001/p2i-lreg`).
- DePoLL is this project's **locked** real paired evaluation (marker TRE), run once at the end.

**Prior evidence the design must respect:**
- defreg results 014: per-cloud InfoNCE (Sonata, partial deformed target ↔ complete source) did not learn.
  - Top-1 was about 3 % at a 2 mm positive radius and 7 % at 4 mm.
  - Cause: smooth, self-similar surfaces, plus ~6 mm deformation larger than the positive radius.
  - Lepard and LiverMatch also fail on these livers.
- Here the core scope is rigid, so ground-truth correspondences are exact. RANSAC needs only a few % inliers, not dense
  flow. A safe radius removes near-duplicate negatives.
- A Stage-1 headroom gate tests learnability early.
- Project 10's DePoLL field was at chance (`correspondences/docs/stage5_depoll.md`).

**P2I-LReg facts (Self-P2IR, arXiv 2504.15152):**
- 21 HCC patients, 346 keyframes, manual liver masks, per-video intrinsics, preop CT liver model.
- Intraoperative point clouds come from Depth-Anything relative depth, scaled by matching PCA extents to the preop model.
- No landmarks and no GT poses. The paper's in-vivo metrics are Dice and Chamfer.
- 6 of the 21 patients are reprocessed P2ILF cases, so an overlap audit is needed.

## Repo setup (week 1)
- uv project, Python 3.12. Dependencies are the same as `correspondences/pyproject.toml` minus `pot` and
  `robust-laplacian`, plus `kaggle`. Training uses a plain PyTorch loop (no Lightning).
- Scratch: `/leonardo_scratch/large/userexternal/mnunzian/organfeat/{data,sif}`, with `data/*` symlinks. Env
  prefix `OFEAT_` (`OFEAT_DATA`, `OFEAT_UNLOCK_EVAL`).
- **Copy, with origin headers, adapted to `ofeat` types:**
  - `corr/mesh.py`
  - `corr/_vendor/{geometry.py (kabsch, hpr, camera_mask, fpfh, knn_area), organ_sim.py, sonata_encoder.py, sonata/}`
  - `corr/synth/{render.py, texture.py}`
  - `corr/features/{dino.py, patches.py, geom.py}`
  - `corr/data/{splits.py, liver_meshes.py, depoll.py, stereomis.py, liver_probe.py}`
  - from `scripts/stage5/depoll_eval.py`: `coloured_mesh`, `render_colours`, `views`, `tps_recon_to_ref` and
    `geodesic_from`, into `ofeat/data/depoll_views.py`
  - `scripts/preprocess/extract_stereomis.py` (`sgbm_depth`)
  - `scripts/leonardo/*` (repointed)
  - tests `test_synth.py`, `test_vendor.py`, `test_splits.py`, `test_depoll.py`
- **From defreg:**
  - `pretrain/p3.py::p3_loss`, the starting point for the loss
  - the open3d RANSAC call pattern in `methods/livermatch.py::_ransac`
- **Data copied, not linked:**
  - the 531 liver PLYs + `index.csv` + `configs/splits/totalseg_liver.yaml` (477 ok, 411/20/46)
  - Sonata weights
  - the DINOv2 `dinov2-with-registers-large` HF snapshot
  - `core.sif` + wheels
  - DePoLL without videos or DICOM (`CT/`, `markers/`, `surfaces/`, `coordinate_transforms/`, per-state
    `video_reconstruction.ply`, `M.yml`); check sizes first
  - StereoMIS P2_0–P2_5 and P3: calibration, `groundtruth.txt` (kinematics), masks, processed `left/`, `depth/` and
    `meta.json`
- **P2I-LReg download:** `download.sbatch` on `lrd_all_serial`. **User action:** place a Kaggle API token
  (`~/.kaggle/kaggle.json`) first; I won't read it.
- **Locked guard:** `LOCKED = ("depoll", "p2i_lreg")`, unlocked only with `OFEAT_UNLOCK_EVAL=<name>`.

## Pre-registration (`docs/protocol.md`, committed before any descriptor is trained)
- Fixed matching parameters (below), metrics, thresholds, the selection rule, the locked sets, and the
  week-2 P2I-LReg go/no-go.
- **Primary thresholds:**
  - match inlier: τ = 10 mm (also report 5 and 20)
  - registration success: RMSE of GT correspondences after ICP < 10 mm (also report RRE < 10° and RTE < 10 mm)
- **Model selection:** only the synthetic val livers (20), on registration recall, with inlier ratio as tie-break.

## Synthetic data (`ofeat/synth/views.py`)
- Split first: render caches are written per split directory, and loaders assert split membership.
- **Views per liver:** 48 (train), 16 (val/test).
  - Laparoscopic cameras: direction within a 60° cone of anterior, random roll, 70–160 mm, fov 70°, 518² pixels
    (37² DINO patches).
- **Stored per view:** `rgb`, `depth`, `vid`, `tri`+`bary` (exact surface point), `tool`, specular mask, `K`, `E`.
  - `render()` must be changed to return `p_rest` and barycentrics.
- **Textures:** a new random procedural generator (fBm colour field plus blob/vessel-like patterns, liver palette).
  - It is redrawn per sample, with seeds independent of vertex IDs.
  - Two texture families are held out for the stress test.
  - The CT side never gets colour.
- **Nuisances:** the renderer's lighting, specular, smoke, vignette, noise and tools. Also stereo-like depth noise
  (σ ∝ z², holes) as augmentation.
- **Overlapping view pairs** (overlap ≥ 30 % by `vid`) for partial↔partial pairs.
- **Stress-only sets on test livers:**
  - mild deformation: `organ_sim.simulate` with 2–8 mm displacements
  - occlusion levels
  - visible-fraction bins
  - held-out textures
  - depth-noise levels

## Inputs and descriptor
- **Partial cloud** (`ofeat/data/points.py`):
  - Pixels with `vid ≥ 0`, not tool, not specular, back-projected. Voxel-downsampled at 2 mm, then N = 4096.
  - Per point:
    - xyz
    - normal (kNN PCA, oriented to the camera)
    - curvature (vectorised torch kNN quadric at 2 radii; replaces the slow per-vertex loop in `geom.py`)
    - lifted DINO feature: the bilinear sample of the ViT-L/14-reg last-layer token grid at the point's pixel,
      PCA → 64 (fitted on train renders only)
- **CT cloud:** area-uniform surface samples (N = 8192), mesh normals, the same curvature, visual = *missing*.
- **GT:** partial point → CT surface point via `vid`/`bary`, then the nearest CT sample. Labels are used in the
  loss only.
- **Leakage control:** an independent random SO(3) + translation per cloud, per-cloud centring, resampling, and no
  RAS coordinates.
- **Model** (`ofeat/models/descriptor.py`):
  - `SonataEncoder` (pretrained init, fine-tuned; mm × 0.005, grid 0.02).
  - Visual adapter: an MLP(64 → stem width) added after the embedding, last layer zero-initialised, so the model starts
    as Sonata. A learned *missing* token is used for points without image features.
  - Head: MLP(1088 → 256 → 32), L2-normalised.
- **Training** (`ofeat/train/`):
  - Multi-positive InfoNCE (adapted `p3_loss`): τ = 0.07, positives within 4 mm, negatives exclude CT points within a
    12 mm safe radius, 1024 anchors.
  - Terms:
    - L(partial_vis ↔ CT)
    - \+ L(partial_geo ↔ CT)
    - \+ λ·L(partial_vis ↔ partial_geo), the cross-modal consistency term
    - \+ μ·L(view A ↔ view B), on overlapping partial pairs
  - Visual-modality dropout: p = 0.5 per sample, 0.1 per point.
  - Photometric augmentation (colour jitter, gamma, blur, specular) runs before an online frozen DINOv2 forward
    (bf16, no grad).

## Fixed matching pipeline (`ofeat/match.py`, `configs/match.yaml`; frozen at S0, same for every method)
1. Cosine mutual nearest neighbours on GPU (chunked). The ratio-test confidence is kept for coverage curves only.
2. Open3D `registration_ransac_based_on_correspondence`:
   - n = 3, inlier distance 7.5 mm
   - edge-length checker 0.9
   - 100k iterations, confidence 0.999, fixed seed
3. Open3D point-to-plane ICP, partial → CT: 5 mm, 50 iterations.

## Baselines, ablations, evaluation
- **Baselines:**
  - FPFH (radius chosen on val from 10/20/30 mm)
  - frozen Sonata (raw 1088-d and PCA-32)
  - simple concatenation: FPFH or Sonata-PCA ⊕ PCA-DINO, with zeros on the CT side
  - our geometry-only descriptor
- **Ablations:**
  - visual features off / train-only (geometry-only inference after multimodal training) / train + infer
  - dropout and the consistency term on/off
  - training data at 25/50/100 % of train livers
  - one Sonata-vs-scratch init run
- **Metrics** (`ofeat/eval/metrics.py`):
  - per-point match recall @τ
  - mutual-match inlier ratio and feature match recall (IR > 5 %)
  - inlier ratio and error vs retained coverage (sorted by confidence), with AUC
  - RRE, RTE and GT-RMSE after RANSAC and after ICP, and registration recall
  - wall-clock per stage and peak GPU/CPU memory on an A100
  - statistics per liver (views are not independent): bootstrap CIs and Wilcoxon tests, paired
- **Synthetic test:** 46 unseen livers. The stress tests above are reported as separate tables. For deformation,
  RMSE is compared with oracle rigid.
- **StereoMIS (no CT): RGB/geometry consistency.**
  - Frame pairs (Δt = 5–50) with SGBM depth. The liver comes from the copied DINOv2 probe; tool masks come from the
    dataset.
  - Pseudo-GT is the relative pose from kinematics. Breathing and kinematic error are stated as limits.
  - Metrics: match recall and pair RRE/RTE, geometry-only vs geometry + image, on P3 (held-out animal).
  - Optional training-data ablation: add P2 pseudo-correspondence pairs.
- **DePoLL (locked, run once):**
  - Partial: the per-view rendered coloured reconstruction (24 views/state, liver kept by the probe mask).
  - CT: the reference surface.
  - Metrics: RRE/RTE vs `M.yml`; marker TRE vs the oracle-rigid TRE (best rigid fit to the markers); success
    = TRE ≤ oracle + 10 mm.
  - The human→porcine gap is reported.
- **P2I-LReg:**
  - Weeks 1–2: an audit of RGB, masks, intrinsics, cloud↔pixel mapping, scale and the P2ILF overlap.
  - After the audit, before any descriptor runs: 4 patients are dev (format and scale debugging) and 17 are locked.
  - Clouds are used as given.
  - Metrics: Dice and contour distance of the projected CT model vs the liver mask; Chamfer is reported as
    consistency only. Results are shown with and without the P2ILF cases.
  - **If it is unusable by 2026-11-09,** it is dropped: no clinical CT-registration claims; conclusions come from
    synthetic + StereoMIS + DePoLL.

## Stages (Nov 2026 → Apr 2027)
- **S0 (wks 1–2):**
  - repo, copied code and data, Kaggle download and audit
  - `protocol.md`
  - matching pipeline and metrics with tests
  - synthetic caches
  - a RANSAC bound: success vs injected GT inlier ratio, i.e. how good descriptors must be
- **S1 (Dec):**
  - baselines on val/test
  - the geometry-only learned descriptor
  - **Gate:** it must beat FPFH on val registration recall. If not, diagnose receptive field and safe radius before
    adding images, and report it.
- **S2 (Jan):** the multimodal descriptor; selection on val.
- **S3 (Feb):** ablations, stress tests, StereoMIS consistency.
- **S4 (Mar):** locked DePoLL and P2I-LReg runs; runtime and memory.
- **S5 (Apr):**
  - write-up
  - release the extractor and benchmark scripts
  - no DePoLL-derived data (CC BY-NC-ND); P2I-LReg under its Kaggle licence

## Verification
- **Unit tests:**
  - the matcher recovers a known rigid transform from correspondences with 90 % outliers
  - metrics are exact on toy cases (identity descriptors give recall 1)
  - lifting: a point's feature equals the bilinear token sample at its pixel
  - a rendered GT surface point projects to its pixel (copied `test_synth`)
  - the missing-modality path equals the CT path
  - FPFH is invariant to the random rigid transform
  - the split guard raises on a test liver in a train cache and on locked sets
- **Smoke:** `gpu_test`/`gpu_smoke` sbatch: 200 training steps on 10 train livers (the loss decreases), then
  the benchmark end to end on 5 val livers.
- **Visual:** rerun logs of matches, inliers and registered clouds per method.

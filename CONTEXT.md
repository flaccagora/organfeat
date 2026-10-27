# ofeat: self-supervised liver point features for partial → CT surface matching

Can self-supervised per-point descriptors that combine surface geometry with frozen DINOv2 features,
lifted onto visible points, improve matching of a **partial visible liver surface** to its **complete
CT surface** when appearance is unavailable on the CT side? The descriptor is the only variable: the
matching pipeline (mutual NN → RANSAC → point-to-plane ICP, `configs/match.yaml`) is frozen before any
training. A negative result is valid. This complements Project 3 (`../defreg`, non-rigid after an
oracle rigid stage) by studying the correspondence features a realistic rigid stage would need.

Plan: docs/plan.md. Pre-registered protocol (metrics, thresholds, selection, locked sets): docs/protocol.md.

**Status (2026-10-26): S0–S4 done.** Results are in docs/summary.md, with per-stage write-ups in
docs/s*_*.md. Post-hoc external benchmarks (2026-10-27): docs/ext_ircad.md, docs/ext_p2i_syn.md.
- Both locked sets (DePoLL, the 17 locked P2I-LReg cases) have been evaluated once. Never rerun them
  to compare new methods.
- Final descriptors: `scripts/bench/final_descs.sh`. Arm A is fine-tuned Sonata; arm B is a local
  PPF-PointNet.
- Training recipe: fp32, head lr 1e-4. bf16 with head lr 1e-3 collapses.

## Data (data/* -> /leonardo_scratch/large/userexternal/$USER/organfeat/data)
| what | where | role |
|---|---|---|
| TotalSeg liver meshes (human CT) | data/processed/totalseg_meshes | synthetic training/val/test, official split (408/20/46; meshes < 300 ml excluded) |
| StereoMIS (porcine, stereo, kinematics) | data/raw/stereomis, data/processed/stereomis | real RGB/geometry consistency (no CT); P2 train, P3 test |
| DePoLL (porcine, CT + reconstruction + 60 markers) | data/raw/depoll | **locked** paired evaluation (marker TRE), run once |
| P2I-LReg (human, Kaggle `junezhou001/p2i-lreg`) | data/raw/p2i_lreg | audit; dev subset + **locked** rest (decided at audit); its synthetic test renders: post-hoc external benchmark (docs/ext_p2i_syn.md) |
| Deformed IRCAD (Neri et al. 2025 generator + 20 3D-IRCADb-01 livers, CC BY-NC-ND) | data/raw/deformed_ircad, data/processed/deformed_ircad | post-hoc external benchmark, test sets regenerated (docs/ext_ircad.md) |

Locked sets raise `LockedDatasetError` unless `OFEAT_UNLOCK_EVAL=<name>` (final evaluation only).
All data was copied from the correspondences project's scratch (scripts/download/copy_local.sh), not linked.

## Code
- `ofeat/_vendor/`: code copied from correspondences (itself partly from defreg): geometry helpers
  (kabsch, HPR, FPFH), organ_sim (ARAP), Sonata encoder + vendored sonata, IO. No imports of either repo.
- `ofeat/synth/`: renderer (nvdiffrast GPU / open3d raycast CPU, exact tri/bary ground truth),
  random procedural textures (redrawn per sample; two held-out families).
- `ofeat/features/`: DINOv2 patch tokens, point-cloud normals/curvature (torch).
- `ofeat/match.py`: the fixed pipeline. `ofeat/eval/metrics.py`: protocol metrics.

## Running on Leonardo
`scripts/leonardo/run.sh <cmd>` runs inside data/sif/core.sif (open3d needs a newer glibc than the
host). `scripts/leonardo/run.sh uv sync --frozen` builds .venv; `scripts/leonardo/run.sh python -m pytest`.
CPU jobs: `scripts/leonardo/cpu.sbatch` (DCGP, account IscrC_FLAC_0). GPU: `gpu_run.sbatch`, `gpu_test.sbatch`.
Downloads: `sbatch -J dl_<name> scripts/leonardo/download.sbatch <name>` (serial partition, internet).
Login nodes: 1 GB memory cap, 10 min CPU.

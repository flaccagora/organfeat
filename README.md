# organfeat (`ofeat`)

**Self-supervised liver point descriptors (surface geometry + lifted DINOv2) for matching a partial
laparoscopic view to the complete CT surface.**

When the CT side has no appearance, do per-point descriptors that combine surface geometry with
frozen DINOv2 image features, lifted onto the visible points, make rigid matching of a **partial
visible liver surface** to its **complete CT surface** better? The descriptor is the only variable.
Point sampling and the matching pipeline (mutual nearest neighbours → RANSAC → point-to-plane ICP)
were frozen before any training. A negative result counts as a valid outcome.

- **Rules of the study:** [`docs/protocol.md`](docs/protocol.md), pre-registered, with a dated
  **Deviations** log.
- **How the code is organised:** [`ARCHITECTURE.md`](ARCHITECTURE.md).
- **Short orientation:** [`CONTEXT.md`](CONTEXT.md).

---

## Status (2026-10-26): S0–S4 done

Study summary: [`docs/summary.md`](docs/summary.md). Venue assessment and plan to submission: [`docs/venue_assessment.md`](docs/venue_assessment.md), [`docs/submission_plan.md`](docs/submission_plan.md).

| Stage | State | Headline |
|---|---|---|
| **S0** repo, protocol, frozen pipeline, synthetic caches | done | FPFH r20 is the selected baseline. The frozen RANSAC needs ~3–5 % correct matches ([s0](docs/s0_findings.md)). |
| **S1** geometry-only learned descriptor + gate | done, gate passed | On val (primary setting): arm A (Sonata) 1.000 recall, arm B (PPF) 0.994; FPFH 0.631 ([s1](docs/s1_findings.md)). |
| **S2** multimodal descriptor | done | Recall is at ceiling. Images raise the inlier ratio: marginally for arm A, clearly for arm B ([s2](docs/s2_findings.md)). |
| **S3** test, stress tests, ablations, StereoMIS | done | Test: learned 0.99–1.00 vs FPFH 0.67. Images help arm A by +3 % and arm B by +20 %, not via texture. Arm A does not transfer to real stereo; arm B does ([test](docs/s3_test.md), [stress](docs/s3_stress.md), [ablations](docs/s3_ablations.md), [StereoMIS](docs/s3_real_consistency.md)). |
| **S4** locked DePoLL + P2I-LReg (run once) | done | DePoLL: no method registers, FPFH included. P2I-LReg: no descriptor beats a random-descriptor null ([DePoLL](docs/s4_depoll.md), [P2I-LReg](docs/s4_p2i.md), [log](docs/s4_locked.md)). |
| **External** (post hoc) deformed IRCAD, P2I-LReg synthetic | done | IRCAD: at the deformation floor at 15–25 % visibility, beating published methods. At 5 %, a post-hoc pose-prior variant (`ofeat/match_ext.py`) halves TRE to the published range. P2I synthetic: recall saturated, 29 % even for random descriptors ([IRCAD](docs/ext_ircad.md), [P2I syn](docs/ext_p2i_syn.md)). |
| **S5** write-up, release | — | |

Both locked sets have been used. Never rerun them to compare new methods.

---

## What is in the box

| Component | Where | What it does |
|---|---|---|
| Fixed matcher | `ofeat/match.py`, `configs/match.yaml` | Cosine mutual NN → Open3D correspondence RANSAC → point-to-plane ICP. Identical for every method. |
| Point inputs | `ofeat/data/points.py`, `configs/points.yaml` | One point per 3 mm voxel on both sides, k-NN PCA normals, stereo-like depth noise, random rigid input frames. |
| Synthetic benchmark | `ofeat/synth/`, `ofeat/eval/`, `scripts/bench/` | Rendered views of TotalSeg livers with exact ground truth: val / test plus 7 stress sets. |
| Descriptors | `ofeat/models/` | Arm A: fine-tuned Sonata/PTv3. Arm B: local PPF-PointNet. Both take a visual channel or a learned *missing* token. |
| Extractor (deliverable) | `ofeat/models/infer.py` | Checkpoint → 32-d per-point descriptors. |
| Real-data evaluation | `scripts/eval/` | StereoMIS frame-pair consistency; DePoLL marker TRE (locked); P2I-LReg mask Dice (dev / locked). |
| External benchmarks | `scripts/eval/{ircad,p2i_syn}_eval.py`, `scripts/external/`, `scripts/bench/ext_report.py` | Deformed IRCAD (Neri et al. 2025; authors' generator) and P2I-LReg synthetic (Self-P2IR), with the papers' metrics. |

---

## Setup (CINECA Leonardo)

The code runs inside a Singularity image, `core.sif`, because the host glibc is too old for
Open3D 0.20. The uv environment (`.venv`) lives on the shared filesystem and is built from `uv.lock`
inside that image.

```bash
# 1. Environment variables (scratch paths, caches, HF_HOME). Every launcher sources this itself.
source scripts/leonardo/env.sh

# 2. Prerequisites (one-off; already done for this checkout):
#    - data/* symlinks -> $OFEAT_SCRATCH/data   (raw/, processed/, weights/, sif/, cache/, tmp/)
#    - data/sif/core.sif                         (scripts/leonardo/build_core.sh)
#    - wheels/{flash_attn,nvdiffrast}-*.whl      (prebuilt; never compiled on cluster nodes)
#    - DINOv2 snapshot in $HF_HOME, Sonata weights in data/weights/sonata (compute nodes are offline)

# 3. Build / refresh .venv (login node)
scripts/leonardo/run.sh uv sync --frozen

# 4. Tests
scripts/leonardo/run.sh python -m pytest          # CPU tests; GPU and dataset tests skip
sbatch scripts/leonardo/gpu_test.sbatch           # same suite on a booster GPU (debug QoS)
sbatch scripts/leonardo/gpu_smoke.sbatch          # torch/CUDA, Open3D, nvdiffrast, Sonata and DINOv2 forwards
```

`scripts/leonardo/run.sh <cmd>` runs any command in the environment and changes to the repo root
first. **Always launch through it** (or one of the `sbatch` wrappers below), because output paths
are relative to the repo root.

| Launcher | Partition | Use |
|---|---|---|
| `scripts/leonardo/gpu_run.sbatch <cmd>` | `boost_usr_prod`, 1 GPU | training, benchmarking, evaluation |
| `scripts/leonardo/cpu.sbatch <cmd>` | `dcgp_usr_prod` (account `IscrC_FLAC_0`) | rendering view caches, CT caches, sanity checks |
| `scripts/leonardo/download.sbatch <name>` | `lrd_all_serial` (internet) | `scripts/download/<name>.sh` |

Login nodes cap a process at 1 GB of memory and 10 minutes of CPU, so anything heavier goes to
`sbatch`. Job logs are written to `jobs/out/<job-name>_<job-id>.out`.

---

## Workflow: from meshes to numbers

Commands run from the repo root. Treat the resource flags (`-c`, `--mem`, `-t`) as examples.

### 1. Data
```bash
sbatch -J dl_copy_local scripts/leonardo/download.sbatch copy_local   # DePoLL subset + StereoMIS extracts
sbatch -J dl_p2i_lreg  scripts/leonardo/download.sbatch p2i_lreg      # Kaggle; needs your own ~/.kaggle token
```
TotalSeg liver meshes, the liver probe, the Sonata weights and `core.sif` were copied once from the
`correspondences` project's scratch ([s0 findings](docs/s0_findings.md)).

### 2. Synthetic caches (after the splits are frozen)
```bash
sbatch -J views_train -c 32 --mem 96G -t 02:00:00 scripts/leonardo/cpu.sbatch \
    python scripts/synth/make_views.py --set train --workers 32
sbatch -J views_val   -c 32 --mem 96G -t 02:00:00 scripts/leonardo/cpu.sbatch \
    python scripts/synth/make_views.py --set val --workers 20
sbatch -J views_test_sets -c 48 --mem 128G -t 02:00:00 scripts/leonardo/cpu.sbatch \
    bash scripts/synth/make_test_sets.sh                                  # test + 7 stress sets
sbatch -J cache_ct -c 32 --mem 64G -t 01:00:00 scripts/leonardo/cpu.sbatch \
    python scripts/train/cache_ct.py --split train --workers 32
sbatch -J fit_pca -t 01:00:00 scripts/leonardo/gpu_run.sbatch python scripts/bench/fit_pca.py   # train split only
```
`make_views.py --set val --preview 4` writes a contact sheet to `outputs/synth/` without caching.

### 3. Baselines and the RANSAC bound
```bash
sbatch -J bench_val -t 02:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/bench/run_bench.py --set val --sigma100 1 --holes 0.1 --desc fpfh_r20 sonata fpfh_r20+dino
sbatch -J ransac_bound -c 16 -t 01:00:00 scripts/leonardo/cpu.sbatch python scripts/bench/ransac_bound.py
```

### 4. Training
```bash
# One run: any key of configs/train/base.yaml can be overridden with key=value
sbatch -J tr_mine -t 12:00:00 scripts/leonardo/gpu_run.sbatch \
    python -m ofeat.train.run --config base name=mine loss.use_vis=true loss.lambda_cross=0

bash scripts/train/s1_geo.sh         # S1: geometry-only, encoder-lr grid
bash scripts/train/s2_mm.sh          # S2: multimodal, λ ∈ {0, 1}
bash scripts/train/s3_ablations.sh   # S3: no dropout, 25 / 50 % livers, scratch init
```
Each run writes `outputs/train/<name>/{config.yaml, log.jsonl, step_<k>.pt}`. The log includes a
val proxy every `eval.every` steps. Use `model.arch=ppf optim.lr=1e-3` for arm B.

### 5. Selection (val only) and comparison
```bash
bash scripts/bench/select.sh <run> [geo]                     # all checkpoints of a run, primary setting
GEO=1 bash scripts/bench/score_ckpts.sh outputs/train/<run>/step_008000.pt   # one checkpoint, img and :geo
python scripts/bench/compare.py --set val_noise1_holes0.1 --ref fpfh_r20      # per-liver means, bootstrap CI, Wilcoxon
python scripts/bench/compare.py --set val_noise1_holes0.1 --by vis_bin        # broken down by visible fraction
```
The selection rule is registration recall first, with inlier ratio@10 as the tie-break, on val in
the primary setting. The test set is scored **once** per final configuration.

### 6. Stress tests and real data
```bash
bash scripts/bench/s3_stress.sh <desc> [<desc> ...]           # held-out textures, tools, deformation, noise levels

sbatch -J smis_cons -t 03:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/eval/stereomis_consistency.py --split test --desc fpfh_r20 learned:<ckpt> learned:<ckpt>:geo

# P2I-LReg dev cases (format / scale debugging only, never selection)
sbatch -J p2i_dev -t 02:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/eval/p2i_eval.py --split dev --desc fpfh_r20

# LOCKED, once, at S4 only:
#   sbatch ... --export=ALL,OFEAT_UNLOCK_EVAL=depoll   scripts/leonardo/gpu_run.sbatch python scripts/eval/depoll_eval.py ...
#   sbatch ... --export=ALL,OFEAT_UNLOCK_EVAL=p2i_lreg scripts/leonardo/gpu_run.sbatch python scripts/eval/p2i_eval.py --split locked ...
```

External benchmarks (post hoc; docs/ext_ircad.md, docs/ext_p2i_syn.md):
```bash
for r in 05 10 15 25; do  # deformed IRCAD test sets, authors' generator (seconds each)
  sbatch -J gen_ircad scripts/leonardo/cpu.sbatch python scripts/external/gen_deformed_ircad.py \
      --crop-ratio 0.$r --out data/processed/deformed_ircad/test_crop$r.h5
done
sbatch -J ext_ircad -t 02:00:00 scripts/leonardo/gpu_run.sbatch python scripts/eval/ircad_eval.py --run final --desc fpfh_r20 ...
sbatch -J ext_p2isyn -t 03:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=p2i_lreg scripts/leonardo/gpu_run.sbatch \
    python scripts/eval/p2i_syn_eval.py --cases 01 02 03 04 05 06 07 --run final_g1 --desc fpfh_r20 ...
python scripts/bench/ext_report.py --compact --ircad outputs/eval/ircad/final.parquet
python scripts/bench/ext_report.py --p2i-syn outputs/eval/p2i_syn/final_g*.parquet
```

---

## Descriptor names (`--desc`)

Every driver accepts the same names (`ofeat/eval/descriptors.py`):

| Name | Descriptor |
|---|---|
| `fpfh_r<R>` | FPFH at radius R mm (Open3D, outward normals); the selected radius is 20 |
| `sonata` / `sonata_pca32` | frozen Sonata 1088-d features / their PCA-32 (fitted on train CT clouds) |
| `<geom>+dino` | `[geometry ‖ DINO-PCA-64]`, each block L2-normalised; the CT block is zero (a no-op under cosine NN, by design) |
| `learned:<ckpt>` | a trained checkpoint; the partial side uses its image if the model was trained with images |
| `learned:<ckpt>:geo` | the same checkpoint with geometry-only inference (the CT path on both sides) |

## Using the extractor

```python
from ofeat.models.infer import Extractor

ex = Extractor("outputs/train/s2_ppf_lam0/step_008000.pt")      # model rebuilt from the checkpoint's config
f_ct = ex(ct_xyz_mm, ct_normals)                                  # CT / geometry-only path
f_view = ex(view_xyz_mm, view_normals, rgb=rgb_u8, pix=uv)        # partial view with its image
```
Clouds should be sampled as in `configs/points.yaml` (one point per 3 mm voxel) with **outward**
normals (towards the camera on a partial view). `pix` holds each point's `(u, v)` pixel in `rgb`.
Descriptors are 32-d and L2-normalised; match them with `ofeat.match.register`.

---

## Ground rules

1. **Do not tune the frozen parts.** `configs/match.yaml`, `configs/points.yaml`, `ofeat/match.py`,
   the splits and the metrics are fixed by the protocol. A protocol change is appended under
   *Deviations* in `docs/protocol.md`, with a date and a reason; nothing above that section is
   edited silently.
2. **Select on val only.** Score the test set once per final configuration, and stress sets once,
   after the main test.
3. **Locked sets stay locked.** DePoLL and the locked P2I-LReg cases raise `LockedDatasetError`
   unless `OFEAT_UNLOCK_EVAL=<name>` is set. Set it only in the final S4 jobs: never while
   training, fitting PCA or selecting.
4. **Keep train-only fits train-only.** DINO and Sonata PCAs are fitted on the train split.
5. **Licences.**
   - DePoLL is CC BY-NC-ND 4.0: never redistribute data derived from it.
   - P2I-LReg is CC BY-NC 4.0. Refer to its cases by folder number only.
   - StereoMIS is licensed for non-commercial use.
6. **Scratch is purged** after 40 days without access. Keep anything irreplaceable elsewhere.

## Documentation

| File | Contents |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | subsystems, data flow, storage, design decisions |
| [`CONTEXT.md`](CONTEXT.md) | one-page orientation |
| [`docs/plan.md`](docs/plan.md) | the 6-month plan and the decisions behind it |
| [`docs/protocol.md`](docs/protocol.md) | pre-registered protocol and deviations |
| [`docs/s0_findings.md`](docs/s0_findings.md), [`s1`](docs/s1_findings.md), [`s2`](docs/s2_findings.md) | stage results |
| [`docs/p2i_audit.md`](docs/p2i_audit.md) | P2I-LReg format and camera audit |

## Provenance

Parts of the code are copied, with origin notes in each file header, from two sibling projects:
`../correspondences` (Project 10) and `../defreg` (Project 3, non-rigid registration after an oracle
rigid stage). Sonata/PTv3 is vendored under `ofeat/_vendor/sonata` with its licence. Nothing
imports either sibling repository.

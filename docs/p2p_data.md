# P2P data (phase-2 external benchmarks E3 / E4; set up 2026-10-28)

- **Source:** Yang et al., IEEE JBHI 2025 (doi:10.1109/JBHI.2025.3583875),
  github.com/zixinyang9109/P2P at commit 7f42cc7.
- **Download:** the authors' Google Drive, via `scripts/download/p2p.sh` (serial partition): in
  silico + in vitro 14 GB, baselines 4 GB. Extracted under `data/raw/p2p/{data,baselines}/*_x/`.
- **Licence:** the LiverMatch README says the P2P dataset is released under CC BY-SA 4.0. The P2P
  repo has no licence file. No derived data is released by us.
- **Loader:** `ofeat/data/p2p.py`. Both test sets are locked (`OFEAT_UNLOCK_EVAL=p2p_test`),
  test-only, scored once at W10 (docs/protocol.md, phase 2).

| set | content | ground truth | protocol |
|---|---|---|---|
| in silico test (E3) | 6,315 pairs. MSD liver sources (~4,500 surface points, mm); FEM (neo-Hookean) deformed, cropped targets (visibility 0.2–1.0); full SO(3) target rotation (`rot_tgt`) | 5,742 volumetric fiducials per pair: RMS-TRE over all of them | noise-free (σ none); bins [0.2, 0.3) … [0.9, 1] on `stat_svd.npz` vis |
| in vitro (E4) | 800 pairs from 4 phantoms (1, 2, 4, 5), CT of undeformed / deformed phantoms, auto-cropped partial targets | 53 embedded fiducials: RMS-TRE | bins on `in_vitro/stat.npz` vis |
| in silico train (`Dataset/Deform_mesh_npz/<id>`) | 1,313 liver variants × 11 meshes (rest + 10 deformations), volumetric points | — | validation subsets only, never training |

**Checks (2026-10-28):**
- **Published tables reproduce exactly.** The authors' per-sample results
  (`repo/Eva_in_silico/Results_in_silico`, `repo/Eva_in_vitro/Results_in_vitro`) give the paper's
  rows to 0.01 mm: in silico LiverMatch, LiverMatch + P2P, Lepard, Lepard + P2P; in vitro LiverMatch
  and Lepard. So our numbers can be compared **per sample**, paired.
- **Ground truth composition.** The rigid ground truth source → rotated target is
  `rot_tgt @ R_gt`, `rot_tgt @ t_gt`. Its fiducial RMS equals `stat_svd.npz` `deform` exactly; the
  authors' script composes `R_gt @ rot_tgt`, used only for their inlier-ratio diagnostics.
- **Floors.** `deform` is the benchmark's deformation floor (rigid part vs FEM deformation). The
  best rigid fit to the fiducials (oracle) is lower: e.g. 4.15 vs 4.74 mm on sample 0.
- **Units:** mm. Livers ~170 × 100 × 140 mm.

# R3a / R5a: DePoLL with whole reconstructions (post hoc, 2026-10-28)

Pre-registered in docs/protocol.md (2026-10-28). Run: `scripts/eval/depoll_whole_eval.py --run r3a`
(`OFEAT_UNLOCK_EVAL=depoll`; per-state outputs in `outputs/eval/depoll_whole/r3a/`). The first run
segfaulted in state 13 (exit 11, as S4's first DePoLL run). The script was made resumable per state
and rerun unchanged.

## Setup
- **Partial, variant `full`:** each state's `video_reconstruction.ply` at 3 mm voxels, median 1,294
  points. This is the protocol of Neri et al. and Huang et al.
- **Partial, variant `liver`:** the union of probe-liver points over the 24 S4 views, median 507
  points. Seven states give fewer than 256 points and are skipped.
- **CT:** the reference surface.
- **Poses:** 20 per state and setting, `bounded` (≤ 45° per Euler axis, ≤ 50 mm) and `so3`.
- **Pipelines:** the frozen one, and `match_ext` (`bounded` only).
- **Oracle TRE** (best rigid fit of the 60 markers, the deformation floor): median 15.6 mm over
  states, range 12.3–47.8 mm.

## Results
Median TRE (mm) over states × poses. Success (TRE ≤ oracle + 10 mm) is ≤ 0.08 for every method in
every cell.

| variant | init | pipeline | FPFH | A_geo | A_mm:geo | A_mm (img) | B_geo | B_mm:geo | B_mm (img) | random |
|---|---|---|---|---|---|---|---|---|---|---|
| full | bounded | frozen | 101.3 | **87.6** | 91.9 | 91.7 | 117.2 | 110.6 | 113.6 | 115.3 |
| full | bounded | match_ext | 79.5 | **65.3** | 67.6 | 68.7 | 75.1 | 87.1 | 77.0 | 85.7 |
| full | so3 | frozen | 108.2 | 102.1 | **99.5** | 101.5 | 112.1 | 110.6 | 112.1 | 111.2 |
| liver | bounded | frozen | 107.4 | 109.4 | 96.0 | 105.3 | 115.5 | **91.2** | 116.8 | 112.8 |
| liver | bounded | match_ext | 116.9 | **80.9** | 82.1 | 83.6 | 91.2 | 112.0 | 82.5 | 93.9 |
| liver | so3 | frozen | **90.7** | 99.5 | 101.3 | 103.7 | 114.1 | 91.2 | 116.0 | 108.8 |

**Gate G1** (pre-registered): the best learned descriptor's median TRE ≤ 15 mm **and** better than
the random null.
- **Not met.** The best median is 65.3 mm (A_geo, `full`, `bounded`, `match_ext`).
- The learned descriptors do beat random in every setting (one-sided paired Wilcoxon, p 1.6e-11 to
  0.04), but at TRE far above the 15.6 mm floor.
- **Decision by the pre-registered rule: framing B** (docs/submission_plan.md).

**R5a (images on vs off):** A_mm vs A_mm:geo and B_mm vs B_mm:geo are indistinguishable. With every
method failing, this cannot inform the image question.

## Reading
- **DePoLL stays unregistrable** for every descriptor, FPFH included, with whole reconstructions as
  well as with the S4 views. It is therefore not only a small-patch problem: the whole
  reconstruction gives ~1,300 points at 3 mm.
- **The remaining candidate causes:**
  - porcine vs human liver shape (the descriptors are trained on human TotalSeg livers; FPFH has no
    training but also fails)
  - stereo reconstruction quality
  - large deformation (oracle floor up to 48 mm)
- **The published numbers** (Huang et al.: 12.9 mm on clips; Neri et al.: best ~42 mm) use their own
  reconstructions or initialisations. Ours sit far from both.

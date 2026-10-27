# S4: locked DePoLL evaluation (run once, 2026-10-26)

`scripts/eval/depoll_eval.py --run final`, OFEAT_UNLOCK_EVAL=depoll. Outputs:
`outputs/eval/depoll/final.parquet` and `final_summary.json`.

## Protocol (docs/protocol.md section 8, as pre-registered)
- **Partial side.** Each state's coloured stereo reconstruction is Poisson-meshed and rendered from
  24 virtual views (vertex colours). Liver pixels come from the frozen DINOv2 probe; voxels are 3 mm,
  and a view needs ≥ 256 points.
- **CT side:** the reference surface.
- **Metrics.** Marker TRE (60 markers) after our rigid T, the oracle-rigid TRE (best rigid fit to
  the markers, i.e. the deformation floor), and success = TRE ≤ oracle + 10 mm.

## Deviations (data handling only; nothing changed in the method)
1. **Run 1 crashed** with a segfault (signal 11) before printing or saving anything. Run 2 was
   resubmitted unchanged, with the Python fault handler, and completed in 7.7 min. The crash did not
   recur, so its location is unknown (likely Open3D, as in Project 10 stage 5).
2. **The smoke test passed beforehand** (state 01, 2 views, no metrics).

## Coverage
- **Only 36 of 312 views were scored.** The other 276 had fewer than 256 liver points after the
  probe mask and 3 mm voxels.
- Scored views per state: 01 (12), 02 (17), 05 (1), 09 (3), 11 (2), 12 (1). States 03, 04, 06, 07,
  08, 10 and 13 have no usable view.
- The scored partials have ~400–530 points, roughly a 6 × 6 cm patch, against ~3,000 points (~20 %
  of the surface) on synthetic views.

## Results (mean over states, then descriptors)
| descriptor | success (TRE ≤ oracle + 10 mm) | marker TRE (mm) | oracle-rigid TRE (mm) | RRE vs oracle (°) | mutual matches |
|---|---|---|---|---|---|
| FPFH r20 | 0.00 | 103.3 | 22.4 | 108.7 | 111 |
| frozen Sonata | 0.00 | 128.6 | 22.4 | 130.2 | 6 |
| A_geo | 0.00 | 111.8 | 22.4 | 136.8 | 12 |
| A_mm (img) | 0.00 | 115.1 | 22.4 | 132.5 | 12 |
| A_mm :geo | 0.01 | 113.4 | 22.4 | 132.2 | 11 |
| B_geo | 0.00 | 118.6 | 22.4 | 133.1 | 140 |
| B_mm (img) | 0.00 | 96.3 | 22.4 | 108.9 | 129 |
| B_mm :geo | 0.00 | 108.8 | 22.4 | 138.3 | 140 |

## Validity check (`scripts/sanity/depoll_frame_check.py`, no descriptor involved)
After the marker oracle-rigid transform, the partial clouds lie a median of 5.3 mm (state 01) and
8.2 mm (state 02) from the reference CT surface (p90 ~20 mm). The partial, marker and CT frames are
consistent, so the failures are real registration failures, not an evaluation bug.

## Reading
- **No descriptor registers DePoLL**, including FPFH. Rotation errors of ~110–140° mean the poses
  are essentially random.
- **This is a locked negative result for the whole fixed pipeline on this data, not only for the
  learned descriptors.** Differences between descriptors here (e.g. B_mm's lower TRE) are not
  interpretable: success is ~0 everywhere and n is small.
- **Likely causes, compounding:**
  - *Tiny partial patches* (~6 × 6 cm, 400–530 points): the low-overlap regime. On synthetic data
    only < 10 % of views were that small, and arm A already lost recall there.
  - *Shape gap:* porcine, multi-lobed livers at test vs human TotalSeg livers in training.
  - *Deformation:* a ~22 mm floor after the best rigid fit.
  - *Reconstruction artefacts* in the Poisson-meshed stereo reconstruction (the same partial
    protocol as Project 10, which also found nothing on DePoLL).
- **Arm A's few mutual matches** (~12 per view vs ~130 for PPF and FPFH) reproduce the StereoMIS
  sim-to-real gap (docs/s3_real_consistency.md).
- **The protocol extracted few usable views,** leaving most states unscored. This is reported as a
  limitation: DePoLL as used here cannot discriminate between descriptors.

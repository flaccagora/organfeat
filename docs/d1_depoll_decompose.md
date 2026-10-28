# D1: what breaks DePoLL? (post hoc diagnostic, 2026-10-28)

Pre-registered in docs/protocol.md (2026-10-28). Run: `scripts/eval/depoll_decompose.py --run d1`
(`OFEAT_UNLOCK_EVAL=depoll`), output `outputs/eval/depoll_decompose/d1.parquet`.

All three conditions use **the same visible region** (the footprint of each state's video
reconstruction) and **the same target** (the porcine reference CT, as R3a):
- **C1, shape only:** the reference surface itself, independently resampled. No deformation.
- **C2, + real deformation:** that state's intraoperative CT surface. A clean CT-quality surface,
  real deformation.
- **C3, + reconstruction quality:** the video reconstruction (= R3a `full`).

13 states × 20 poses per setting. Median partial sizes are 1,378 / 903 / 1,294 points. The
deformation floor (oracle TRE) is 0 for C1 and 15.6 mm (median) for C2 and C3.

## Raw results
```
n_partial median by condition: {'C1_shape': 1378.0, 'C2_deform': 903.0, 'C3_recon': 1294.0}
oracle TRE median by condition: {'C1_shape': 0.0, 'C2_deform': 15.6, 'C3_recon': 15.6}

tre_mm (median over 13 states x 20 poses)
cond                        C1_shape  C2_deform  C3_recon
init    pipeline  d                                      
bounded frozen    A_geo         0.06      97.70     89.72
                  A_mm:geo      0.08      87.79     88.64
                  B_geo         0.06      95.75    115.17
                  B_mm:geo      0.06     111.26    110.60
                  fpfh_r20      0.06      94.81    101.32
                  random      112.94     109.94    113.41
        match_ext A_geo         0.06      62.48     64.75
                  A_mm:geo      0.06      62.42     68.57
                  B_geo         0.05      70.03     75.44
                  B_mm:geo      0.06      61.46     90.39
                  fpfh_r20      0.05      83.46     80.31
                  random       72.85      87.00     84.96
so3     frozen    A_geo         0.08     110.31    107.05
                  A_mm:geo      0.09     103.79     98.50
                  B_geo         0.06     107.72    110.22
                  B_mm:geo      0.06     108.39    103.82
                  fpfh_r20      0.06      80.34    105.71
                  random      102.27     112.56    113.45

success (mean over 13 states x 20 poses)
cond                        C1_shape  C2_deform  C3_recon
init    pipeline  d                                      
bounded frozen    A_geo         0.68       0.02      0.00
                  A_mm:geo      0.67       0.02      0.00
                  B_geo         1.00       0.02      0.00
                  B_mm:geo      1.00       0.01      0.00
                  fpfh_r20      1.00       0.02      0.01
                  random        0.12       0.00      0.00
        match_ext A_geo         0.79       0.03      0.00
                  A_mm:geo      0.80       0.05      0.01
                  B_geo         1.00       0.00      0.00
                  B_mm:geo      1.00       0.01      0.00
                  fpfh_r20      1.00       0.08      0.00
                  random        0.20       0.01      0.01
so3     frozen    A_geo         0.61       0.02      0.00
                  A_mm:geo      0.59       0.03      0.00
                  B_geo         1.00       0.02      0.00
                  B_mm:geo      1.00       0.03      0.00
                  fpfh_r20      1.00       0.00      0.02
                  random        0.12       0.00      0.00

C1, bounded, match_ext: success by state
d      A_geo  A_mm:geo  B_geo  B_mm:geo  fpfh_r20  random
state                                                    
01      1.00      1.00    1.0       1.0       1.0    0.30
02      1.00      1.00    1.0       1.0       1.0    0.50
03      0.95      0.95    1.0       1.0       1.0    0.00
04      1.00      1.00    1.0       1.0       1.0    0.15
05      1.00      1.00    1.0       1.0       1.0    0.45
06      0.05      0.15    1.0       1.0       1.0    0.00
07      0.05      0.05    1.0       1.0       1.0    0.05
08      1.00      1.00    1.0       1.0       1.0    0.45
09      1.00      1.00    1.0       1.0       1.0    0.40
10      0.75      0.75    1.0       1.0       1.0    0.00
11      1.00      1.00    1.0       1.0       1.0    0.25
12      1.00      1.00    1.0       1.0       1.0    0.00
13      0.45      0.55    1.0       1.0       1.0    0.05
```

## Reading
- **Species / shape is not the bottleneck.**
  - On the undeformed porcine surface, arm B and FPFH register **every** pose of every state (success
    1.00, median TRE 0.05 mm), in both pose settings.
  - Arm A is weaker: 0.59–0.80, failing mostly on states 06, 07 and 13. Its global context seems
    tied to the human training shapes, consistent with its weak real-video transfer (StereoMIS).
  - The random null stays at 0.12–0.20.
- **Real deformation is the bottleneck.**
  - Swapping in the intraoperative CT surface of the same region (C2) collapses every method to
    success ≤ 0.08 and median TRE 60–110 mm, against a 15.6 mm floor.
  - The surface is clean, so this is deformation, not reconstruction.
- **Reconstruction quality adds nothing measurable.** C3 (video reconstruction) is as bad as C2.
- **C2's partials are smaller** (903 vs 1,378 points, the crop within 5 mm of the reconstruction).
  Size alone does not explain the collapse: on IRCAD, ~900-point partials (25 %) register at
  0.97–1.00.
- **Implication.** Between the preoperative and intraoperative states, DePoLL's deformation (median
  rigid residual 15.6 mm, up to 48 mm; about 2.5× IRCAD's) changes the visible geometry enough that
  no rigid matching, learned or hand-crafted, finds the pose.
  - Our training deformation (2–8 mm ARAP stress) and IRCAD's (≤ 25 mm on one lobe, ~6 mm residual)
    are far milder.
  - Closing this gap needs deformation-aware matching or training with realistic large
    deformation, not better sim-to-real appearance or reconstruction.

# R2: training-seed replicates (2026-10-28)

Pre-registered in docs/protocol.md (2026-10-28).
- **Models:** the four final models retrained with seeds 1 and 2: original overrides, 10k steps,
  each taken at the step selected for seed 0 (no new selection).
- **Evaluation:** `scripts/bench/seed_replicates.sh` on the synthetic test (primary setting), the
  locked IRCAD v2 set and StereoMIS P2.
- **Training curves:** no NaN steps. Arm A's replicates train like seed 0. Arm B's seed 2 ends with
  a higher loss (training acc@10 0.40 vs 0.48–0.50), yet evaluates like the others.

## Synthetic test, StereoMIS P2 (raw output)
```
## Synthetic test (primary setting), per-liver means; per seed 0 / 1 / 2, then mean ± SD
A_geo     success 0.996 / 0.996 / 0.995 = 0.995 ± 0.001 | IR 0.349 / 0.355 / 0.349 = 0.351 ± 0.004
A_mm      success 0.996 / 0.999 / 0.999 = 0.998 ± 0.002 | IR 0.359 / 0.363 / 0.360 = 0.361 ± 0.002
A_mm:geo  success 0.995 / 0.997 / 0.997 = 0.996 ± 0.002 | IR 0.355 / 0.356 / 0.353 = 0.355 ± 0.002
B_geo     success 0.990 / 0.993 / 0.990 = 0.991 ± 0.002 | IR 0.147 / 0.153 / 0.153 = 0.151 ± 0.004
B_mm      success 0.999 / 0.999 / 0.996 = 0.998 ± 0.002 | IR 0.176 / 0.178 / 0.179 = 0.178 ± 0.002
B_mm:geo  success 0.988 / 0.996 / 0.992 = 0.992 ± 0.004 | IR 0.150 / 0.152 / 0.154 = 0.152 ± 0.002

Image effect per seed (IR, mm with images vs mm:geo same checkpoint; livers better, Wilcoxon):
  A seed 0: Δ +0.0036 (+1 %), better 30/46, p=0.02
  A seed 1: Δ +0.0063 (+2 %), better 33/46, p=6e-05
  A seed 2: Δ +0.0069 (+2 %), better 35/46, p=2e-05
  B seed 0: Δ +0.0260 (+17 %), better 46/46, p=3e-14
  B seed 1: Δ +0.0256 (+17 %), better 46/46, p=3e-14
  B seed 2: Δ +0.0255 (+17 %), better 46/46, p=3e-14

## StereoMIS P2 (345 pairs): success (RRE < 15°, RTE < 20 mm) and IR@10; per seed then mean ± SD
FPFH      success 0.464 | IR 0.134
A_geo     success 0.380 / 0.427 / 0.432 = 0.413 ± 0.029 | IR 0.173 / 0.212 / 0.214 = 0.200 ± 0.023
A_mm      success 0.397 / 0.406 / 0.389 = 0.397 ± 0.009 | IR 0.185 / 0.217 / 0.211 = 0.205 ± 0.017
A_mm:geo  success 0.371 / 0.406 / 0.415 = 0.397 ± 0.023 | IR 0.185 / 0.213 / 0.219 = 0.206 ± 0.018
B_geo     success 0.542 / 0.585 / 0.568 = 0.565 ± 0.022 | IR 0.184 / 0.203 / 0.202 = 0.196 ± 0.011
B_mm      success 0.565 / 0.624 / 0.637 = 0.609 ± 0.038 | IR 0.241 / 0.258 / 0.254 = 0.251 ± 0.009
B_mm:geo  success 0.516 / 0.564 / 0.590 = 0.557 ± 0.037 | IR 0.182 / 0.200 / 0.200 = 0.194 ± 0.010
```

## IRCAD v2 (frozen pipeline, `given`): mean TRE over livers, mean ± SD over the 3 seeds

| model | 5 % | 10 % | 15 % | 25 % |
|---|---|---|---|---|
| A_geo | 45.86 ± 1.16 | 15.99 ± 0.84 | 8.60 ± 0.13 | 5.29 ± 0.05 |
| A_mm:geo | 44.36 ± 2.08 | 16.48 ± 0.47 | 8.35 ± 0.42 | 5.23 ± 0.01 |
| B_geo | 56.07 ± 0.94 | 15.52 ± 1.44 | 7.80 ± 0.80 | 5.39 ± 0.13 |
| B_mm:geo | 55.28 ± 3.04 | 16.55 ± 1.65 | 8.21 ± 0.77 | 5.54 ± 0.15 |

## Reading
- **The headline results do not depend on the training seed.**
  - On the synthetic test, success SDs are ≤ 0.004 and inlier-ratio SDs ≤ 0.004.
  - On IRCAD v2, TRE SDs are ≤ 1.7 mm at 10–25 % and ≤ 3 mm at 5 %. FPFH stays 25–60 mm worse at
    5–15 %.
- **The image effect replicates in every seed** (same checkpoint, images vs `:geo`):
  - Arm B: +17 % inlier ratio, better on 46 / 46 livers, p 3e-14 for each seed.
  - Arm A: small but consistent, +1–2 %, better on 30–35 / 46 livers, p ≤ 0.02.
- **StereoMIS P2** (real video, kinematic pseudo-GT; success = RRE < 15° and RTE < 20 mm):
  - Ranking: B_mm 0.609 ± 0.038 > B_geo 0.565 ± 0.022 ≈ B_mm:geo 0.557 ± 0.037 > FPFH 0.464 >
    arm A ≈ 0.40.
  - Images raise arm B's success by 0.05–0.06 in every seed, and its inlier ratio by +29 %
    (0.251 vs 0.194).
  - Arm A stays below FPFH for every seed, so its sim-to-real gap is not a seed effect.

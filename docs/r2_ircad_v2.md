# R2: IRCAD v2, the locked 20-liver test (2026-10-28)

Pre-registered in docs/protocol.md (2026-10-28, plus the generation deviation of the same day).
`scripts/eval/ircad_eval.py --split v2`, run once for the seed-0 descriptors. Seed replicates follow
(`scripts/bench/seed_replicates.sh`). Report: `scripts/bench/ircad_v2_report.py
outputs/eval/ircad_v2/*.parquet`.

**Set.** Livers 1–20 × 25 pairs at 5 / 10 / 15 %, and 15 livers × 25 at 25 %: livers 4, 8, 9, 13
and 17 cannot provide 25 % visibility. Seed base 9000, the generator unchanged otherwise.
**Deformation-only (GT) residual** (mean ± SD over pairs):

| crop | v2 | test livers 1–2 |
|---|---|---|
| 5 % | 6.9 ± 2.8 mm | 5.7 mm |
| 10 % | 6.8 ± 2.8 mm | 5.5 mm |
| 15 % | 6.6 ± 2.7 mm | 5.4 mm |
| 25 % | 5.9 ± 2.7 mm | 5.1 mm |

So the average liver deforms more than livers 1–2.

## Across-liver means (seed-0 models; 95 % bootstrap CI over livers)

| frozen, given: TRE [CI] / success | 5 % | 10 % | 15 % | 25 % |
|---|---|---|---|---|
| A_geo | 47.09 [34.46, 61.35] / 0.57 | 16.44 [8.61, 27.60] / 0.90 | 8.67 [6.12, 12.25] / 0.95 | 5.34 [4.61, 6.39] / 0.97 |
| A_mm:geo | 43.75 [31.74, 57.13] / 0.60 | 16.28 [8.23, 28.06] / 0.89 | 8.16 [6.10, 11.58] / 0.95 | 5.22 [4.59, 6.07] / 0.97 |
| B_geo | 55.27 [43.43, 68.08] / 0.56 | 14.24 [10.39, 18.65] / 0.93 | 7.26 [6.26, 8.37] / 0.96 | 5.54 [4.54, 7.12] / 0.98 |
| B_mm:geo | 58.72 [46.35, 71.48] / 0.53 | 16.03 [11.73, 21.45] / 0.90 | 7.99 [6.44, 9.98] / 0.95 | 5.38 [4.63, 6.45] / 0.97 |
| fpfh_r20 | 106.65 [99.46, 113.27] / 0.11 | 62.98 [49.37, 77.70] / 0.48 | 33.77 [24.15, 45.40] / 0.71 | 13.75 [6.61, 24.65] / 0.90 |
| random | 128.08 [123.35, 132.57] / 0.01 | 122.66 [118.09, 127.45] / 0.03 | 118.90 [114.04, 123.44] / 0.06 | 94.28 [83.99, 103.06] / 0.21 |
| sonata | 83.71 [74.31, 93.69] / 0.15 | 81.41 [73.62, 89.05] / 0.23 | 70.90 [63.70, 78.00] / 0.31 | 46.09 [36.90, 54.95] / 0.51 |

| frozen, so3: TRE [CI] / success | 5 % | 10 % | 15 % | 25 % |
|---|---|---|---|---|
| A_geo | 54.70 [41.10, 69.15] / 0.54 | 18.57 [8.96, 32.59] / 0.88 | 9.66 [6.15, 14.93] / 0.94 | 5.33 [4.60, 6.38] / 0.97 |
| A_mm:geo | 51.12 [36.74, 67.61] / 0.58 | 18.07 [8.75, 31.51] / 0.89 | 8.93 [6.14, 13.09] / 0.95 | 5.33 [4.59, 6.40] / 0.97 |
| B_geo | 55.37 [43.65, 68.23] / 0.54 | 13.77 [9.91, 18.83] / 0.93 | 7.66 [6.20, 9.70] / 0.96 | 5.49 [4.55, 6.96] / 0.97 |
| B_mm:geo | 59.37 [47.11, 71.68] / 0.52 | 16.01 [11.40, 22.01] / 0.91 | 7.69 [6.33, 9.43] / 0.95 | 5.37 [4.59, 6.52] / 0.97 |
| fpfh_r20 | 113.86 [106.17, 121.03] / 0.08 | 58.94 [46.28, 71.87] / 0.49 | 31.33 [22.21, 41.50] / 0.72 | 15.52 [7.16, 27.68] / 0.89 |
| random | 129.34 [124.38, 134.17] / 0.01 | 122.80 [117.71, 127.71] / 0.03 | 117.37 [111.54, 123.35] / 0.06 | 89.13 [80.82, 96.94] / 0.23 |
| sonata | 130.03 [123.19, 136.59] / 0.01 | 122.31 [117.99, 126.79] / 0.03 | 115.49 [110.26, 121.17] / 0.04 | 107.01 [100.44, 113.78] / 0.08 |

| knn3_prior, given: TRE [CI] / success | 5 % | 10 % | 15 % | 25 % |
|---|---|---|---|---|
| A_geo | 27.77 [18.08, 39.96] / 0.73 | 11.85 [7.88, 17.78] / 0.93 | 7.35 [6.02, 9.31] / 0.96 | 5.24 [4.57, 6.15] / 0.98 |
| A_mm:geo | 26.99 [16.86, 39.77] / 0.75 | 12.90 [7.74, 20.43] / 0.92 | 7.02 [5.94, 8.68] / 0.96 | 5.25 [4.57, 6.21] / 0.98 |
| B_geo | 27.76 [21.61, 34.70] / 0.71 | 9.27 [7.89, 11.01] / 0.95 | 6.91 [6.11, 7.99] / 0.95 | 5.23 [4.53, 6.21] / 0.97 |
| B_mm:geo | 30.26 [22.66, 39.43] / 0.72 | 11.46 [8.30, 15.92] / 0.93 | 6.84 [6.08, 7.72] / 0.96 | 5.19 [4.54, 6.07] / 0.97 |
| fpfh_r20 | 78.28 [67.16, 89.32] / 0.29 | 29.23 [21.93, 37.58] / 0.70 | 14.07 [9.66, 20.74] / 0.86 | 8.22 [4.99, 13.62] / 0.94 |
| random | 96.39 [90.70, 102.21] / 0.03 | 91.15 [86.14, 95.99] / 0.08 | 81.38 [76.30, 86.16] / 0.14 | 53.53 [45.45, 61.49] / 0.36 |
| sonata | 86.13 [72.61, 100.29] / 0.19 | 79.10 [69.27, 89.81] / 0.29 | 53.85 [44.18, 64.84] / 0.48 | 26.32 [19.89, 33.39] / 0.73 |

## Primary comparisons (frozen pipeline, given init; per-liver mean TRE; Wilcoxon, Holm over crops)

- B_geo vs fpfh_r20, 5 %: median Δ -52.49 mm, livers better / worse 20 / 0, p 1.9e-06 (Holm 7.6e-06)
- B_geo vs fpfh_r20, 10 %: median Δ -50.79 mm, livers better / worse 20 / 0, p 1.9e-06 (Holm 7.6e-06)
- B_geo vs fpfh_r20, 15 %: median Δ -23.89 mm, livers better / worse 20 / 0, p 1.9e-06 (Holm 7.6e-06)
- B_geo vs fpfh_r20, 25 %: median Δ -1.14 mm, livers better / worse 12 / 3, p 0.0026 (Holm 0.0026)
- A_geo vs fpfh_r20, 5 %: median Δ -66.16 mm, livers better / worse 19 / 1, p 5.7e-06 (Holm 2.3e-05)
- A_geo vs fpfh_r20, 10 %: median Δ -48.26 mm, livers better / worse 19 / 1, p 3.6e-05 (Holm 7.2e-05)
- A_geo vs fpfh_r20, 15 %: median Δ -23.87 mm, livers better / worse 19 / 1, p 1.9e-05 (Holm 5.7e-05)
- A_geo vs fpfh_r20, 25 %: median Δ -0.75 mm, livers better / worse 13 / 2, p 0.0034 (Holm 0.0034)

## match_ext (given): livers 1-2 (clean) vs 3-20 (flagged: selected on dev pairs of these livers)

| TRE livers 1-2 / 3-20 | 5 % | 10 % | 15 % | 25 % |
|---|---|---|---|---|
| A_geo | 33.4 / 27.1 (succ 0.54 / 0.75) | 5.8 / 12.5 (succ 1.00 / 0.92) | 5.2 / 7.6 (succ 1.00 / 0.96) | 4.8 / 5.3 (succ 1.00 / 0.97) |
| A_mm:geo | 28.7 / 26.8 (succ 0.62 / 0.76) | 5.9 / 13.7 (succ 1.00 / 0.91) | 5.2 / 7.2 (succ 1.00 / 0.96) | 4.8 / 5.3 (succ 1.00 / 0.97) |
| B_geo | 19.6 / 28.7 (succ 0.76 / 0.70) | 5.8 / 9.7 (succ 1.00 / 0.95) | 5.2 / 7.1 (succ 1.00 / 0.94) | 4.6 / 5.3 (succ 1.00 / 0.97) |
| B_mm:geo | 14.3 / 32.0 (succ 0.82 / 0.71) | 6.3 / 12.0 (succ 0.98 / 0.92) | 5.2 / 7.0 (succ 1.00 / 0.96) | 4.7 / 5.3 (succ 1.00 / 0.97) |
| fpfh_r20 | 66.5 / 79.6 (succ 0.38 / 0.28) | 22.3 / 30.0 (succ 0.76 / 0.69) | 10.3 / 14.5 (succ 0.90 / 0.85) | 4.6 / 8.8 (succ 1.00 / 0.93) |
| random | 92.4 / 96.8 (succ 0.02 / 0.04) | 76.8 / 92.7 (succ 0.20 / 0.06) | 73.6 / 82.2 (succ 0.18 / 0.13) | 53.6 / 53.5 (succ 0.40 / 0.35) |
| sonata | 77.3 / 87.1 (succ 0.12 / 0.19) | 79.8 / 79.0 (succ 0.24 / 0.30) | 60.0 / 53.2 (succ 0.46 / 0.48) | 37.8 / 24.6 (succ 0.64 / 0.74) |

**Per-liver success** (frozen, `given`), for example at 10 % / 15 %:
- B_geo is ≥ 0.80 on every liver except liver 12 at 10 % (0.68; 0.80 at 15 %).
- A_geo also drops on liver 3 at 10 % (0.24).
- FPFH ranges from 0.00 to 0.92 across livers.

## Reading
- **The learned descriptors beat FPFH on every liver** where geometry is scarce:
  - 20 / 20 livers at 5, 10 and 15 %, and 12–13 / 15 at 25 %
  - Holm-corrected p ≤ 0.003 at every crop, for both arms
  - median per-liver TRE gain: 52–66 mm at 5 %, 48–51 at 10 %, 24 at 15 %, 0.8–1.1 at 25 %
- **Success is high on the full set:** 0.89–0.93 at 10 %, 0.95–0.96 at 15 %, 0.97–0.98 at 25 %
  (frozen, `given`). The mean TRE is pulled up by the remaining failures, which concentrate on a
  few livers (12; 3 for arm A).
- **The paper's test livers 1–2 are easier than average.** On livers 1–2 we reach 5.2–6.3 mm at
  10–15 % with `match_ext`, against 7.0–13.7 mm on livers 3–20. The comparison with LiverMatch,
  reported on livers 1–2 only, therefore says nothing about the other 18 livers. A fair
  all-liver comparison needs the baselines re-run (R1).
- **`match_ext`** (k-NN + ≤ 75° prior; selected on dev pairs of livers 3–20):
  - lowers mean TRE at 5–10 % (e.g. B_geo 10 %: 14.2 → 9.3 mm; 5 %: 55 → 28 mm)
  - leaves 15–25 % unchanged
  - Its clean numbers are livers 1–2. On 3–20 it is flagged.
- **Arms and images.** Arm A and arm B are within each other's CIs at every crop. Image-trained
  checkpoints run geometry-only (`:geo`) do not differ from geometry-only training, as on livers 1–2.
- **Initial pose (`so3` vs `given`)** changes nothing for the learned descriptors and FPFH. Frozen
  Sonata collapses without the generator's limited rotation.
- **Nulls.** Random descriptors fail (TRE ≥ 89 mm, success ≤ 0.23). With the prior, random
  descriptors reach 0.36 success at 25 %, so `match_ext` numbers at large crops partly reflect the
  prior's geometric search.

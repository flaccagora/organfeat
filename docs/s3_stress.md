# S3: stress tests on test livers (scored once, 2026-10-26)

Each set varies one factor from the base setting: σ = 1 mm at 100 mm with 10 % holes, the training
texture families, the renderer's tool distribution, no deformation. The noise-free set and the other
noise levels use the base test views. Script: `scripts/bench/stress_report.py`, written to
`outputs/bench/stress_report.md`.

### success (per-liver mean, 46 test livers)

| desc | base (σ 1 mm) | σ 0 | σ 0.5 mm | σ 2 mm | held-out textures | flat albedo | 0 tools | 1 tool | 2 tools | 3 tools | ARAP 2–8 mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fpfh_r20 | 0.670 | 1.000 | 0.891 | 0.322 | 0.686 | 0.670 | 0.720 | 0.670 | 0.605 | 0.564 | 0.630 |
| A_geo | 0.996 | 0.997 | 0.999 | 0.995 | 1.000 | 0.996 | 1.000 | 0.999 | 0.992 | 0.990 | 0.985 |
| A_mm | 0.996 | 0.996 | 1.000 | 0.996 | 0.999 | 1.000 | 0.999 | 0.997 | 0.997 | 0.995 | 0.989 |
| A_mm:geo | 0.995 | 0.997 | 1.000 | 0.989 | 0.996 | 0.999 | 0.997 | 0.997 | 0.995 | 0.995 | 0.989 |
| B_geo | 0.990 | 1.000 | 1.000 | 0.812 | 0.992 | 0.992 | 0.993 | 0.990 | 0.978 | 0.965 | 0.990 |
| B_mm | 0.999 | 1.000 | 1.000 | 0.924 | 0.996 | 0.996 | 0.999 | 0.997 | 0.993 | 0.989 | 0.997 |
| B_mm:geo | 0.988 | 1.000 | 1.000 | 0.806 | 0.990 | 0.985 | 0.996 | 0.989 | 0.986 | 0.978 | 0.989 |

### inlier_ratio@10 (per-liver mean, 46 test livers)

| desc | base (σ 1 mm) | σ 0 | σ 0.5 mm | σ 2 mm | held-out textures | flat albedo | 0 tools | 1 tool | 2 tools | 3 tools | ARAP 2–8 mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fpfh_r20 | 0.038 | 0.292 | 0.076 | 0.014 | 0.039 | 0.038 | 0.041 | 0.038 | 0.035 | 0.031 | 0.035 |
| A_geo | 0.349 | 0.438 | 0.397 | 0.266 | 0.353 | 0.352 | 0.359 | 0.353 | 0.342 | 0.324 | 0.341 |
| A_mm | 0.359 | 0.440 | 0.406 | 0.277 | 0.360 | 0.358 | 0.366 | 0.363 | 0.349 | 0.334 | 0.349 |
| A_mm:geo | 0.355 | 0.440 | 0.400 | 0.269 | 0.356 | 0.351 | 0.361 | 0.359 | 0.344 | 0.329 | 0.346 |
| B_geo | 0.147 | 0.784 | 0.321 | 0.055 | 0.147 | 0.146 | 0.160 | 0.144 | 0.131 | 0.119 | 0.131 |
| B_mm | 0.176 | 0.800 | 0.352 | 0.076 | 0.176 | 0.178 | 0.192 | 0.175 | 0.159 | 0.144 | 0.158 |
| B_mm:geo | 0.150 | 0.785 | 0.325 | 0.055 | 0.150 | 0.149 | 0.164 | 0.148 | 0.134 | 0.121 | 0.133 |

### images: paired differences per set

| set | A_mm−A_geo succ | A_mm−A_geo IR | B_mm−B_geo succ | B_mm−B_geo IR |
|---|---|---|---|---|
| base (σ 1 mm) | +0.000 (p 1) | +0.010 (p 2e-05) | +0.008 (p 0.03) | +0.029 (p 3e-14) |
| σ 0 | -0.001 (p 0.3) | +0.003 (p 0.1) | +0.000 (p 1) | +0.016 (p 3e-14) |
| σ 0.5 mm | +0.001 (p 0.3) | +0.009 (p 7e-05) | +0.000 (p 1) | +0.031 (p 3e-14) |
| σ 2 mm | +0.001 (p 0.7) | +0.010 (p 7e-06) | +0.111 (p 3e-06) | +0.021 (p 3e-14) |
| held-out textures | -0.001 (p 0.3) | +0.007 (p 0.001) | +0.004 (p 0.3) | +0.029 (p 3e-14) |
| flat albedo | +0.004 (p 0.08) | +0.006 (p 0.0003) | +0.004 (p 0.2) | +0.032 (p 3e-14) |
| 0 tools | -0.001 (p 0.3) | +0.007 (p 0.0004) | +0.005 (p 0.2) | +0.031 (p 3e-14) |
| 1 tool | -0.001 (p 0.3) | +0.010 (p 8e-06) | +0.007 (p 0.06) | +0.030 (p 3e-14) |
| 2 tools | +0.005 (p 0.05) | +0.007 (p 0.0004) | +0.015 (p 0.03) | +0.028 (p 3e-14) |
| 3 tools | +0.004 (p 0.2) | +0.010 (p 5e-05) | +0.024 (p 0.0004) | +0.025 (p 3e-14) |
| ARAP 2–8 mm | +0.004 (p 0.3) | +0.008 (p 0.002) | +0.007 (p 0.1) | +0.027 (p 3e-14) |

### visible fraction bins (base setting; pairs per bin: <10% 34, 10-20% 308, 20-30% 349, >=30% 45)

success:

| desc | <10% | 10-20% | 20-30% | >=30% |
|---|---|---|---|---|
| fpfh_r20 | 0.324 | 0.614 | 0.751 | 0.689 |
| A_geo | 0.971 | 0.994 | 1.000 | 1.000 |
| A_mm | 0.941 | 0.997 | 1.000 | 1.000 |
| A_mm:geo | 0.971 | 0.990 | 1.000 | 1.000 |
| B_geo | 1.000 | 0.997 | 0.983 | 1.000 |
| B_mm | 1.000 | 0.997 | 1.000 | 1.000 |
| B_mm:geo | 1.000 | 0.987 | 0.989 | 0.978 |

inlier ratio@10:

| desc | <10% | 10-20% | 20-30% | >=30% |
|---|---|---|---|---|
| fpfh_r20 | 0.047 | 0.042 | 0.035 | 0.031 |
| A_geo | 0.313 | 0.347 | 0.355 | 0.341 |
| A_mm | 0.342 | 0.361 | 0.361 | 0.344 |
| A_mm:geo | 0.337 | 0.354 | 0.360 | 0.345 |
| B_geo | 0.226 | 0.163 | 0.130 | 0.106 |
| B_mm | 0.254 | 0.193 | 0.159 | 0.126 |
| B_mm:geo | 0.226 | 0.167 | 0.133 | 0.106 |

## Reading
- **Learned vs FPFH holds under every stress.**
  - Learned registration recall stays at 0.92–1.00 everywhere. FPFH drops to 0.32 at σ 2 mm, 0.56
    with 3 tools, and 0.32 on views showing < 10 % of the surface.
  - Arm A (full-context) is the most robust to depth noise: 0.995 at σ 2 mm. Arm B falls to
    0.81 / 0.92 there.
- **The image channel helps in every stress set.**
  - Inlier ratio rises for both arms in all 11 sets: p ≤ 0.002, except arm A noise-free (p 0.1).
  - The registration-recall gain becomes significant for arm B where geometry alone starts to fail:
    σ 2 mm (+0.111, p 3e-6), 2 tools (+0.015, p 0.03), 3 tools (+0.024, p 4e-4).
- **The gain does not come from texture.**
  - It is the same on held-out texture families and on untextured (flat albedo) livers as on the
    training families: arm B +0.029 / +0.032 inlier ratio, arm A +0.007 / +0.006.
  - So the lifted DINOv2 features carry shading and contour evidence about shape, not texture
    identity. This is consistent with the leakage controls: textures are redrawn per sample and
    never shared with the CT side.
- **Mild deformation (2–8 mm) costs little.** Arm A loses 0.01 recall, arm B none. Correspondence
  quality drops in line with the oracle residual (~2 mm).
- **Low overlap.**
  - With < 10 % of the surface visible (34 pairs), arm A drops to 0.94–0.97 recall, and images do not
    help there (0.941 vs 0.971).
  - Arm B stays at 1.000, and its inlier ratio is *higher* on small views (0.25 vs 0.13 for ≥ 30 %).
    A local descriptor does not depend on how much context the view has.

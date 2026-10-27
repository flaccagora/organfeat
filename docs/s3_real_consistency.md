# S3: real-data consistency on StereoMIS (2026-10-26; P3 as pre-registered, P2 added; see protocol deviations)

Frame pairs come from the same sequence, binned by kinematic camera motion. Both sides are real
partial liver clouds (SGBM depth, liver from the frozen DINOv2 probe, tools and speculars removed).
The pseudo ground truth is the kinematic relative pose, with an error floor: registering with its
own correspondences ("oracle") passes 58 % on P3 (median 13.5° / 5.2 mm).

## P3 (held-out animal; 12 usable pairs out of the 90 requested)
| descriptor | success (RRE < 15°, RTE < 20 mm) | median mutual matches | of which in the overlap | inlier ratio@10 |
|---|---|---|---|---|
| oracle (pseudo-GT correspondences) | 0.58 | – | – | – |
| FPFH r20 | 0.17 | 19–103 | 4–48 | 0.08–0.19 |
| A_geo | 0.08 | 6–8 | 2–3 | 0.00 |
| A_mm (img) / :geo | 0.08 / 0.08 | 4–8 | 1–2 | 0.00 |
| B_geo | 0.33 | 54–140 | 5–60 | 0.14–0.21 |
| B_mm (img) / :geo | 0.17 / 0.17 | 50–134 | 9–62 | 0.15–0.24 |

Medians are per motion bin (5–10, 10–20, 20–40 mm).

## Diagnosis: arm A's features collapse partly on real clouds (`scripts/sanity/real_desc_probe.py`, P2_1)
| | A_mm synthetic | A_mm real | B_mm synthetic | B_mm real |
|---|---|---|---|---|
| mean pairwise cosine of descriptors | 0.03–0.05 | 0.17–0.43 | 0.02–0.04 | 0.14–0.22 |
| effective rank (of 32) | 17–24 | 6–13 | 27–28 | 17–21 |
| mutual matches img ↔ :geo, same points (of ~4,000) | – | 139–163 | – | 2,742–3,009 |

## Reading
- **Arm A does not transfer to real stereo clouds.** The full-context Sonata descriptor loses
  diversity on real data. Its image and geometry inference paths disagree, and it produces almost no
  mutual matches between two real views of the same liver.
- **Arm B transfers about as well as FPFH or better.** The local PPF descriptor keeps its diversity
  and gives FPFH-level or better correspondence numbers.
- **Likely causes** (not tested separately):
  - real depth noise and holes unlike the synthetic noise model
  - blocky liver masks from the patch-level probe, which change the cloud boundary
  - a global encoder's sensitivity to the cloud's extent and shape
- **Small sample.** The numbers are underpowered: 12 pairs, with the pseudo-GT capping success at
  ~0.6. The match-count gap between the arms is large, though.
- **No change to the selected models.** By protocol nothing is selected on real data. The locked
  DePoLL / P2I-LReg evaluation will show whether arm A's gap also hurts there.

## P2 (second animal; 345 pairs; oracle success 0.835, median 5.6° / 7.4 mm)
| descriptor | success (RRE < 15°, RTE < 20 mm) | median mutual matches | inlier ratio@10 (motion 5–10 / 10–20 / 20–40 mm) | point recall@10 (same bins) |
|---|---|---|---|---|
| oracle | 0.835 | – | – | – |
| FPFH r20 | 0.464 | 258–326 | 0.124 / 0.095 / 0.063 | 0.121 / 0.098 / 0.063 |
| A_geo | 0.380 | 18–23 | 0.200 / 0.077 / 0.048 | 0.161 / 0.090 / 0.062 |
| A_mm (img) | 0.397 | 18–20 | 0.222 / 0.105 / 0.091 | 0.166 / 0.096 / 0.069 |
| A_mm :geo | 0.371 | 18–21 | 0.214 / 0.100 / 0.036 | 0.152 / 0.086 / 0.066 |
| B_geo | 0.542 | 351–422 | 0.206 / 0.126 / 0.109 | 0.180 / 0.131 / 0.112 |
| **B_mm (img)** | **0.565** | 329–397 | **0.284 / 0.194 / 0.176** | **0.244 / 0.184 / 0.161** |
| B_mm :geo | 0.516 | 346–411 | 0.209 / 0.132 / 0.102 | 0.180 / 0.131 / 0.109 |

## Reading, both animals
- **Real-data ranking: B_mm > B_geo > FPFH > A_mm > A_geo.** The order is the same on P2 (powered)
  and P3 (underpowered). The full-context arm A falls below FPFH on real stereo data, even though
  it is the best descriptor on synthetic noisy test.
- **Arm A produces ~20 mutual matches per real pair** (vs ~300–400 for FPFH and PPF), with
  collapsed descriptor diversity: a synthetic-to-real gap.
- **Images help on real video too.** For arm B, inlier ratio rises 30–70 % relative and success
  0.516 → 0.565 (multimodal checkpoint, img vs :geo). Arm A gains +0.026 success. This matches the
  synthetic finding (strong for the local descriptor, small for the global one), now on real
  laparoscopic appearance.
- **The pseudo-GT caps every success rate:** P2 oracle 0.835, P3 0.58.
- **Statistics.** Pairs within a sequence are not independent and P2 has only 6 sequences, so these
  are descriptive; no p-values are claimed.

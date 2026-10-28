# D2: a deformation-tolerant rigid stage (post hoc, 2026-10-28)

Pre-registered in docs/protocol.md (2026-10-28). `scripts/eval/d2_pose_search.py`, outputs in
`outputs/eval/d2/{depoll_d2,ircad_v2_d2}/`.

**Pose search.** defreg's closest-point pose search (OT-20), re-implemented as
`ofeat.match_ext.closest_pose_search`:
- 171 rotations within 75° of the initial pose (22.5° grid)
- 20 ICP updates each, scored by the mean nearest-CT distance
- then the frozen ICP

**Arms.**
- `search`: grid only, descriptor-free.
- `search+<desc>`: the grid plus the descriptor's frozen and `match_ext` poses as seeds.
- Initial poses are bounded only: D1's poses on DePoLL; `given` on IRCAD v2.

## Results
```
DePoLL (bounded init, 13 states x 20 poses): success / median TRE / median clips TRE
                              success  tre_med  clips_med  rre_med
cond      arm                                                     
C1_shape  frozen/A_geo           0.67     0.08       0.08     0.07
          frozen/B_geo           1.00     0.06       0.06     0.06
          frozen/fpfh_r20        1.00     0.06       0.06     0.06
          match_ext/A_geo        0.79     0.06       0.06     0.06
          match_ext/B_geo        1.00     0.05       0.06     0.06
          match_ext/fpfh_r20     1.00     0.05       0.06     0.06
          search                 0.73     0.08       0.08     0.07
          search+A_geo           0.90     0.06       0.06     0.06
          search+B_geo           1.00     0.06       0.06     0.06
          search+fpfh_r20        1.00     0.06       0.06     0.06
C2_deform frozen/A_geo           0.02    97.08      94.75    80.73
          frozen/B_geo           0.02    99.84      89.29   125.90
          frozen/fpfh_r20        0.02    97.27      97.10   100.10
          match_ext/A_geo        0.03    62.48      69.88    44.22
          match_ext/B_geo        0.00    70.03      69.80    46.69
          match_ext/fpfh_r20     0.08    83.46      76.50    27.16
          search                 0.06    64.18      68.66    59.57
          search+A_geo           0.07    58.13      61.90    44.60
          search+B_geo           0.05    62.93      67.31    54.52
          search+fpfh_r20        0.09    62.49      61.91    36.67
C3_recon  frozen/A_geo           0.00    89.72      92.68    69.93
          frozen/B_geo           0.00   113.59     119.91   131.15
          frozen/fpfh_r20        0.01   101.32     101.75   112.42
          match_ext/A_geo        0.00    64.75      65.61    46.40
          match_ext/B_geo        0.00    75.44      64.58    42.39
          match_ext/fpfh_r20     0.00    80.31      68.45    72.66
          search                 0.02    60.64      63.32    47.70
          search+A_geo           0.00    62.72      64.57    45.99
          search+B_geo           0.01    64.79      64.77    46.57
          search+fpfh_r20        0.02    63.39      63.46    45.96
oracle median by cond: {'C1_shape': 0.0, 'C2_deform': 15.6, 'C3_recon': 15.6}
from_seed share (search+desc): {('C1_shape', 'search+A_geo'): 0.8, ('C1_shape', 'search+B_geo'): 0.96, ('C1_shape', 'search+fpfh_r20'): 0.97, ('C2_deform', 'search+A_geo'): 0.54, ('C2_deform', 'search+B_geo'): 0.4, ('C2_deform', 'search+fpfh_r20'): 0.58, ('C3_recon', 'search+A_geo'): 0.5, ('C3_recon', 'search+B_geo'): 0.44, ('C3_recon', 'search+fpfh_r20'): 0.3}

IRCAD v2 (given): mean over livers of per-liver mean TRE / success
                    tre_mm                      success                  
crop                    5      10     15     25      5     10    15    25
arm                                                                      
frozen/A_geo         47.83  16.15   8.63   5.32    0.57  0.90  0.95  0.97
frozen/B_geo         56.98  14.70   7.49   5.51    0.54  0.93  0.96  0.98
frozen/fpfh_r20     105.18  61.99  35.03  14.80    0.12  0.49  0.71  0.89
match_ext/A_geo      27.77  11.85   7.35   5.24    0.73  0.93  0.96  0.98
match_ext/B_geo      27.76   9.27   6.91   5.23    0.71  0.95  0.95  0.97
match_ext/fpfh_r20   78.28  29.23  14.07   8.22    0.29  0.70  0.86  0.94
search               60.14  40.66  25.34   9.54    0.38  0.59  0.72  0.91
search+A_geo         23.85  11.06   7.05   5.25    0.79  0.94  0.96  0.98
search+B_geo         24.28   8.22   6.52   5.30    0.81  0.97  0.97  0.98
search+fpfh_r20      48.42  16.86   8.86   5.88    0.53  0.88  0.95  0.97
```

**Diagnostic: why the search fails on deformed DePoLL.**
- At the best rigid (marker-oracle) pose, the deformed visible surface lies a median **8.3 mm** (C2)
  / **8.5 mm** (C3) from the reference CT. Only **~50 %** of its points are within 7.5 mm.
- The poses the search picks, ~60 mm wrong in TRE, fit at a median **4.0 mm**.
- In **26 / 26** state × condition cases the chosen wrong pose fits better than the true one. Only
  state 10, nearly undeformed, has 100 % of points within 7.5 mm at the oracle.

## Reading
- **DePoLL: rigid registration is ill-posed for surface-fit criteria.**
  - Under real deformation, the visible patch matches other places on the undeformed CT better than
    its true location.
  - Neither RANSAC overlap, closest-point fit nor descriptor consensus prefers the true pose, so no
    rigid search can recover it.
  - The search lowers median TRE (90–113 → 58–64 mm) but success stays ≤ 0.09. That misses the
    pre-registered bar (0.5), so BCPD is **not** added.
  - Recovering the pose needs deformation-aware matching (modelling the deformation inside the
    search) or constraints beyond the surface shape: anatomical landmarks such as ridges and the
    falciform ligament (as P2ILF uses), or priors on camera and patient position.
- **The descriptors carry the information, the search refines it.**
  - Without descriptors the search is weak even without deformation (DePoLL C1: 0.73; IRCAD v2 5 %:
    0.38).
  - Seeded with descriptor poses it becomes the best arm on IRCAD v2. search+B_geo vs `match_ext`
    B_geo:

    | crop | search+B_geo | `match_ext` B_geo |
    |---|---|---|
    | 5 % | 24.3 mm, success 0.81 | 27.8 mm, 0.71 |
    | 10 % | 8.2 mm, 0.97 | 9.3 mm, 0.95 |
    | 15 % | 6.5 mm, 0.97 | 6.9 mm, 0.95 |
    | 25 % | 5.3 mm, 0.98 | 5.2 mm, 0.97 |

  - It is a post-hoc pipeline variant, scored once on the locked v2 set.

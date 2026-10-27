# S4: locked evaluation log

- 2026-10-26: user decision. Run S4 now with the selected descriptors (no sim-to-real retraining
  first). Arm A's real-data gap (docs/s3_real_consistency.md) is reported, not fixed.
- 2026-10-26: DePoLL pipeline smoke (`--smoke`: state 01, 2 views, no metrics) passed. Liver points
  per view were ~400; the marker oracle-rigid TRE on state 01 is 24 mm.
- 2026-10-26: **locked runs submitted, once each**, with the same descriptors on both sets: fpfh_r20,
  frozen Sonata, and A_geo, A_mm, A_mm:geo, B_geo, B_mm, B_mm:geo (`scripts/bench/final_descs.sh`).
  - DePoLL: `scripts/eval/depoll_eval.py --run final`, OFEAT_UNLOCK_EVAL=depoll.
  - P2I-LReg locked cases: `scripts/eval/p2i_eval.py --split locked --run final`,
    OFEAT_UNLOCK_EVAL=p2i_lreg.
- 2026-10-26: **DePoLL final run 1 crashed with a segfault** (signal 11) after 6 min. It printed and
  saved no metrics. Resubmitted unchanged with PYTHONFAULTHANDLER=1 to locate the crash.
  Data-handling deviation, as in Project 10 stage 5. The method, descriptors and parameters are
  unchanged.
- 2026-10-26: **DePoLL run 2 completed** (7.7 min); results are in docs/s4_depoll.md. Success is
  ~0 for every descriptor, FPFH included, and 36 of 312 views had enough liver points. A validity
  check without descriptors (`scripts/sanity/depoll_frame_check.py`) confirms the frames are
  consistent: after the oracle-rigid fit the partial clouds lie 5–8 mm from the CT surface.
- 2026-10-26: **P2I-LReg locked run completed** (1 h 34 min, 281 frames, 17 cases). All descriptors
  gave Dice 0.53–0.54, except frozen Sonata (0.52).
- 2026-10-26: **post-hoc null calibration on the locked P2I-LReg cases.** Random descriptors go
  through the identical pipeline (`--desc random --run null`). It evaluates no method and changes
  nothing; it only says what Dice a pose from uninformative matches gets. Not pre-registered;
  reported as post-hoc.
- 2026-10-26: **P2I-LReg locked run completed** (17 cases, 281 frames); results in docs/s4_p2i.md.
- 2026-10-26: **post-hoc reference** (no descriptor): a random-pose CT projection gives Dice 0.498.
  All pipelines beat it by +0.02–0.05; the descriptors are indistinguishable from each other.
- 2026-10-26: noted a change made outside this session. `ofeat/eval/descriptors.py` gained a `random`
  descriptor (a null reference) at 19:59, after both locked runs had finished, so they are unaffected.
  Its seed uses Python's per-process salted `hash()`, so it is not reproducible across runs.
- 2026-10-26: P2I-LReg null calibration done. Random descriptors give Dice 0.532, against
  0.534–0.544 for all descriptors (docs/s4_p2i.md). S4 complete.
- 2026-10-26: **post-hoc null with random descriptors** through the full pipeline (`S4_p2i_null`,
  `outputs/eval/p2i/locked_null.parquet`): Dice 0.532.
  - No real descriptor beats it reliably (docs/s4_p2i.md). The +0.04 over a random pose comes from
    RANSAC + ICP, not from the descriptors.
  - P2I-LReg's reprojection metric does not discriminate descriptors.
  - This corrects the earlier log line ("all pipelines beat the random pose"): that is true, but it
    is not evidence for the descriptors.

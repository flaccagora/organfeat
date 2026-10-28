# Plan: a MICCAI 2027 method paper (phase 2, written 2026-10-28)

Decision (user, 2026-10-28): aim for MICCAI 2027 and open a new project phase.
- **Deadline:** not announced; we assume **~1 March 2027** and recheck in January. That leaves ~17
  weeks.
- **Built on:** docs/venue_assessment.md, docs/submission_plan.md and the phase-1 diagnostics D1 / D2.

## Thesis
**Under real deformation, a laparoscopic liver surface fits the CT better at wrong poses than at the
true one.** So rigid initialisation must come from deformation-tolerant correspondence consensus,
not from surface fit.
- *Evidence already in hand:*
  - DePoLL C2 / C3: wrong poses fit at 4.0 mm vs 8.3 mm at the true pose, in 26 / 26 cases
    (docs/d2_pose_search.md).
  - Every overlap- or fit-scored method fails there, while the same descriptors register 100 % of
    the undeformed porcine surface (docs/d1_depoll_decompose.md).
- *Method that follows from it:*
  - descriptors trained to be invariant to realistic large deformation
  - correspondences selected by local isometry consistency, not global rigid consensus
  - the pose from a robust fit to those correspondences, never from overlap or closest-point scores
  - optional non-rigid refinement

## Contributions (target)
1. **Analysis.** Real deformation makes surface-fit rigid registration ill-posed (DePoLL D1 / D2),
   and common benchmarks hide it (random-descriptor nulls; docs/s4_p2i.md, docs/ext_p2i_syn.md).
2. **Method.**
   - *M1:* self-supervised local descriptors (arm B, PPF-PointNet: the arm that transfers to real
     video) trained on TotalSeg CT renders with realistic large deformations.
   - *M2:* isometry-consistent correspondence selection.
   - *M3:* a robust consensus pose (+ BCPD refinement).
3. **Zero-shot evaluation** on three external benchmarks with published baselines (IRCAD / Neri;
   P2P in silico / in vitro) and on real in vivo data (DePoLL).
- **Images are out of the headline.** R4 showed that geometry + both-sides lifting beats our
  missing-modality design (docs/r4_lifting.md). They stay as an ablation at most.

## Method
- **M1: deformation-invariant descriptors.**
  - Retrain arm B on simulated deformations at DePoLL scale: rigid residuals 10–40 mm against
    phase 1's 2–8 mm ARAP stress.
  - Deformations: lobe displacements (as the IRCAD generator), global pneumoperitoneum-like
    flattening / compression, and ARAP handles (`ofeat/_vendor/organ_sim.py`). Exact correspondences
    from the simulation.
  - Loss as phase 1 (multi-positive InfoNCE). The positive radius scales with the local
    deformation, as a curriculum over deformation magnitude.
  - A small grid on synthetic val only (deformation range, positive radius); ~2.5 GPU-h per run.
- **M2: isometry-consistent selection.**
  - Liver deformation is near-isometric locally. For mutual / k-NN matches (i→j), (k→l) with
    |p_i − p_k| < r, require | |p_i − p_k| − |q_j − q_l| | < ε + δ·|p_i − p_k|.
  - Build the compatibility graph, then a spectral / maximal-clique inlier set (SC²-PCR / MAC
    style).
  - r, ε, δ are set from the simulation, on val.
- **M3: consensus pose.**
  - Robust rigid fit on the selected set: GNC-TLS / weighted Kabsch with a deformation-scaled
    kernel; optionally piecewise-rigid (multi-model) for lobes.
  - Overlap or closest-point scoring is never used to pick the pose (D2).
  - Then point-to-plane ICP with a deformation-scaled threshold, and optionally BCPD (defreg) for
    end-to-end TRE below the rigid floor.

## Evaluation (pre-registered in docs/protocol.md before any run of the new method)

| id | data | role | baselines |
|---|---|---|---|
| E1 | synthetic large-deformation test (TotalSeg test livers, our renders) | internal | FPFH, phase-1 pipelines (frozen / `match_ext` / D2 hybrid) |
| E2 | deformed IRCAD: the paper's test livers 1–2, and v2 (20 livers, locked) | external | published Table 2; LiverMatch retrained (R1) |
| E3 | P2P in silico (MSD livers, FEM, 6,050 test targets, visibility 0.2–1.0, full SO(3)) | external, zero-shot | published LiverMatch / Lepard ± P2P; LiverMatch / Lepard released weights |
| E4 | P2P in vitro phantoms (4 phantoms, 800 samples, CT fiducials) | **real physical deformation** | published numbers as E3 |
| E5 | DePoLL in vivo. Split by state, fixed now: **dev 01–04**, **test 05–13** | real, in vivo | all of the above where runnable |

- **Hygiene.** Selection only on synthetic val, P2P *train* subsets (we never train on MSD),
  IRCAD dev livers 3–20 (dev seeds) and DePoLL dev states.
- **Final runs.** E2-v2, E3 test, E4 and E5 test are run **once**, at W10.
- **Caveat on DePoLL.** States 05–13 were already seen by phase-1 methods (post hoc), but never by
  the new method or its tuning.

## Gates
- **G0, feasibility (end of W2).** On simulated DePoLL-scale deformations, corrupt GT
  correspondences to inlier ratios of 5–50 %.
  - Pass if M2 + M3 recover the oracle rigid pose where overlap-scored RANSAC fails, at the inlier
    ratio phase-1 descriptors reach under such deformation.
  - Fail → the scoring idea is wrong. Revisit (piecewise-rigid; deformation-aware search) before
    investing in M1.
- **G1, dev results (14 Dec).** The full method on dev sets.
  - (a) It must beat the phase-1 pipeline and retrained LiverMatch on P2P-train subsets and IRCAD
    dev.
  - (b) It must reach DePoLL-dev success ≥ 0.5, or TRE ≤ floor + 10 mm on ≥ 50 % of poses.
  - Outcomes:
    - (a) + (b): full paper.
    - (a) only: paper with E4 (in vitro) as the real-deformation evidence and DePoLL as the
      analysed limit.
    - Neither: fall back to the evaluation paper (IJCARS).

## Timeline (weeks from Mon 2 Nov 2026)
| weeks | work |
|---|---|
| W0 (28 Oct–1 Nov) | Pre-register this phase (methods, E1–E5, the DePoLL split, gates). Download P2P data and baseline weights (serial partition) and check licence and format. Finish the LiverMatch retraining (R1, queued). |
| W1–W2 | **G0** feasibility study. Implement M2 / M3 (`ofeat/match_ext.py`). Deformation simulator for M1 (lobe + global + ARAP, magnitudes calibrated on DePoLL's rigid residuals). |
| W3–W6 | M1 data generation and training (arm B; small grid on val). Integrate BCPD (copied from defreg with an origin note). Dev runs. |
| **14 Dec: G1** | go / no-go and scope (above) |
| W7–W9 | Ablations: M1 vs phase-1 descriptors; M2 vs mutual NN; M3 vs RANSAC-overlap; ± BCPD; ± images. 3 training seeds. Robustness to visibility and noise. |
| W10 (4–10 Jan) | Final runs, once: E2-v2, E3, E4, E5-test. |
| W11–W16 | Writing (8 pages + refs, anonymised), figures, internal review; code-release prep (licences: no IRCAD-generator copy, no DePoLL / SERV-CT derived data). |
| W17 | Submit (recheck the deadline in January). |

**Compute:** ~25 GPU-h training (M1 grid + seeds), ~50 GPU-h evaluation, CPU for simulation. This
is within the budget used so far.

## Risks
- **M1 may not learn at large deformation.** defreg 014: InfoNCE with Sonata failed at ~6 mm.
  Mitigations: the local PPF arm (it learned with 2–8 mm), a positive radius scaled with
  deformation, and a curriculum. G0 states how many inliers M2 / M3 need.
- **DePoLL may stay unsolved.** E4 (in vitro) still carries the real-deformation claim, and D1 / D2
  make the DePoLL limit a reported finding, not a hidden failure.
- **P2P data format or availability.** 2–3 days of loaders in W0; LiverMatch's code base is already
  vendored.
- **Novelty vs P2P / SC²-PCR / MAC.** Each piece exists in general registration. The contribution is
  the combination motivated by the ill-posedness analysis, CT-only self-supervision, and the
  zero-shot transfer across three liver benchmarks plus in vivo data. It must be argued, not
  assumed.
- **Dual submission.** Do not send the phase-1 evaluation paper to MIDL (archival, notification
  12 Feb, before MICCAI decisions). Its analysis goes into the MICCAI paper as motivation. An IPCAI
  long abstract (15 Jan) only if non-archival; check first.

## Decisions needed from the user
1. Skip MIDL (recommended, to avoid overlap).
2. Drop images from the headline (recommended, per R4).
3. Approve the DePoLL dev / test split by state (01–04 / 05–13).

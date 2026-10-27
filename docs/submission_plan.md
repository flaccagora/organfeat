# Plan: from the current study to a submission (written 2026-10-27)

Starting point: docs/venue_assessment.md, the five recommendations it ends with, and where the
evidence stands today (docs/summary.md, docs/ext_ircad.md).

## Target and timeline
- **Main target:** MICCAI 2027 (Auckland). The deadline is not announced; we assume about
  **1 March 2027** and recheck in January.
- **Checkpoint:** an IPCAI 2027 long abstract, **15 Jan 2027**.
- **Optional:** MIDL 2027 (abstract 30 Nov, paper 4 Dec 2026), only if gate G1 picks the
  evaluation framing (B, below).

| weeks (from) | work | gate or deadline |
|---|---|---|
| W0 (28 Oct) | Email Neri et al. (R1). Pre-register R2 / R3 / R5 in docs/protocol.md. Generate the IRCAD v2 test sets. One-day dataset survey (R3b). | – |
| W1–W2 (2 Nov) | R2: all-20-liver evaluation and training-seed replicates. R3: DePoLL whole-reconstruction protocol. Start R4a. | – |
| **G1, 16 Nov** | Does R3 give a real CT registration in the published range (TRE ~13 mm, Huang et al.)? Choose framing A or B. | MIDL go / no-go |
| W3–W6 (16 Nov) | R1: authors' files, or retrain LiverMatch / Lepard. R4a / R4b. R5 on DePoLL, if G1 is positive. | MIDL 4 Dec, framing B only |
| W7–W10 (14 Dec) | R4c (optional). Consolidate tables and figures. Draft the IPCAI abstract. | **IPCAI long abstract, 15 Jan** |
| W11–W17 (11 Jan) | Write the MICCAI paper (8 pages + references, anonymised). Internal review. Prepare the code release. | **MICCAI, ~1 Mar** |

## Ground rules (carried over from the study)
- **Write the protocol before running.** Every new experiment gets its protocol in docs/protocol.md
  (Deviations, dated) before it runs: data, metrics, success criterion, what is compared.
- **Spent sets stay labelled.** DePoLL and P2I-LReg are spent. Anything new on them is post hoc and
  labelled as such.
- **One new locked set.** The IRCAD v2 sets (R2) are the paper's new locked test, frozen before any
  descriptor runs on them.
- **Standalone.** Baseline code is copied or containerised into organfeat with origin notes; nothing
  imports defreg (see the correspondences note on independence from defreg).
- **No tuning on test.** Selection only on the synthetic val livers or on IRCAD dev (livers 3–20,
  dev seeds).

## R1: fair IRCAD baselines
**Goal.** LiverMatch and Lepard on exactly our test pairs, removing the "regenerated sets" caveat.
1. **Ask the authors first (W0).** Request the four test `.h5` files and the trained
   LiverMatch / Lepard / RegTR weights, or their per-pair predictions.
   - With their test files: rerun ours on them (`ircad_eval.py`, with a path option to add) and
     report against Table 2 like for like.
   - With their weights: run the baselines on our v2 sets too.
2. **Otherwise retrain (W3–W6).** Reproduce their training protocol:
   - Data: `gen_deformed_ircad.py --livers 3..20 --per-liver 560`, new seed base 5000.
   - Code: the upstream LiverMatch and Lepard training code, in a container built from defreg's
     recipe (ADR-0005). defreg has inference adapters only, and the published weights fail at
     IRCAD density (defreg results 009: 83–100 mm).
   - Cost: about 1–2 GPU-days per model.
   - **Check:** the retrained LiverMatch must land near the paper on livers 1–2 (e.g. ~6.4 mm at
     15 %) before it is used as a baseline.
3. **Report.** Baselines vs ours on identical pairs, with paired per-pair Wilcoxon tests and
   per-liver numbers. Baselines trained on livers 3–20 are compared on livers 1–2 only.

**Risk.** The KPConv build or the training code may not run on Leonardo. Fallback: report the
published numbers, with the regeneration caveat and GT-residual check of docs/ext_ircad.md.

## R2: all 20 livers as test, with confidence intervals
**Goal.** A larger, cleaner zero-shot test. Our descriptors never trained on IRCAD.
- **IRCAD v2 test set.** Livers 1–20 × 25 pairs per crop ratio (5 / 10 / 15 / 25 %), new seed base
  9000, so 500 pairs per crop. Generated and frozen in W0, run once.
- **What to report:**
  - frozen pipeline and `match_ext`, both init settings
  - per-liver means, with bootstrap 95 % CIs over livers (liver = unit)
  - `match_ext` is clean only on livers 1–2 (it was selected on dev livers 3–20); report it
    separately on 3–20
- **Training-seed replicates.** Retrain A_geo, A_mm, B_geo and B_mm with 2 extra seeds, under the
  same recipe and step-selection rule: 8 runs × ~2.3 h, about 18 GPU-h. Report mean ± SD over seeds
  on the synthetic test, IRCAD v2 and StereoMIS.
- **Optional.** Cross-validate the supervised baselines over livers (4 folds of 5), if R1 retraining
  works and there is time. That allows a 20-liver baseline comparison.
- **Cost.** About 2 GPU-h of evaluation plus 18 GPU-h of training.

## R3: one positive real CT registration
**Goal.** A real-data CT result in the range the literature reports.
- **R3a, DePoLL with whole per-state reconstructions** (post hoc; Neri et al. and Huang et al.
  protocol):
  - Partial: each state's `video_reconstruction.ply` (13 states). CT: the reference surface.
  - TRE on the 60 markers, against the oracle-rigid TRE.
  - 20 random initial poses per state, both inside the paper's ≤ 45° / ≤ 50 mm range and fully
    random, so 13 states are not the only sample.
  - Methods: FPFH, learned A/B (geometry only and with images), `match_ext`, the random null.
  - Compare with Huang et al. 12.9 mm and Neri et al.'s best of ~42 mm.
- **R3b, dataset survey (1 day, W0).** Any other public liver data with real surfaces and CT ground
  truth. Candidate already at hand: the Sparse Data Challenge phantom (stylus traces; data and
  scoring exist in defreg, to be copied). Our descriptors are off-distribution there, but a rigid
  result is cheap.
- **Gate G1:**
  - R3a TRE at or below ~15 mm, clearly better than the null → **framing A**.
  - At chance again → **framing B**, with the DePoLL failure analysed: the human→porcine shape gap,
    reconstruction artefacts, the deformation floor.

## R4: lifting baselines (the novelty critique)
**Goal.** Show that the missing-modality learned descriptor beats simply lifting appearance onto
the untextured CT.
- **R4a, DINOv2 on both sides** (Vödisch / Back-to-3D style):
  - Partial: DINOv2 lifted from the frame, as now, but without geometry.
  - CT: render the untextured, shaded mesh with the existing nvdiffrast renderer from ~32 views over
    the anterior hemisphere; lift DINOv2 to the visible points and average across views.
  - Then the fixed pipeline. About 1 week.
- **R4b, geometry ⊕ rendered-CT DINOv2** (DINOReg-like): FPFH or our geometry descriptor
  concatenated with R4a on both sides. A few days on top of R4a.
- **R4c, Diff3F** (optional, W7–W10): the official code (Stable Diffusion + ControlNet + DINOv2) to
  "texture" the CT, then render and lift. Weights fetched on the serial partition. About 1 GPU-day.
- **Evaluate** on the synthetic test in the primary setting (σ 1 mm, 10 % holes; images exist on
  the partial side) and on DePoLL if G1 is positive.
  - Expected: rendered-CT DINOv2 features do not match real or textured frames well. This is the
    case for the missing-modality design.

## R5: the image claim on real data with ground truth
**Goal.** Test images on/off on real frames that have CT ground truth.
- **R5a, DePoLL (only if G1 is positive).** Under the R3a protocol, compare A_mm vs A_mm:geo and
  B_mm vs B_mm:geo, paired over states × initial poses.
  - Image features come from the coloured reconstruction, rendered from virtual views and averaged
    per point (as in docs/s4_depoll.md). Better: from the real video frames, if DePoLL provides a
    camera trajectory (check `M.yml` and the video folder in the correspondences scratch, W0).
- **R5b, stronger real-video consistency** (optional): a dataset with accurate depth and pose ground
  truth (e.g. SCARED, structured-light depth), to complement StereoMIS's kinematic pseudo-ground
  truth. Still no CT.
- **If both fail:** the image result stays synthetic plus StereoMIS, and the paper presents it as
  analysis ("where images help: local descriptors, not via texture"), not as the headline.

## Paper framings (chosen at G1)
**A, method paper (MICCAI)**: "Self-supervised missing-modality liver descriptors for partial-to-CT
registration". Contributions:
1. A descriptor trained on CT renderings only, with an image channel on the laparoscopic side and a
   learned missing token on the CT side.
2. Zero-shot results on deformed IRCAD v2, against re-run LiverMatch / Lepard (R1, R2).
3. A real CT result on DePoLL (R3).
4. Lifting baselines (R4).
5. Where images help (R5).
6. A short evaluation-pitfalls section: the random-descriptor null and saturated metrics.

**B, evaluation paper (MIDL 4 Dec, or IJCARS)**: "What do partial-to-CT liver registration
benchmarks measure?"
1. Random-descriptor nulls match every method on P2I-LReg (Dice 0.532 vs 0.534–0.544).
2. Saturated synthetic recall (the null reaches 29 %).
3. Overlap-scored RANSAC rescues random matches.
4. Zero-shot IRCAD at the deformation floor.
5. The locked real-data failures, with their analysis.

Most of this exists already; R1 and R2 strengthen it.

## Release and housekeeping (W11–W17)
- **Licences before the repo goes public:**
  - The vendored IRCAD generator has no licence: remove it, ask the authors, or ship a download
    script instead.
  - No DePoLL- or IRCAD-derived data (CC BY-NC-ND); P2I-LReg under its Kaggle licence.
- **Anonymity.** Follow MICCAI's rules for arXiv preprints and code links during review.
- **Summary.** Update docs/summary.md and the README status table with each workstream as it
  finishes.

## Compute and effort (rough)
| item | GPU | engineering |
|---|---|---|
| R2: seed replicates + v2 evaluation | ~20 GPU-h | 2 days |
| R1: retrain two baselines (if needed) | 2–4 GPU-days | 1–2 weeks |
| R3a / R5a: DePoLL whole-reconstruction | < 10 GPU-h | 1 week |
| R4a / R4b | ~10 GPU-h | 1–1.5 weeks |
| R4c: Diff3F (optional) | ~1 GPU-day | 1 week |
| Writing | – | 6–7 weeks |

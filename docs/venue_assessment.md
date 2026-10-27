# State of the art on deformed IRCAD, and publication venues (2026-10-27)

This assessment came out of a deep-research run: 5 search angles, 20 fetched sources, and 24 claims
checked by 3 independent adversarial votes each. The run's final synthesis step hit a usage limit,
so the synthesis below was written by hand from the verified claims.
- The MIDL 2027 and IPCAI 2027 deadlines were re-checked on the official pages on 2026-10-27.
- The IROS, ICRA and WACV dates come from single fetched pages that were not vote-checked.
- Our numbers come from docs/ext_ircad.md.

Execution plan: docs/submission_plan.md.

## 1. Are we state of the art on deformed IRCAD?
**Partly.** We have the best published TRE at 10–25 % visibility, but the margins are
sub-millimetre and the test sets are regenerated. At 5 %, LiverMatch is still better.

**Published results.** The only published results on this benchmark are Table 2 of Neri et al.
2025 ([PMC12666070](https://pmc.ncbi.nlm.nih.gov/articles/PMC12666070/)). There, LiverMatch has the
lowest TRE at every visibility level. No later work reports rigid numbers on it:
- The generator repository has no sign of third-party use (1 star, 0 forks).
- The authors' April 2026 preprint is non-rigid and uses neither these test sets nor the wARAP
  generator ([arXiv 2604.13186](https://arxiv.org/html/2604.13186v1)).

**Comparison**, with the paper's initial poses (`given`). Each entry is TRE mm / MAE(R) ° / MAE(t) mm:

| visibility | best published | ours | verdict |
|---|---|---|---|
| 25 % | LiverMatch 4.81 / 2.78 / 2.78; Neri et al. MAE(R) 2.33 | 4.45 / 2.72 / 2.82 (frozen pipeline) | best TRE by 0.4 mm; Neri et al. better on rotation |
| 15 % | LiverMatch 6.39 / 3.08 / 3.73; Neri et al. 2.85 ° / 2.60 mm | 5.55 / 2.95 / 3.55 (frozen) | best TRE; Neri et al. better on rotation and translation |
| 10 % | LiverMatch 7.91 / 3.64 / 4.68 | 6.6–6.9 / 3.3–3.4 / 4.0 | best on all three; approximate, our set differs |
| 5 % | LiverMatch 18.69 / 7.47 / 8.41 | 24.5 / 10.6 / – (`match_ext`, ≤ 75° prior) | not SOTA; ~25 % of pairs still fail |

**Why this is not a clean SOTA claim:**
- **Regenerated test sets.** The public generator cannot reproduce the paper's test sets
  bit-for-bit: the test seed and the liver exclusion are commented out
  ([repo](https://github.com/Alberto-Neri/Laparoscopic_Organ_Deformation_wARAP)). Our 5 % and 10 %
  sets also have a lower GT residual than the paper's (5.7 vs 6.5 mm, 5.5 vs 6.2 mm).
- **Practically tied.** At 15–25 % the differences are below 1 mm, against a pooled SD of ~4.5 mm
  and a deformation residual of ~5 mm. The verifiers confirmed the GT row is not a floor: rigid
  methods can beat it on the hidden landmarks, as LiverMatch does at 25 %. There is no per-pair
  published data, so no significance test is possible.
- **Richer inputs than the published methods:** CT-side mesh normals, and partial normals oriented
  with the generator's camera position.
- **In our favour:** the published methods were trained on this generator's livers 3–20. Ours were
  trained on TotalSeg livers and are zero-shot on IRCAD.

**Neighbouring benchmarks** (not directly comparable):
- **P2P** ([arXiv 2412.19328](https://arxiv.org/html/2412.19328v2)): an add-on to LiverMatch and
  Lepard, on MSD livers at ≥ 20 % visibility. It reduces RMS-TRE, e.g. LiverMatch 12.9 → 8.7 mm in
  silico at the lowest visibility.
- **Huang et al., HTL 2025** ([PMC12673278](https://pmc.ncbi.nlm.nih.gov/articles/PMC12673278/)):
  DePoLL TRE 12.9 mm (Self-P2IR 24.4 mm), from stereo reconstructions. We are at chance on DePoLL
  under our view-based protocol.
- **Neri et al.'s patient-specific matcher**
  ([arXiv 2604.13186](https://arxiv.org/html/2604.13186v1)): inlier ratio 92 % on IRCAD liver 1,
  against LiverMatch 23 %. This is a strong correspondence-quality reference, though non-rigid and
  trained per patient.

## 2. Can this become a paper at a good venue?
**Yes, but not on the current evidence.**

**Not novel.** Lifting 2D foundation-model features, DINOv2 included, onto 3D surfaces as point
descriptors:
- [Diff3F](https://openaccess.thecvf.com/content/CVPR2024/html/Dutt_Diffusion_3D_Features_Diff3F_Decorating_Untextured_Shapes_with_Distilled_Semantic_CVPR_2024_paper.html),
  CVPR 2024; it already fuses DINOv2 with Stable Diffusion features
- [Back to 3D](https://arxiv.org/abs/2311.18113), CVPR 2024
- [Uzolas et al.](https://arxiv.org/abs/2503.18254), which adds self-supervised surface-aware
  refinement
- [Vödisch et al., RSS 2025](https://arxiv.org/abs/2502.19374): lifted DINOv2 descriptors through
  cosine nearest neighbours → RANSAC → ICP for LiDAR registration
- DINOv2 fused with geometric features for registration:
  [DINOReg](https://arxiv.org/abs/2509.24370) and [ZeroReg](https://arxiv.org/abs/2312.03032)

**What remains open.** All of the above need images on both sides (RGB-D or multi-view RGB). The
asymmetric case is unaddressed: an image on the laparoscopic side only, none on the CT. Candidate
contributions:
- the **missing-modality descriptor**: learned missing token and modality dropout, trained
  self-supervised on CT renderings only
- **where images help**: +20 % inlier ratio for a local descriptor, marginal for a global one, not
  via texture
- **zero-shot transfer** to deformed IRCAD that matches or beats supervised methods at ≥ 10–15 %
  visibility
- a **benchmark critique**:
  - on P2I-LReg, random descriptors match every method's Dice (0.532 vs 0.534–0.544)
  - its synthetic split is saturated (random descriptors register 29 %)
  - Open3D's overlap-scored RANSAC rescues random matches

**Weaknesses reviewers will find:**
- **The image claim has no real CT ground truth.** It rests on synthetic data plus StereoMIS
  consistency; IRCAD has no images.
- **The real CT registration tests failed.** DePoLL is at chance, while Huang et al. report
  12.9 mm. P2I-LReg cannot separate methods from a null.
- **The SOTA claim is weak:** regenerated sets, sub-millimetre margins, and the 5 % gap.
- **Precedent** (a fetched source, not vote-checked): LiverMatch was published through
  IJCARS/IPCAI with synthetic-only evaluation, so synthetic-only evidence is not disqualifying for
  that venue.

**Deadlines as of 2026-10-27:**

| venue | status | fit |
|---|---|---|
| IPCAI 2027 (Berlin, 28–30 Jun; IJCARS special issue) | full papers: intent closed 23 Oct, submission 30 Oct, so out. Long abstracts due **15 Jan 2027** ([site](https://sites.google.com/view/ipcai2027)) | good topic fit; the long abstract is a low-cost step |
| MIDL 2027 (14–16 Jul) | abstract **30 Nov 2026**, paper **4 Dec 2026** ([dates](https://2027.midl.io/important-dates)) | expects separate internal and external testing; best suited to an evaluation-centred paper |
| MICCAI 2027 (Auckland, 27 Sep–1 Oct) | deadline not announced ([MICCAI](https://miccai.org/upcoming-conferences/)); past cycles suggest late Feb or early Mar 2027 | main target for a method paper |
| IROS 2027 | 1 Mar 2027 ([tracker](https://mldeadlines.com/conference/iros/)) | robotics alternative |
| ICRA 2027, WACV 2027 | closed | – |
| IJCARS, MedIA, TMI | rolling | MedIA and TMI need a working real-data result |

## Recommendations
Target **MICCAI 2027**, with an **IPCAI long abstract** (15 Jan) as an early checkpoint. Submit to
**MIDL (4 Dec)** only if an evaluation-centred paper is wanted. Before submitting, in this order:
1. **Fair IRCAD baselines.** Run LiverMatch and Lepard ourselves on our regenerated sets, or obtain
   the authors' test files and weights.
2. **All 20 IRCAD livers as test**, with CIs. Our models never trained on IRCAD. Livers 3–20 were
   used only to select `match_ext`.
3. **One positive real CT result**, e.g. DePoLL with whole per-state reconstructions, as Neri and
   Huang do. It is post hoc, since DePoLL is spent.
4. **Lifting baselines:** DINOv2-only and Diff3F-style rendered-CT features, to meet the novelty
   critique.
5. **The image claim on real data with ground truth.** If impossible, frame images as analysis
   rather than the headline.

## Sources (verified 3-0 unless noted)
- Neri et al. 2025, Table 2: https://pmc.ncbi.nlm.nih.gov/articles/PMC12666070/
- Generator repo: https://github.com/Alberto-Neri/Laparoscopic_Organ_Deformation_wARAP
- Neri et al. 2026, patient-specific non-rigid: https://arxiv.org/html/2604.13186v1
- P2P: https://arxiv.org/html/2412.19328v2
- Huang, Yang, Jia, HTL 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12673278/
- Self-P2IR: https://arxiv.org/abs/2504.15152 (the dataset claim was not verified; the run hit the
  limit)
- Diff3F: https://arxiv.org/abs/2311.17024
- Back to 3D: https://arxiv.org/abs/2311.18113
- Uzolas et al.: https://arxiv.org/abs/2503.18254 (1 of 3 votes refuted the framing, not the numbers)
- Vödisch et al.: https://arxiv.org/abs/2502.19374
- DINOReg: https://arxiv.org/abs/2509.24370
- ZeroReg: https://arxiv.org/abs/2312.03032
- Venues: https://2027.midl.io/important-dates · https://sites.google.com/view/ipcai2027 ·
  https://miccai.org/upcoming-conferences/ · https://mldeadlines.com/conference/iros/ ·
  https://2027.ieee-icra.org/announcements/call-for-technical-papers/ · https://wacv.thecvf.com
- Refuted claims, not used:
  - "learned matchers already saturate P2I-LReg": the numbers came from the synthetic subset
  - "DINOv2 + geometry fusion is untried": DINOReg did it
  - "our TRE below the GT row means a protocol error": the GT row is not a floor

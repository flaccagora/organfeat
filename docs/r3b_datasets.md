# R3b: public data with real liver surfaces and CT ground truth (survey, 2026-10-28)

Purpose (docs/submission_plan.md, R3 and R5): find real data on which the descriptors can show a
CT registration, ideally with RGB on the intraoperative side. Searches covered public liver
registration datasets, ex vivo and in vivo porcine stereo + CT data, and the benchmarks cited by
recent liver registration papers.

| dataset | what it has | GT | usable for | status here |
|---|---|---|---|---|
| **DePoLL** (Modrzejewski et al., IJCARS 2019) | in vivo porcine, 13 deformed states; coloured stereo video reconstruction per state; pre- and intraop CT; 60 markers | markers (TRE) | R3a (whole reconstructions); R5a (images from the coloured reconstruction; no camera trajectory shipped) | on disk; spent at S4, so post hoc |
| **SERV-CT** (Edwards et al., MedIA 2022; [figshare](https://doi.org/10.5522/04/26352199)) | 2 ex vivo porcine torsos, 2 × 8 rectified stereo pairs (720 × 576 RGB); CT-derived reference depth and occlusion per view; per view, the CT anatomy surface (`CT/<nnn>/Anatomy.vtk`, ~684k points, the whole visible anatomy, **not liver-segmented**) and its CT→camera registration; a Creaform RGB scan of liver + kidney (Experiment 2) | CT-to-endoscope alignment (depth ~2 mm) | Downloaded 2026-10-28 (1.5 GB). Views are **close-ups of liver lobes** (checked Experiment 1 / 001), so partials are small, like the S4 DePoLL views. A liver-only CT model must first be segmented out of `Anatomy.vtk`. No deformation between CT and video. **Licence:** the readme says **CC BY-NC-SA 4.0**, figshare says CC BY 4.0, so we follow the stricter one and release no derived data. | downloaded; low priority unless R3a is positive |
| Clinical LLR + LUS ground truth (Rabbani et al., EnCoV, 4 patients) | monocular laparoscopy, CT, laparoscopic ultrasound | LUS-based registration error | 3D–2D only; would need monocular depth like P2I-LReg | not pursued |
| PIVOTS laparoscopic liver ([arXiv 2507.20337](https://arxiv.org/pdf/2507.20337)) | 6 surgical videos, CT models | semi-automatic intraoperative targets, no markers | weak GT | not pursued |
| KIT OpenCAS liver registration | silicone phantom: stereo-endoscopic partial surface + CT surface | phantom alignment | sim-to-real on geometry only; no realistic appearance | low priority |
| Sparse Data Challenge ([PMC10773576](https://pmc.ncbi.nlm.nih.gov/articles/PMC10773576/)) | phantom, sparse stylus traces, subsurface targets | blinded TRE | rigid category only; our descriptors are off-distribution on sparse traces | in defreg; copy if needed |
| Maier-Hein et al. 2014 (ex vivo porcine liver / kidney / heart, stereo + CT markers) | small masked regions | markers | no download found | – |
| P2I-LReg (in vivo, 21 patients) | monocular-depth clouds, masks, CT models | none (Dice proxy) | metric does not discriminate (docs/s4_p2i.md) | spent |

**Conclusion.** DePoLL (R3a) remains the main real CT test.
- **SERV-CT does contain liver views**, with real RGB and exact CT poses. But it needs two things
  first:
  - a liver segmentation of the CT anatomy surface
  - a way around its close-up, small-patch views, the regime where our descriptors already fail
    (DePoLL S4, IRCAD 5 %)
- **Use it only if R3a shows the descriptors work on real whole surfaces.** Its protocol goes into
  docs/protocol.md before any descriptor runs on it.

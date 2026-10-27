# Study summary: self-supervised liver point features for partial → CT matching (2026-10-26)

**Question.** Do self-supervised per-point descriptors that combine surface geometry with frozen
DINOv2 features, lifted onto visible points, improve matching of a partial visible liver surface to
its complete CT surface when the CT side has no appearance? The matching pipeline (mutual NN →
RANSAC → point-to-plane ICP) is fixed throughout.

Protocol: `docs/protocol.md` (pre-registered, with dated deviations). Stage write-ups:
- `docs/s0_findings.md`: setup, RANSAC bound, val baselines
- `docs/s1_findings.md`: geometry-only learning, diagnosis, selection
- `docs/s2_findings.md`: multimodal, selection
- `docs/s3_test.md`, `docs/s3_stress.md`, `docs/s3_ablations.md`: test, stress tests, ablations
- `docs/s3_real_consistency.md`: StereoMIS
- `docs/p2i_audit.md`: P2I-LReg data audit
- `docs/s4_depoll.md`, `docs/s4_p2i.md`, `docs/s4_locked.md`: locked evaluations and their log

## Methods compared
- **Baselines.**
  - FPFH r20 (radius chosen on val)
  - frozen Sonata, raw and PCA-32
  - "geometry ‖ DINO" concatenation with a zero image block on the CT side
- **Arm A (pre-registered):** Sonata (PTv3), fine-tuned, full context.
- **Arm B (added in S1 with the user):** a strictly local PPF-PointNet trained from scratch.
- **Variants per arm:**
  - *geo*: geometry-only training
  - *mm*: multimodal training, inference with images
  - *mm:geo*: the multimodal checkpoint with geometry-only inference
- **Training:** synthetic laparoscopic renders of 408 TotalSeg training livers, randomised textures,
  stereo-like depth noise, independent random rigid frames per cloud. Multi-positive InfoNCE with a
  safe radius; visual-modality dropout; λ = 0 (the consistency term hurt); μ = 0.5 view pairs.

## Main results
1. **Learned descriptors vs FPFH, synthetic test** (46 unseen livers; primary setting σ = 1 mm, 10 %
   holes).
   - Registration recall: learned 0.99–1.00, FPFH 0.67, frozen Sonata 0.14.
   - The learned descriptors stay at 0.92–1.00 under every stress (σ 2 mm, held-out or flat
     textures, 0–3 tools, ARAP deformation, < 10 % visible surface). FPFH falls to 0.32–0.72.
2. **Images help consistently but modestly, and they help the local descriptor most** (synthetic
   test).
   - Registration recall is at ceiling, so the effect shows in correspondence quality.
   - Arm A: inlier ratio +0.010 (+3 %, 36 / 46 livers, p 2e-5).
   - Arm B: +0.029 (+20 %, 46 / 46 livers), plus a recall gain where geometry fails: σ 2 mm,
     0.81 → 0.92.
   - The gain is unchanged on held-out texture families and on untextured livers. The DINOv2 channel
     supplies shading and contour evidence about shape, not texture identity.
   - Multimodal training alone does not improve geometry-only inference.
3. **Naive concatenation cannot use images when the CT side has none.** It is rank-equivalent to the
   geometry descriptor under cosine nearest neighbours.
4. **Ablations.**
   - Sonata pretraining matters: −0.086 inlier ratio from scratch.
   - Training data saturates by ~100 livers.
   - Modality dropout makes no measurable difference on synthetic data.
   - The cross-modal consistency term hurts.
5. **Real stereo video (StereoMIS, no CT; pseudo ground truth from kinematics).**
   - Arm A does not transfer. Its descriptors partly collapse on real SGBM clouds: ~20 mutual matches
     per pair, below FPFH.
   - Arm B transfers. On P2 (345 pairs) success is B_mm 0.565 > B_geo 0.542 > FPFH 0.464 > A_mm
     0.397, and images raise B's inlier ratio by 30–70 % relative. The oracle ceiling is 0.835.
6. **Locked DePoLL** (porcine; run once): **no method registers, FPFH included.**
   - Success ~0, rotation errors ~110–140°.
   - Only 36 of 312 views had ≥ 256 liver points.
   - Causes: tiny partial reconstructions (~6 × 6 cm), the human → porcine shape gap, a 22 mm
     deformation floor.
7. **Locked P2I-LReg** (human; run once, 17 cases, 281 frames):
   - All descriptors give Dice 0.534–0.544 (ceiling 0.881). They beat a random *pose* (0.498;
     p ≤ 4e-4), but not random *descriptors* fed through the same pipeline (0.532).
   - So the small gain comes from the RANSAC + ICP surface fit, not from any descriptor.
   - Frozen Sonata (0.517) is worse.
   - Two causes: the provided monocular clouds are affine-like, not metric, and the reprojection
     metric is insensitive to errors of a surface-consistent pose. P2I-LReg as evaluated cannot
     separate descriptors (`docs/s4_p2i.md`).

## Answer to the research question
- **On synthetic CT-rendered data:** self-supervised descriptors far outperform FPFH and frozen
  scene encoders for partial → CT matching. Lifted image features improve correspondence quality
  without any appearance on the CT side, strongly for a local descriptor and marginally for a
  global-context one.
- **On real data:** the image benefit carries over to stereo video for the local descriptor. The
  global-context descriptor suffers a sim-to-real gap.
- **On the two paired real CT registration sets:** neither shows a measurable registration benefit
  of any descriptor over FPFH. DePoLL is unregistrable under this protocol for every method;
  P2I-LReg's reprojection metric does not separate the methods.
- **Clinical CT-registration claims are not supported.** The positive evidence is synthetic plus
  real-video consistency.

## Limitations
- **Training data:** human CT shapes only; porcine test data (StereoMIS, DePoLL).
- **Synthetic realism:** procedural textures and a simple stereo-noise model, which arm A over-fits.
- **Registration-recall ceiling:** the primary endpoint saturates in the amended setting, so the
  image question rests on the pre-registered secondary metrics.
- **P2I-LReg clouds:** they are affine-like liftings from monocular depth (audit), so reprojection
  uses fitted cameras with a ceiling below 0.9.
- **Real-video pseudo-GT:** it caps success at 0.58–0.84.

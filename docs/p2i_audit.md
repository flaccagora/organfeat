# P2I-LReg audit (2026-10-26; before any descriptor was run on the data)

Source: Kaggle `junezhou001/p2i-lreg` (CC BY-NC 4.0), 72.8 GB, in `data/raw/p2i_lreg/Liver_regis/`.
Scripts: `scripts/preprocess/audit_p2i_lreg.py` (all cases, format only) and
`scripts/sanity/p2i_cloud_probe.py` (one frame per case, cloud convention). Outputs:
`outputs/audit/p2i_lreg_{cases,frames}.csv` and the mask overlay sheets `outputs/audit/p2i_<case>.jpg`.

## Content
- **Cases and frames.** 21 cases and 344 real keyframes (the paper says 21 patients and 346).
  Each case also has 2,000–2,500 Blender synthetic frames with poses; these are not used.
- **Provenance.** Cases 01–06 are the six P2ILF-derived patients (the dataset readme maps them to
  P2ILF patients 1, 3, 6, 8, 10, 7). Cases 07–21 are the authors' own.
  - Some readme entries contain what appear to be patient initials. They are never copied into any
    output: cases are referred to by folder number only.
- **Real frame files.** Each frame has:
  - a labelme export: `img.png` (RGB) and `label.png` (liver mask). The masks follow the liver
    outline well on visual inspection (overlay sheets).
  - per-video intrinsics `CAM_K.yml`
  - a per-frame scale factor (99–197)
  - a dense coloured cloud
- **Image size.** 1080 × 1920 everywhere except case 05 (2160 × 3840). Case 05's cloud has one point
  per 2 × 2 pixels, so the frame is downsampled for use.
- **Preoperative model.** `model/reconstructed_mesh_world_m.obj`, in metres (extent 0.26–0.36 m) →
  converted to mm.
- **Authors' split.** Test frames for cases 01, 02, 13, 17, 18 (fold 1); train frames for the rest.
- **No reference poses or landmarks** for real frames, as stated in the paper.

## Cloud convention
- **Ordering.** One point per pixel, row-major: the colours match `img.png` better in row-major
  than in transposed order.
- **Scale.** The clouds are roughly mm-scaled. The in-mask extent is ~150–320 mm and depth z is
  ~10–110, against CT models of ~260–360 mm.
- **They are not pinhole back-projections with `CAM_K`.**
  - Reprojecting points with `CAM_K` misses their own pixel by 900–4,000 px (median).
  - x is almost exactly linear in u (corr 0.99–1.00), and likewise y in v.
  - x / z correlates less with (u − cx) / fx (0.98), and implies a focal length 1.5–3.3× smaller
    than `CAM_K`.
- **Camera fitted per frame** to the cloud-to-pixel correspondences inside the mask (one frame per
  case):

  | camera model | median residual (px) | p90 residual (px) |
  |---|---|---|
  | affine (8 parameters) | 12–49 | 38–114 |
  | pinhole with free f, cx, cy | 28–540, several degenerate (f ≈ 0) | – |

  The provided reconstructions are affine-like liftings of the image (z from relative monocular
  depth), not metric pinhole clouds.

## Consequences for the protocol (decided before any descriptor ran)
- **Clouds as given.** The clouds are used as given (protocol). Their distortion relative to a
  metric surface is part of the realistic condition; the paper's own pipeline uses them too.
- **Reprojection metric.** The registered CT model is projected with each frame's fitted affine
  camera (`ofeat.data.p2i_lreg.fit_camera`), not with `CAM_K`. Dice and mean boundary distance vs
  the annotated mask follow from that projection. The fit uses only the provided reconstruction,
  no reference.
- **Ceiling per frame.** The fitted camera itself has ~12–49 px error. Each frame therefore also
  gets the Dice of its own cloud, projected with the same camera. This is the reprojection ceiling,
  and Dice gaps smaller than it are not interpretable.
- **Chamfer distance** to the given cloud is reported as consistency only.
- **Split** (`configs/splits/p2i_lreg.yaml`). Dev is 03, 05, 15, 19, for format / scale / camera
  debugging only. The other 17 cases are locked, including the authors' test cases 01, 02, 13, 17
  and 18. Results are reported with and without the P2ILF-derived cases 01–06.
- **Partial cloud for descriptors.** Pixels inside the mask, minus specular pixels (same detector as
  everywhere). Normals are oriented towards the cloud's viewing side (−z), voxel 3 mm.

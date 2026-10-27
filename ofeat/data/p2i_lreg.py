"""P2I-LReg (Self-P2IR, arXiv 2504.15152; Kaggle junezhou001/p2i-lreg, CC BY-NC 4.0).

Real frames per case (data/raw/p2i_lreg/Liver_regis/<case>/real): labelme export (img.png RGB,
label.png liver mask), intrinsics CAM_K.yml, a dense reconstructed cloud per frame (one point per pixel,
row-major; case 05: one per 2 x 2 pixels of its 4K frames), and a per-frame scale factor. Preoperative
model: model/reconstructed_mesh_world_m.obj, in metres (converted to mm here).
Synthetic frames per case (<case>/syn, split lists <case>/{train,test}_syn.txt): rendered visible
liver points (camera frame, mm, no RGB) with exact camera poses; used only for the post-hoc
comparison with Self-P2IR (docs/ext_p2i_syn.md).

Audit findings (docs/p2i_audit.md): the clouds are roughly mm-scaled but are not pinhole
back-projections with CAM_K; x and y are nearly linear in u and v. Reprojection metrics therefore use
the camera fitted to each frame's own cloud-to-pixel correspondences (`fit_camera`). That fit uses
only the given reconstruction, no reference.

Split (configs/splits/p2i_lreg.yaml): `dev` cases load freely; `locked` cases need
OFEAT_UNLOCK_EVAL=p2i_lreg (final evaluation only). Cases 01-06 are P2ILF-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

from ofeat.data.splits import load_split, require_unlocked
from ofeat.mesh import Mesh, data_root


P2ILF_DERIVED = ("01", "02", "03", "04", "05", "06")  # dataset readme: reprocessed P2ILF patients


def root() -> Path:
    return data_root() / "raw" / "p2i_lreg" / "Liver_regis"


def cases(split: str) -> list[str]:
    return load_split("p2i_lreg")[split]


def _guard(case: str) -> None:
    if case not in cases("dev"):
        require_unlocked("p2i_lreg")


def frames(case: str) -> list[str]:
    _guard(case)
    return sorted(p.name for p in (root() / case / "real" / "labels").iterdir() if p.is_dir())


def model(case: str) -> Mesh:
    """Preoperative liver surface, mm."""
    import trimesh

    _guard(case)
    obj = sorted((root() / case / "model").glob("*.obj"))[0]
    m = trimesh.load(obj, process=False, force="mesh")
    return Mesh(np.asarray(m.vertices) * 1000.0, np.asarray(m.faces), mesh_id=f"p2i_lreg/{case}")


@dataclass
class P2IFrame:
    case: str
    frame: str
    rgb: np.ndarray  # (H, W, 3) u8, at the cloud's resolution
    mask: np.ndarray  # (H, W) bool liver
    cloud: np.ndarray  # (H, W, 3) the provided reconstruction (as given)
    K: np.ndarray  # CAM_K at the cloud's resolution (not consistent with the cloud, see module doc)
    scale: float


def load_frame(case: str, frame: str) -> P2IFrame:
    import yaml
    from plyfile import PlyData

    _guard(case)
    d = root() / case / "real"
    lab = d / "labels" / frame
    rgb = cv2.imread(str(lab / "img.png"))[..., ::-1]
    msk = cv2.imread(str(lab / "label.png"), cv2.IMREAD_UNCHANGED)
    mask = (msk.max(-1) if msk.ndim == 3 else msk) > 0
    Kd = yaml.safe_load((d / "CAM_K.yml").read_text())
    K = np.array(Kd[sorted(Kd)[0]][0]["cam_K"], np.float64).reshape(3, 3)
    v = PlyData.read(str(d / "liverPcds" / f"{frame}.ply"))["vertex"].data
    H, W = mask.shape
    if len(v) != H * W:  # case 05: 4K frames, cloud at half resolution
        f = int(round(np.sqrt(H * W / len(v))))
        H, W = H // f, W // f
        if len(v) != H * W:
            raise ValueError(f"{case}/{frame}: cloud size {len(v)} does not match the image")
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        K = np.diag([1 / f, 1 / f, 1.0]) @ K
    cloud = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64).reshape(H, W, 3)
    scale = float((d / "scale" / f"{frame}.txt").read_text().split()[0])
    return P2IFrame(case=case, frame=frame, rgb=np.ascontiguousarray(rgb), mask=mask, cloud=cloud, K=K,
                    scale=scale)


def syn_frames(case: str, split: str = "test") -> list[str]:
    """The authors' synthetic split (<case>/<split>_syn.txt), as frame ids ("01900", ...)."""
    _guard(case)
    return [n.removeprefix("ptsLiverCam") for n in (root() / case / f"{split}_syn.txt").read_text().split()]


@cache
def _syn_poses(case: str) -> dict:
    import yaml

    return yaml.load((root() / case / "syn" / "camPose.yml").read_text(), Loader=yaml.CSafeLoader)


def load_syn(case: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic frame: the rendered visible liver points (N, 3), camera frame, mm (camera at the
    origin, looking down -z), and the 4x4 camera-to-model transform (camPose.yml) into model() mm.
    The renders are of model() itself: the transformed points lie on it (median 0.5 mm on dev case 15)."""
    from plyfile import PlyData

    _guard(case)
    v = PlyData.read(str(root() / case / "syn" / "liverPcds" / f"ptsLiverCam{frame}.ply"))["vertex"].data
    p = _syn_poses(case)[frame][0]
    T = np.eye(4)
    T[:3, :3] = np.asarray(p["cam_R_c2w"], np.float64).reshape(3, 3)
    T[:3, 3] = p["cam_t_c2w"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), T


def fit_camera(fr: P2IFrame, kind: str = "affine", n: int = 50000) -> np.ndarray:
    """(2, 4) affine camera [u, v] = A [x, y, z, 1] fitted to the cloud-to-pixel correspondences
    inside the mask (least squares). Used to project a registered CT model into the image."""
    if kind != "affine":
        raise ValueError(kind)
    v, u = np.nonzero(fr.mask)
    P = fr.cloud[v, u]
    sel = np.random.default_rng(0).choice(len(P), min(n, len(P)), replace=False)
    A = np.c_[P[sel], np.ones(len(sel))]
    return np.stack([np.linalg.lstsq(A, u[sel], rcond=None)[0], np.linalg.lstsq(A, v[sel], rcond=None)[0]])

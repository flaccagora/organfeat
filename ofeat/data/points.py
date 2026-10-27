"""Point clouds for matching: the partial visible surface of a view and the complete CT surface.

Both sides use the same sampling (one point per `voxel_mm` voxel, then capped) and the same normal
estimator (k-NN PCA), so density and normal noise do not differ by modality. Normals face the camera
on the partial side and agree with the mesh normals on the CT side: both are outward.

Frames: a partial cloud is built in its camera frame, the CT cloud in the CT world frame (mm). `gt`
holds each partial point's true surface position in the CT world frame (exact, from the noise-free
depth and the camera pose; labels for the loss and the metrics only, never an input). `to_input`
applies an independent random rigid transform per cloud and centres it, so no canonical / scanner
coordinate reaches a descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from ofeat.features.geom import normals_pca
from ofeat.features.lift import backproject, specular_mask
from ofeat.mesh import REPO, Mesh


@dataclass(frozen=True)
class PointsConfig:
    voxel_mm: float = 3.0
    max_partial: int = 8192
    max_ct: int = 32768
    min_partial: int = 256
    normal_k: int = 16
    ct_draws: int = 400_000  # dense area-uniform draws before voxelisation


def load_config(path: Path = REPO / "configs" / "points.yaml") -> PointsConfig:
    from omegaconf import OmegaConf

    return PointsConfig(**OmegaConf.to_container(OmegaConf.load(path)))


@dataclass
class Cloud:
    xyz: np.ndarray  # (N, 3) float32, mm
    normals: np.ndarray  # (N, 3) unit, outward
    pix: np.ndarray | None = None  # (N, 2) int (u, v) source pixel (partial clouds)
    gt: np.ndarray | None = None  # (N, 3) true position in the CT world frame (partial clouds)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.xyz)


def voxel_pick(x: np.ndarray, voxel: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of one point per occupied voxel (the one nearest the voxel centre), on a grid with a
    random origin, so the sampling is never aligned with a canonical frame."""
    off = rng.uniform(0, voxel, 3)
    if len(x) == 0:
        return np.zeros(0, np.int64)
    g = np.floor((x + off) / voxel).astype(np.int64)
    d = np.linalg.norm(x + off - (g + 0.5) * voxel, axis=1)
    _, inv = np.unique(g, axis=0, return_inverse=True)
    order = np.lexsort((d, inv))  # by voxel, then distance to its centre
    first = np.r_[True, inv[order][1:] != inv[order][:-1]]
    return np.sort(order[first])


def _cap(idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(idx, n, replace=False)) if len(idx) > n else idx


def _normals(x: np.ndarray, k: int, **kw) -> np.ndarray:
    # CUDA only in the main process: DataLoader workers are forked and cannot re-initialise CUDA
    main = torch.utils.data.get_worker_info() is None
    dev = "cuda" if main and torch.cuda.is_available() else "cpu"
    t = {key: torch.as_tensor(v, dtype=torch.float32, device=dev) for key, v in kw.items()}
    return normals_pca(torch.as_tensor(x, dtype=torch.float32, device=dev), k, **t).cpu().numpy()


def noisy_depth(depth: np.ndarray, rng: np.random.Generator, sigma100_mm: float,
                hole_frac: float = 0.0) -> np.ndarray:
    """Stereo-like depth error: Gaussian with sigma = sigma100 (z / 100 mm)^2, plus smooth holes
    covering ~`hole_frac` of the valid pixels."""
    z = depth.astype(np.float64).copy()
    ok = z > 0
    z[ok] += rng.normal(0, 1, ok.sum()) * sigma100_mm * (z[ok] / 100.0) ** 2
    if hole_frac > 0:
        from scipy.ndimage import zoom

        H, W = z.shape
        g = zoom(rng.random((8, 8)), (H / 8, W / 8), order=3)[:H, :W]
        z[ok & (g < np.quantile(g[ok], hole_frac))] = 0
    return z


def partial_cloud(rgb: np.ndarray, depth: np.ndarray, tool: np.ndarray, K: np.ndarray,
                  E: np.ndarray, cfg: PointsConfig, rng: np.random.Generator,
                  depth_obs: np.ndarray | None = None, mask: np.ndarray | None = None,
                  gt_map: np.ndarray | None = None) -> Cloud:
    """Visible surface of one view, camera frame. `depth` is the exact depth (ground truth);
    `depth_obs` (default `depth`) is what the method sees (e.g. with noise or holes). Pixels:
    organ, not tool, not specular, observed depth > 0, and inside `mask` if given. `gt_map`
    (H, W, 3): true CT-frame point per pixel when the view is deformed (else E^-1 backproject)."""
    d_obs = depth if depth_obs is None else depth_obs
    keep = (depth > 0) & (d_obs > 0) & ~tool & ~specular_mask(rgb)
    if mask is not None:
        keep &= mask
    v, u = np.nonzero(keep)
    x = backproject(d_obs, K)[v, u]
    idx = _cap(voxel_pick(x, cfg.voxel_mm, rng), cfg.max_partial, rng)
    x, u, v = x[idx], u[idx], v[idx]
    if gt_map is not None:
        gt = gt_map[v, u]
    else:
        exact_cam = backproject(depth, K)[v, u]
        Einv = np.linalg.inv(E)
        gt = exact_cam @ Einv[:3, :3].T + Einv[:3, 3]
    n = _normals(x, cfg.normal_k, viewpoint=np.zeros(3)) if len(x) >= cfg.normal_k else np.zeros_like(x)
    return Cloud(xyz=x.astype(np.float32), normals=n.astype(np.float32),
                 pix=np.stack([u, v], 1).astype(np.int32), gt=gt.astype(np.float32),
                 meta={"n_visible_px": int(keep.sum())})


def ct_cloud(mesh: Mesh, cfg: PointsConfig, rng: np.random.Generator) -> Cloud:
    """Area-uniform samples of the complete surface at the common voxel density, world frame."""
    v, f = mesh.verts, mesh.faces
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    tri = rng.choice(len(f), cfg.ct_draws, p=area / area.sum())
    b = rng.dirichlet(np.ones(3), cfg.ct_draws)
    x = np.einsum("nk,nkd->nd", b, v[f[tri]])
    nref = np.einsum("nk,nkd->nd", b, mesh.normals[f[tri]])
    idx = _cap(voxel_pick(x, cfg.voxel_mm, rng), cfg.max_ct, rng)
    x, nref = x[idx], nref[idx]
    n = _normals(x, cfg.normal_k, reference=nref)
    return Cloud(xyz=x.astype(np.float32), normals=n.astype(np.float32),
                 meta={"area_mm2": float(area.sum())})


def random_rigid(rng: np.random.Generator) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng.integers(2 ** 31)).as_matrix()
    T[:3, 3] = rng.normal(0, 100.0, 3)
    return T


def to_input(c: Cloud, T: np.ndarray) -> tuple[Cloud, np.ndarray]:
    """Apply rigid T, then centre the cloud. Returns (cloud, the full 4x4 applied)."""
    x = c.xyz @ T[:3, :3].T + T[:3, 3]
    C = np.eye(4)
    C[:3, 3] = -x.mean(0)
    A = C @ T
    return replace(c, xyz=(x + C[:3, 3]).astype(np.float32),
                   normals=(c.normals @ T[:3, :3].T).astype(np.float32)), A

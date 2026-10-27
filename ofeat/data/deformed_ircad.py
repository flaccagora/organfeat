"""Deformed IRCAD test sets (Neri et al., Front. Robot. AI 2025, doi:10.3389/frobt.2025.1702360).

Regenerated with the authors' generator (scripts/external/gen_deformed_ircad.py) at the paper's four
crop ratios: data/processed/deformed_ircad/test_crop{05,10,15,25}.h5, livers 1-2 x 50 pairs each.
Post-hoc external benchmark only (docs/ext_ircad.md); nothing is selected on it.

Per pair, in the generator's normalised coordinates (each axis min-max scaled to [-1, 1]):
  src        (3500, 3) the complete undeformed liver (the "CT" side)
  tgt        (N, 3)    visible crop of the deformed, rigidly moved liver, jittered (sigma 0.01)
  gt_T       (3, 4)    the rigid part, src -> tgt (applied after the lobe deformation)
  corr       (N,)      src index of each tgt point
  tre_src, tre_tgt (80, 3) hidden-region landmarks: undeformed in src / deformed and moved in tgt
  half       (3,)      mm per normalised unit along each axis, (max - min) / 2
  pov        (3,)      the generator's virtual camera, target frame
Millimetres follow the paper: target-frame differences de-normalised per axis (x half). This
reproduces its ground-truth residual TRE at 15 % (5.39 +- 4.91 here vs 5.37 +- 4.82 mm).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ofeat.mesh import data_root

CROPS = (5, 10, 15, 25)


def path(crop: int, split: str = "test") -> Path:
    """split "test": livers 1-2 (the paper's test set); "dev": livers 3-20 (its training livers, other
    seeds), for diagnosis and any selection, so the test pairs are scored once."""
    return data_root() / "processed" / "deformed_ircad" / f"{split}_crop{crop:02d}.h5"


@dataclass
class IrcadPair:
    crop: int
    idx: int
    liver: int
    src: np.ndarray
    src_normals: np.ndarray  # mesh vertex normals, mapped into normalised coordinates
    tgt: np.ndarray
    gt_T: np.ndarray
    corr: np.ndarray
    tre_src: np.ndarray
    tre_tgt: np.ndarray
    half: np.ndarray
    pov: np.ndarray


@cache
def _mesh(liver: int) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    v = PlyData.read(str(data_root() / "raw" / "deformed_ircad" / "livers_ply" / f"liver{liver}.ply"))["vertex"].data
    return (np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64),
            np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64))


def src_normals(liver: int, src: np.ndarray) -> np.ndarray:
    """Vertex normals of the source points (src = normalised mesh vertices). Normalising x' = D x + c
    maps normals by D^-T, i.e. n' ~ n * (max - min) / 2."""
    x, n = _mesh(liver)
    lo, hi = x.min(0), x.max(0)
    d, j = cKDTree(2 * (x - lo) / (hi - lo) - 1).query(src)
    if d.max() > 1e-4:
        raise ValueError(f"liver {liver}: src points are not mesh vertices (max dist {d.max():.2e})")
    m = n[j] * (hi - lo) / 2
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def load(crop: int, split: str = "test") -> list[IrcadPair]:
    import h5py

    with h5py.File(path(crop, split), "r") as f:
        z = {k: f[k][:] for k in f}
    out = []
    for i in range(len(z["gt_T"])):
        c = z["corr_partial"][i].astype(np.int64)  # rows [src index, tgt index]
        corr = np.empty(len(c), np.int64)
        corr[c[:, 1]] = c[:, 0]
        liver = int(z["liver"][i])
        src = z["src_complete"][i].astype(np.float64)
        out.append(IrcadPair(crop=crop, idx=i, liver=liver, src=src, src_normals=src_normals(liver, src),
                             tgt=z["tgt_partial_def"][i].astype(np.float64), gt_T=z["gt_T"][i].astype(np.float64),
                             corr=corr, tre_src=z["TRE_src"][i].astype(np.float64),
                             tre_tgt=z["TRE_tgt"][i].astype(np.float64),
                             half=((z["max"][i] - z["min"][i]) / 2).astype(np.float64),
                             pov=z["pov"][i].astype(np.float64)))
    return out


def paper_metrics(R: np.ndarray, t: np.ndarray, p: IrcadPair) -> dict:
    """The paper's metrics for an estimated src -> tgt rigid transform (normalised units):
    TRE on the hidden landmarks, MAE of the Euler angles (scipy "xyz", degrees, as RPM-Net / OMNet)
    and MAE of the translation components (mm), plus geodesic rotation and translation-norm errors."""
    e = np.linalg.norm((p.tre_src @ R.T + t - p.tre_tgt) * p.half, axis=1)
    Rg, tg = p.gt_T[:, :3], p.gt_T[:, 3]
    eul = np.abs(Rotation.from_matrix(R).as_euler("xyz", degrees=True)
                 - Rotation.from_matrix(Rg).as_euler("xyz", degrees=True))
    dt = (t - tg) * p.half
    cos = (np.trace(R @ Rg.T) - 1) / 2
    return {"tre_mm": float(e.mean()), "tre_sq_mm2": float((e ** 2).mean()), "mae_r_deg": float(eul.mean()),
            "mae_t_mm": float(np.abs(dt).mean()), "rre_gt_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
            "rte_gt_mm": float(np.linalg.norm(dt))}

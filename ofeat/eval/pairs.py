"""Synthetic (partial view, complete CT) pairs from the cached views, in descriptor input frames.

For each liver: one CT cloud (fixed seed per liver), and one partial cloud per cached view. Each side
gets its own random rigid transform (to_input), so a pair carries:
  P, nP     partial points / normals, partial input frame
  G         true positions of P in the CT input frame (labels: loss and metrics only)
  Q, nQ     CT points / normals, CT input frame
  pix, rgb, K   for lifting image features (partial side only)
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ofeat.data import liver_meshes
from ofeat.data.points import Cloud, PointsConfig, ct_cloud, partial_cloud, random_rigid, to_input
from ofeat.mesh import Mesh, data_root


@dataclass
class Pair:
    subject: str
    view: int
    P: np.ndarray
    nP: np.ndarray
    G: np.ndarray
    Q: np.ndarray
    nQ: np.ndarray
    pix: np.ndarray  # (N, 2) u, v
    rgb: np.ndarray  # (H, W, 3) u8
    K: np.ndarray
    meta: dict


def views_path(view_set: str, subject: str) -> Path:
    return data_root() / "processed" / "views" / view_set / f"{subject}.npz"


def _seed(*parts) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def ct_for(subject: str, cfg: PointsConfig) -> Cloud:
    return ct_cloud(liver_meshes.load(subject), cfg, np.random.default_rng(_seed("ct", subject)))


def pairs(view_set: str, subjects: list[str], cfg: PointsConfig, seed: int = 0,
          views: slice | None = None, depth_noise=None) -> Iterator[Pair]:
    """All usable (view, CT) pairs of the given subjects. `depth_noise(depth, rng) -> depth_obs`
    perturbs the observed depth (stress tests); the ground truth stays exact."""
    for s in subjects:
        f = np.load(views_path(view_set, s))
        z = {k: f[k] for k in f.files}  # decompress once per liver
        deformed = "ct_verts" in z
        if deformed:  # CT side = the organ_sim rest-pose mesh the views were deformed from
            ct = ct_cloud(Mesh(z["ct_verts"], z["ct_faces"]), cfg, np.random.default_rng(_seed("ct", s)))
        else:
            ct = ct_for(s, cfg)
        n = len(z["E"])
        for v in range(n)[views or slice(None)]:
            rng = np.random.default_rng(_seed(seed, view_set, s, v))
            depth = z["depth"][v].astype(np.float32)
            d_obs = depth_noise(depth, rng) if depth_noise else None
            gt_map = None
            if deformed:
                o = z["gt_offsets"]
                gt_map = np.zeros(depth.shape + (3,), np.float32)
                gt_map.reshape(-1, 3)[z["gt_idx"][o[v]:o[v + 1]]] = z["gt_xyz"][o[v]:o[v + 1]]
            pc = partial_cloud(z["rgb"][v], depth, z["tool"][v], z["K"], z["E"][v], cfg, rng, d_obs,
                               gt_map=gt_map)
            if len(pc) < cfg.min_partial:
                continue
            p_in, _ = to_input(pc, random_rigid(rng))
            q_in, B = to_input(ct, random_rigid(rng))
            G = pc.gt @ B[:3, :3].T + B[:3, 3]
            yield Pair(subject=s, view=v, P=p_in.xyz, nP=p_in.normals, G=G.astype(np.float32),
                       Q=q_in.xyz, nQ=q_in.normals, pix=pc.pix, rgb=z["rgb"][v], K=z["K"],
                       meta={"vis_frac": float(z["vis_frac"][v]), "family": str(z["family"][v]),
                             "group": int(z["group"][v]), "n_partial": len(pc), "n_ct": len(ct)})

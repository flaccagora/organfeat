"""Training samples from the cached synthetic views (train split only).

One sample = one liver:
  A      a partial view (camera frame, observed depth with optional stereo-like noise)
  B      an overlapping view of the same group (same texture), absent if no partner overlaps
  CT     one of the cached CT samplings of the complete surface (resampling augmentation)
Each cloud gets an independent random rigid transform and is centred (no canonical frame). The true
world positions (labels, never inputs) are carried separately as `world`: A.gt, B.gt and the CT
world points. Images stay uint8; photometric augmentation and DINOv2 run on the GPU (ofeat.train.run).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ofeat.data.points import Cloud, PointsConfig, noisy_depth, partial_cloud, random_rigid, to_input
from ofeat.eval.pairs import views_path
from ofeat.mesh import data_root


def ct_cache_path(subject: str):
    return data_root() / "processed" / "ct_cache" / f"{subject}.npz"


@dataclass(frozen=True)
class AugConfig:
    p_depth_noise: float = 0.5
    sigma100_mm: tuple[float, float] = (0.0, 1.5)
    p_holes: float = 0.3
    hole_frac: tuple[float, float] = (0.0, 0.15)
    min_overlap: float = 0.3
    p_ct_crop: float = 0.0  # S1 diagnosis: crop the CT to a ball around a visible point, so the
    ct_crop_mm: tuple[float, float] = (60.0, 150.0)  # extent of the context is not informative


def _as_dict(c: Cloud, world: np.ndarray, rng: np.random.Generator, rgb: np.ndarray | None = None) -> dict:
    ci, _ = to_input(c, random_rigid(rng))
    d = {"xyz": torch.from_numpy(ci.xyz), "normals": torch.from_numpy(ci.normals),
         "world": torch.from_numpy(np.asarray(world, np.float32))}
    if rgb is not None:
        d["pix"] = torch.from_numpy(c.pix).long()
        d["rgb"] = torch.from_numpy(np.ascontiguousarray(rgb))
    return d


class LiverViews(Dataset):
    def __init__(self, subjects: list[str], pcfg: PointsConfig, aug: AugConfig = AugConfig(),
                 seed: int = 0, view_set: str = "train") -> None:
        self.subjects, self.pcfg, self.aug, self.seed, self.view_set = subjects, pcfg, aug, seed, view_set
        self._calls = 0  # per worker copy: fresh draws every epoch even with persistent workers

    def __len__(self) -> int:
        return len(self.subjects)

    def _partial(self, z: dict, v: int, rng: np.random.Generator) -> Cloud:
        depth = z["depth"][v].astype(np.float32)
        obs = None
        if rng.random() < self.aug.p_depth_noise:
            hole = rng.uniform(*self.aug.hole_frac) if rng.random() < self.aug.p_holes else 0.0
            obs = noisy_depth(depth, rng, rng.uniform(*self.aug.sigma100_mm), hole)
        return partial_cloud(z["rgb"][v], depth, z["tool"][v], z["K"], z["E"][v], self.pcfg, rng, obs)

    def __getitem__(self, i: int) -> dict:
        s = self.subjects[i]
        self._calls += 1
        rng = np.random.default_rng([torch.initial_seed() % 2 ** 32, self._calls, zlib.crc32(s.encode())])
        f = np.load(views_path(self.view_set, s))
        z = {k: f[k] for k in ("rgb", "depth", "tool", "K", "E", "group", "overlap")}
        V = len(z["E"])
        for _ in range(10):
            a = int(rng.integers(V))
            A = self._partial(z, a, rng)
            if len(A) >= self.pcfg.min_partial:
                break
        out = {"subject": s, "A": _as_dict(A, A.gt, rng, z["rgb"][a])}
        partners = np.flatnonzero((z["group"] == z["group"][a]) & (np.arange(V) != a)
                                  & (z["overlap"][a] >= self.aug.min_overlap))
        if len(partners):
            b = int(rng.choice(partners))
            B = self._partial(z, b, rng)
            if len(B) >= self.pcfg.min_partial:
                out["B"] = _as_dict(B, B.gt, rng, z["rgb"][b])
        ct = np.load(ct_cache_path(s))
        k = int(rng.integers(len(ct["xyz"])))
        C = Cloud(xyz=ct["xyz"][k], normals=ct["normals"][k])
        if rng.random() < self.aug.p_ct_crop:
            centre = A.gt[rng.integers(len(A))]
            keep = np.linalg.norm(C.xyz - centre, axis=1) < rng.uniform(*self.aug.ct_crop_mm)
            if keep.sum() >= self.pcfg.min_partial:
                C = Cloud(xyz=C.xyz[keep], normals=C.normals[keep])
        out["CT"] = _as_dict(C, C.xyz, rng)
        return out


def collate(batch: list[dict]) -> list[dict]:
    return batch

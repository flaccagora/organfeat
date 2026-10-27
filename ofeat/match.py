"""The fixed matching pipeline: descriptor mutual nearest neighbours -> RANSAC -> point-to-plane ICP.

Frozen at S0 (configs/match.yaml, docs/protocol.md) before any descriptor is trained; every
descriptor (FPFH, frozen Sonata, concatenations, ours) goes through exactly this code with the same
parameters. Source = partial visible surface, target = complete CT surface; T maps source -> target.
The RANSAC call follows defreg/methods/livermatch.py::_ransac (open3d correspondence RANSAC).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ofeat.mesh import REPO


@dataclass(frozen=True)
class MatchConfig:
    mutual: bool = True
    ransac_n: int = 3
    ransac_dist_mm: float = 7.5
    edge_len: float = 0.9
    max_iter: int = 100_000
    confidence: float = 0.999
    icp_dist_mm: float = 5.0
    icp_iters: int = 50
    seed: int = 0


def load_config(path: Path = REPO / "configs" / "match.yaml") -> MatchConfig:
    from omegaconf import OmegaConf

    return MatchConfig(**OmegaConf.to_container(OmegaConf.load(path)))


@dataclass
class Matches:
    nn: np.ndarray  # (N,) nearest target descriptor of every source point
    src: np.ndarray  # (K,) source index of each (mutual) match
    tgt: np.ndarray  # (K,) target index
    conf: np.ndarray  # (K,) ratio-test confidence 1 - d1/d2 (coverage curves only)


@dataclass
class Registration:
    T_ransac: np.ndarray  # (4, 4)
    T: np.ndarray  # (4, 4) after ICP
    matches: Matches
    n_ransac_inliers: int
    seconds: dict = field(default_factory=dict)


@torch.no_grad()
def match(fs: np.ndarray | torch.Tensor, ft: np.ndarray | torch.Tensor, mutual: bool = True,
          chunk: int = 4096) -> Matches:
    """Cosine nearest neighbours between source (N, D) and target (M, D) descriptors."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.nn.functional.normalize(torch.as_tensor(fs, dtype=torch.float32, device=dev), dim=1)
    b = torch.nn.functional.normalize(torch.as_tensor(ft, dtype=torch.float32, device=dev), dim=1)
    nn_ab, d1, d2 = [], [], []
    for s in range(0, len(a), chunk):
        sim = a[s:s + chunk] @ b.T
        top = sim.topk(min(2, len(b)), dim=1)
        nn_ab.append(top.indices[:, 0])
        d = torch.sqrt(torch.clamp(2 - 2 * top.values, min=0))  # unit-vector distances
        d1.append(d[:, 0])
        d2.append(d[:, -1])
    nn_ab, d1, d2 = torch.cat(nn_ab), torch.cat(d1), torch.cat(d2)
    if mutual:
        nn_ba = torch.cat([(b[s:s + chunk] @ a.T).argmax(1) for s in range(0, len(b), chunk)])
        src = torch.nonzero(nn_ba[nn_ab] == torch.arange(len(a), device=dev)).squeeze(1)
    else:
        src = torch.arange(len(a), device=dev)
    conf = 1 - d1[src] / torch.clamp(d2[src], min=1e-9)
    return Matches(nn=nn_ab.cpu().numpy(), src=src.cpu().numpy(), tgt=nn_ab[src].cpu().numpy(),
                   conf=conf.cpu().numpy())


def ransac(ps: np.ndarray, pt: np.ndarray, src: np.ndarray, tgt: np.ndarray,
           cfg: MatchConfig) -> tuple[np.ndarray, int]:
    """Open3D correspondence RANSAC -> (T (4, 4), number of inlier correspondences)."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    if len(src) < cfg.ransac_n:
        return np.eye(4), 0
    o3d.utility.random.seed(cfg.seed)
    s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(ps, np.float64)))
    t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pt, np.float64)))
    corres = o3d.utility.Vector2iVector(np.stack([src, tgt], 1).astype(np.int32))
    res = reg.registration_ransac_based_on_correspondence(
        s, t, corres, cfg.ransac_dist_mm, reg.TransformationEstimationPointToPoint(False),
        cfg.ransac_n,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(cfg.edge_len),
         reg.CorrespondenceCheckerBasedOnDistance(cfg.ransac_dist_mm)],
        reg.RANSACConvergenceCriteria(cfg.max_iter, cfg.confidence))
    return np.asarray(res.transformation), len(res.correspondence_set)


def icp(ps: np.ndarray, pt: np.ndarray, nt: np.ndarray, T0: np.ndarray, cfg: MatchConfig) -> np.ndarray:
    """Point-to-plane ICP of the source onto the target (with target normals), from T0."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(ps, np.float64)))
    t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pt, np.float64)))
    t.normals = o3d.utility.Vector3dVector(np.asarray(nt, np.float64))
    res = reg.registration_icp(s, t, cfg.icp_dist_mm, T0, reg.TransformationEstimationPointToPlane(),
                               reg.ICPConvergenceCriteria(max_iteration=cfg.icp_iters))
    return np.asarray(res.transformation)


def register(ps: np.ndarray, fs, pt: np.ndarray, ft, nt: np.ndarray,
             cfg: MatchConfig | None = None) -> Registration:
    """The whole fixed pipeline on one (partial, CT) pair."""
    cfg = cfg or load_config()
    sec = {}
    t0 = time.perf_counter()
    m = match(fs, ft, cfg.mutual)
    sec["match"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    T_r, n_in = ransac(ps, pt, m.src, m.tgt, cfg)
    sec["ransac"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    T = icp(ps, pt, nt, T_r, cfg)
    sec["icp"] = time.perf_counter() - t0
    return Registration(T_ransac=T_r, T=T, matches=m, n_ransac_inliers=n_in, seconds=sec)

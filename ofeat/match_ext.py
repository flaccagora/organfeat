"""Post-hoc extension of the fixed pipeline for tiny partials (docs/ext_ircad.md). ofeat/match.py stays
frozen; nothing here is used by the pre-registered study.

Selected on the deformed-IRCAD dev set (livers 3-20; scripts/sanity/ircad_small_patch.py), never on the
test pairs:
  - matches: each partial point's k = 3 nearest CT descriptors (instead of mutual nearest neighbours)
  - RANSAC (torch, 100k hypotheses, the frozen sampling / edge / distance checks and 7.5 mm threshold),
    ranked like Open3D's by geometric overlap of the whole partial, then by match consensus
  - an optional pose prior: hypotheses rotating by more than `max_angle` degrees are rejected
  - then the frozen point-to-plane ICP
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ofeat._vendor.geometry import kabsch
from ofeat.eval.metrics import apply
from ofeat.match import MatchConfig, Matches, Registration, icp, load_config


def _kabsch_batch(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, n, 3) -> R (B, 3, 3), t (B, 3) with b ~ a R^T + t."""
    ca, cb = a.mean(1, keepdim=True), b.mean(1, keepdim=True)
    U, _, Vh = torch.linalg.svd((a - ca).transpose(1, 2) @ (b - cb))
    d = torch.sign(torch.linalg.det(Vh.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], 1))
    R = Vh.transpose(1, 2) @ D @ U.transpose(1, 2)
    return R, (cb - ca @ R.transpose(1, 2)).squeeze(1)


class DistGrid:
    """Distance (mm) from any point to the nearest target point, on a 2 mm grid (for overlap scoring)."""

    def __init__(self, pt: np.ndarray, h: float = 2.0, pad: float = 20.0):
        from scipy.ndimage import distance_transform_edt

        self.lo, self.h = pt.min(0) - pad, h
        shape = np.ceil((pt.max(0) + pad - self.lo) / h).astype(int) + 1
        occ = np.ones(shape, bool)
        occ[tuple(np.round((pt - self.lo) / h).astype(int).T)] = False
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.d = torch.as_tensor(distance_transform_edt(occ) * h, dtype=torch.float32, device=dev)
        self.lo_t = torch.as_tensor(self.lo, dtype=torch.float32, device=dev)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        i = torch.round((x - self.lo_t) / self.h).long()
        s = torch.as_tensor(self.d.shape, device=x.device)
        inside = ((i >= 0) & (i < s)).all(-1)
        i = torch.minimum(torch.clamp(i, min=0), s - 1)
        return torch.where(inside, self.d[i[..., 0], i[..., 1], i[..., 2]], torch.full_like(x[..., 0], 1e3))


@torch.no_grad()
def ransac_torch(ps, pt, src, tgt, cfg, n_hyp: int = 100_000, batch: int = 10_000, max_angle: float | None = None,
                     grid: DistGrid | None = None) -> tuple[np.ndarray, int]:
    """RANSAC over correspondences ranked by consensus (# within cfg.ransac_dist_mm), ties by residual.
    max_angle: reject hypotheses rotating by more than this (deg; a pose prior). grid: rank by geometric
    overlap of all source points instead (as Open3D does), consensus as tie-break."""
    if len(src) < cfg.ransac_n:
        return np.eye(4), 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.as_tensor(ps[src], dtype=torch.float32, device=dev)
    Q = torch.as_tensor(pt[tgt], dtype=torch.float32, device=dev)
    Pall = torch.as_tensor(ps, dtype=torch.float32, device=dev)
    cos_min = np.cos(np.radians(max_angle)) if max_angle is not None else -2.0
    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    tau, sim = cfg.ransac_dist_mm, cfg.edge_len
    best = (-1, np.inf, None, None)
    ij = ([0, 0, 1], [1, 2, 2])
    for _ in range(n_hyp // batch):
        idx = torch.randint(len(P), (batch, 3), generator=g, device=dev)
        a, b = P[idx], Q[idx]
        da = (a[:, ij[0]] - a[:, ij[1]]).norm(dim=-1)
        db = (b[:, ij[0]] - b[:, ij[1]]).norm(dim=-1)
        ok = ((da >= db * sim) & (db >= da * sim)).all(1) & (da.min(1).values > 1e-3)
        if not ok.any():
            continue
        R, t = _kabsch_batch(a[ok].double(), b[ok].double())
        R, t = R.float(), t.float()
        ok2 = ((a[ok] @ R.transpose(1, 2) + t[:, None] - b[ok]).norm(dim=-1) < tau).all(1)
        R, t = R[ok2], t[ok2]
        keep = (R.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2 >= cos_min  # pose prior (rotation angle)
        R, t = R[keep], t[keep]
        if len(R) == 0:
            continue
        r = (P[None] @ R.transpose(1, 2) + t[:, None] - Q[None]).norm(dim=-1)
        inl = r < tau
        cnt = inl.sum(1)
        err = torch.where(inl, r, torch.zeros_like(r)).sum(1)
        if grid is not None:  # geometric overlap of the whole source cloud (fraction within tau), then consensus
            cnt = (grid(Pall[None] @ R.transpose(1, 2) + t[:, None]) < tau).sum(1) * 10_000 + cnt
        k = int(torch.argmax(cnt.double() * 1e6 - err.double()))  # count first, then residual
        if int(cnt[k]) > best[0] or (int(cnt[k]) == best[0] and float(err[k]) < best[1]):
            best = (int(cnt[k]), float(err[k]), R[k].double().cpu().numpy(), t[k].double().cpu().numpy())
    if best[2] is None:
        return np.eye(4), 0
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = best[2], best[3]
    Pn, Qn = ps[src].astype(np.float64), pt[tgt].astype(np.float64)
    for _ in range(3):  # refit on the inliers
        inl = np.linalg.norm(apply(T, Pn) - Qn, axis=1) < tau
        if inl.sum() < 3:
            break
        T_new = kabsch(Pn[inl], Qn[inl])
        if max_angle is not None and np.trace(T_new[:3, :3]) < 1 + 2 * cos_min:
            break  # keep the refit inside the prior
        T = T_new
    return T, best[0]


def knn_matches(fs, ft, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.nn.functional.normalize(torch.as_tensor(fs, dtype=torch.float32, device=dev), dim=1)
    b = torch.nn.functional.normalize(torch.as_tensor(ft, dtype=torch.float32, device=dev), dim=1)
    j = (a @ b.T).topk(k, dim=1).indices.cpu().numpy()
    return np.repeat(np.arange(len(fs)), k), j.reshape(-1)


def register_prior(ps: np.ndarray, fs, pt: np.ndarray, ft, nt: np.ndarray, cfg: MatchConfig | None = None,
                   max_angle: float | None = 75.0, k: int = 3) -> Registration:
    """The extended pipeline on one (partial, CT) pair; same interface as ofeat.match.register."""
    cfg = cfg or load_config()
    sec = {}
    t0 = time.perf_counter()
    src, tgt = knn_matches(fs, ft, k)
    sec["match"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    T_r, n_in = ransac_torch(ps, pt, src, tgt, cfg, max_angle=max_angle, grid=DistGrid(pt))
    sec["ransac"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    T = icp(ps, pt, nt, T_r, cfg)
    sec["icp"] = time.perf_counter() - t0
    nn = tgt[::k]
    m = Matches(nn=nn, src=src, tgt=tgt, conf=np.zeros(len(src)))  # no ratio-test confidence for k-NN matches
    return Registration(T_ransac=T_r, T=T, matches=m, n_ransac_inliers=n_in, seconds=sec)


@torch.no_grad()
def closest_pose_search(ps: np.ndarray, pt: np.ndarray, seeds: list[np.ndarray], step_deg: float = 22.5,
                        max_deg: float = 75.0, iters: int = 20, max_src: int = 1024, max_tgt: int = 4096,
                        seed: int = 0) -> tuple[np.ndarray, dict]:
    """Deformation-tolerant local rigid search (D2; defreg OT-20's `ot_pose_search score="closest"`, here moving
    the partial onto the CT). `seeds[0]` is the initial partial -> CT pose: a cubic rotation grid (spacing
    `step_deg`, inside a ball of `max_deg`) about the partial's centroid at that pose gives the hypotheses; any
    further seeds (e.g. descriptor registrations) are added as they are. Each hypothesis gets `iters` ICP updates
    (every partial point -> its nearest CT point, Kabsch) and is scored by the mean nearest-CT distance; the best
    pose is returned (follow it with the frozen ICP)."""
    from scipy.spatial.transform import Rotation

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    si = rng.choice(len(ps), max_src, replace=False) if len(ps) > max_src else np.arange(len(ps))
    ti = rng.choice(len(pt), max_tgt, replace=False) if len(pt) > max_tgt else np.arange(len(pt))
    n = int(np.floor(max_deg / step_deg + 1e-9))
    r = np.arange(-n, n + 1, dtype=np.float64)
    g = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3) * step_deg
    g = g[np.linalg.norm(g, axis=1) <= max_deg + 1e-9]
    T0 = seeds[0]
    x0 = apply(T0, ps[si])
    c = x0.mean(0)
    hyps = []
    for v in np.radians(g):  # rotate the initial pose about the partial's centroid
        D = np.eye(4)
        D[:3, :3] = Rotation.from_rotvec(v).as_matrix()
        D[:3, 3] = c - D[:3, :3] @ c
        hyps.append(D @ T0)
    hyps += list(seeds[1:])
    H = torch.as_tensor(np.stack(hyps), dtype=torch.float32, device=dev)  # (K, 4, 4)
    xs = torch.as_tensor(ps[si], dtype=torch.float32, device=dev)
    y = torch.as_tensor(pt[ti], dtype=torch.float32, device=dev)
    R, t = H[:, :3, :3], H[:, :3, 3]
    for _ in range(iters):
        x = xs[None] @ R.transpose(1, 2) + t[:, None]
        j = torch.stack([torch.cdist(xk, y).argmin(-1) for xk in x])  # (K, N) nearest CT point
        dR, dt = _kabsch_batch(x.double(), y[j].double())
        dR, dt = dR.float(), dt.float()
        R, t = dR @ R, (dR @ t[..., None])[..., 0] + dt
    x = xs[None] @ R.transpose(1, 2) + t[:, None]
    score = torch.stack([torch.cdist(xk, y).min(-1).values.mean() for xk in x])
    k = int(torch.argmin(score))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R[k].double().cpu().numpy(), t[k].double().cpu().numpy()
    sc = np.sort(score.cpu().numpy())
    return T, {"n_hyp": len(hyps), "best": k, "from_seed": bool(k >= len(g)), "score_mm": float(sc[0]),
               "score_gap_mm": float(sc[1] - sc[0]) if len(sc) > 1 else 0.0}

"""Rotation-invariant per-point geometry of a surface point cloud (partial view or CT samples).

  knn               k nearest neighbours (torch, chunked; CPU or GPU)
  normals_pca       PCA normals over k neighbours, oriented towards a viewpoint (camera) or along
                    reference normals (CT mesh normals)
  curvature         principal curvatures k1 >= k2 (1/mm; positive = convex w.r.t. the outward normal)
                    from a weighted least-squares quadric over the neighbours within r mm, batched
  shape_index / curvedness (Koenderink & van Doorn 1992)
  geom_features     [S_r, log(1 + C_r r)] per radius: the per-point geometric input channels

Same definitions as correspondences/corr/features/geom.py (quadric fit in the normal frame), but
vectorised over points so partial clouds can be processed on the fly.
"""

from __future__ import annotations

import numpy as np
import torch

RADII_MM = (6.0, 15.0)


def _t(x, device=None) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x,
                           dtype=torch.float32, device=device)


def knn(x: torch.Tensor, k: int, query: torch.Tensor | None = None,
        chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Distances and indices (M, k) of the k nearest points of x (N, 3) to query (M, 3)
    (default x itself, including the point)."""
    q = x if query is None else query
    k = min(k, len(x))
    ds, js = [], []
    for s in range(0, len(q), chunk):
        d = torch.cdist(q[s:s + chunk], x)
        dd, jj = d.topk(k, dim=1, largest=False)
        ds.append(dd)
        js.append(jj)
    return torch.cat(ds), torch.cat(js)


def normals_pca(x: torch.Tensor, k: int = 16, viewpoint: torch.Tensor | None = None,
                reference: torch.Tensor | None = None) -> torch.Tensor:
    """(N, 3) unit normals: smallest principal axis of the k neighbours; flipped to face
    `viewpoint` (camera centre, same frame) or to agree with `reference` normals."""
    _, j = knn(x, k)
    nb = x[j] - x[j].mean(1, keepdim=True)
    cov = nb.transpose(1, 2) @ nb
    n = torch.linalg.eigh(cov)[1][..., 0]
    if viewpoint is not None:
        flip = ((viewpoint - x) * n).sum(1) < 0
    elif reference is not None:
        flip = (reference * n).sum(1) < 0
    else:
        flip = ((x - x.mean(0)) * n).sum(1) < 0  # outward from the centroid (closed surfaces only)
    return torch.where(flip[:, None], -n, n)


def curvature(x: torch.Tensor, n: torch.Tensor, r: float, k: int = 96,
              min_nb: int = 8) -> torch.Tensor:
    """(N, 2) principal curvatures [k1, k2] at scale r (neighbours within r among the k nearest;
    at least `min_nb` nearest are always used)."""
    d, j = knn(x, k)
    w = ((d < r) | (torch.arange(d.shape[1], device=x.device)[None] < min_nb)).float()
    rel = x[j] - x[:, None]  # (N, k, 3)
    e3 = n
    a = torch.where((e3[:, :1].abs() < 0.9), torch.tensor([1.0, 0, 0], device=x.device),
                    torch.tensor([0, 1.0, 0], device=x.device))
    e1 = torch.nn.functional.normalize(torch.cross(e3, a, dim=1), dim=1)
    e2 = torch.cross(e3, e1, dim=1)
    u, v, z = (rel * e1[:, None]).sum(-1), (rel * e2[:, None]).sum(-1), (rel * e3[:, None]).sum(-1)
    # z = a u^2 + b u v + c v^2 + d u + e v (through the point), weighted least squares
    A = torch.stack([u * u, u * v, v * v, u, v], -1) * w[..., None]
    zb = (z * w)[..., None]
    reg = 1e-6 * torch.eye(5, device=x.device)
    coef = torch.linalg.solve(A.transpose(1, 2) @ A + reg, A.transpose(1, 2) @ zb)[..., 0]
    H = torch.stack([torch.stack([2 * coef[:, 0], coef[:, 1]], -1),
                     torch.stack([coef[:, 1], 2 * coef[:, 2]], -1)], -2)
    # the outward normal points away from the organ: a convex cap bends towards -e3
    ev = torch.linalg.eigvalsh(-H)  # ascending
    return ev.flip(-1)


def shape_index(k: torch.Tensor) -> torch.Tensor:
    return (2 / np.pi) * torch.atan2(k[:, 0] + k[:, 1], k[:, 0] - k[:, 1] + 1e-12)


def curvedness(k: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((k[:, 0] ** 2 + k[:, 1] ** 2) / 2)


def geom_features(x: torch.Tensor, n: torch.Tensor, radii: tuple[float, ...] = RADII_MM) -> torch.Tensor:
    """(N, 2 len(radii)) [shape index, log(1 + curvedness r)] per radius."""
    cols = []
    for r in radii:
        k = curvature(x, n, r)
        cols += [shape_index(k), torch.log1p(curvedness(k) * r)]
    return torch.stack(cols, 1)

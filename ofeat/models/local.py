"""A strictly local learned descriptor (S1 receptive-field diagnosis; `model.arch: ppf`).

Per point p (outward normal n) and each of its k nearest neighbours q (normal m) within radius r,
the rotation-invariant point-pair feature (Drost et al. 2010; PPF-FoldNet)
    [|d| / r, cos(n, d), cos(m, d), cos(n, m)],  d = q - p
goes through a shared MLP and is max- and mean-pooled over the neighbours, at each of the radii.
The per-point geometric channels and the visual channel (or the learned missing vector) are
concatenated to the pooled code, and a head maps everything to the L2-normalised descriptor.
The receptive field is the largest radius, as for FPFH, so a partial view and the complete CT see
the same context except near the view's border. No reference frame and no rotation augmentation are
needed.
"""

from __future__ import annotations

import torch
from torch import nn

from ofeat.features.geom import knn


def ppf(x: torch.Tensor, n: torch.Tensor, k: int, r: float, spacing: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """-> features (N, k, 4) and validity mask (N, k): k neighbours spread over the ball of radius r
    (enough nearest neighbours to fill the ball at the sampling `spacing`, then every few; the point
    itself is excluded)."""
    k_ball = min(len(x) - 1, max(k, int(1.3 * 3.1416 * r * r / spacing ** 2)))
    dist, j = knn(x, k_ball + 1)
    pick = torch.linspace(1, k_ball, k, device=x.device).round().long().unique()
    dist, j = dist[:, pick], j[:, pick]
    d = x[j] - x[:, None]
    du = d / dist.clamp(min=1e-6)[..., None]
    m = n[j]
    f = torch.stack([dist / r, (n[:, None] * du).sum(-1), (m * du).sum(-1), (n[:, None] * m).sum(-1)], -1)
    return f, dist < r


class PPFDescriptor(nn.Module):
    def __init__(self, geom_dim: int = 4, vis_dim: int = 64, out_dim: int = 32,
                 radii: tuple[float, ...] = (10.0, 25.0), k: int = 64, width: int = 128) -> None:
        super().__init__()
        self.radii, self.k = tuple(radii), k
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, width), nn.GELU(),
                                                nn.Linear(width, width)) for _ in self.radii])
        self.missing = nn.Parameter(torch.zeros(vis_dim))
        self.vis_dim = vis_dim
        self.head = nn.Sequential(nn.Linear(2 * width * len(self.radii) + geom_dim + vis_dim, 256), nn.GELU(),
                                  nn.Linear(256, 256), nn.GELU(), nn.Linear(256, out_dim))

    def _one(self, x, n, g, v, ok):
        codes = []
        for r, mlp in zip(self.radii, self.mlp):
            f, valid = ppf(x, n, self.k, r)
            h = mlp(f)
            neg = torch.finfo(h.dtype).min
            mx = h.masked_fill(~valid[..., None], neg).max(1).values
            mx = torch.where(valid.any(1, keepdim=True), mx, torch.zeros_like(mx))
            mean = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
            codes += [mx, mean]
        if v is None:
            vv = self.missing.expand(len(x), -1)
        else:
            okm = torch.ones(len(x), dtype=torch.bool, device=x.device) if ok is None else ok
            vv = torch.where(okm[:, None], v, self.missing.expand(len(x), -1))
        z = self.head(torch.cat(codes + [g, vv.to(g.dtype)], 1))
        return nn.functional.normalize(z.float(), dim=1)

    def forward(self, xyz, normals, geom, vis=None, vis_ok=None):
        vis = vis or [None] * len(xyz)
        vis_ok = vis_ok or [None] * len(xyz)
        return [self._one(x, n, g, v, o) for x, n, g, v, o in zip(xyz, normals, geom, vis, vis_ok)]

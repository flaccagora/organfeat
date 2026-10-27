"""Sonata / PTv3 point encoder producing per-point features. Copied from defreg/models/sonata_encoder.py.

Input clouds are in model units (data mm * `metric_scale`, default 0.005, so a 200 mm liver is
~1 unit, close to the object scale Sonata was pretrained on with 2 cm voxels). Each cloud is
voxelised at `grid_size`; the encoder output is upcast as in the official demo (concatenate the
last `concat_levels` levels, copy down to the finest), then mapped back to every input point.

Copied into organfeat from correspondences/corr/_vendor/sonata_encoder.py.
"""

from __future__ import annotations

import torch
from torch import nn

from ofeat._vendor import sonata as _sonata

SONATA_FEAT_DIM = {"sonata": 512 + 384 + 192, "sonata_small": None}


class SonataEncoder(nn.Module):
    def __init__(self, name: str = "sonata", pretrained: bool = True, grid_size: float = 0.02,
                 concat_levels: int = 3, weights_dir: str = "data/weights/sonata",
                 freeze: bool = False, init_ckpt: str | None = None) -> None:
        """init_ckpt: encoder exported by an encoder pretraining run ({"net": state})
        loaded over the (Sonata or random) initialisation."""
        super().__init__()
        if pretrained:
            self.net = _sonata.load(name, repo_id="facebook/sonata", download_root=weights_dir)
        else:
            cfg = _sonata.load(name, repo_id="facebook/sonata", download_root=weights_dir,
                               ckpt_only=True)["config"]
            self.net = _sonata.model.PointTransformerV3(**cfg)
        if init_ckpt:
            state = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            self.net.load_state_dict(state["net"] if "net" in state else state)
        self.grid_size = grid_size
        self.concat_levels = concat_levels
        enc = list(self.net.enc_channels) if hasattr(self.net, "enc_channels") else None
        self.out_dim = sum(enc[-concat_levels:]) if enc else SONATA_FEAT_DIM[name]
        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False

    def _voxelize(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (representative point index per voxel (V,), grid coords (V,3), inverse (N,))."""
        g = torch.floor((xyz - xyz.min(0).values) / self.grid_size).long()
        _, inverse = torch.unique(g, dim=0, return_inverse=True)
        n_vox = int(inverse.max()) + 1
        first = torch.full((n_vox,), len(xyz), device=xyz.device, dtype=torch.long)
        first.scatter_reduce_(0, inverse, torch.arange(len(xyz), device=xyz.device), "amin")
        return first, g[first], inverse

    def forward(self, clouds: list[torch.Tensor],
                normals: list[torch.Tensor | None] | None = None,
                extras: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """clouds: B tensors (N_i, 3) in model units -> B tensors (N_i, out_dim).
        extras (organfeat addition): optional per-point inputs (N_i, E), averaged per voxel and passed
        as point["extra"] (consumed by ofeat.models.descriptor.ExtraEmbedding)."""
        coords, grids, feats, inverses, offsets, extra = [], [], [], [], [], []
        total = 0
        for i, xyz in enumerate(clouds):
            first, g, inv = self._voxelize(xyz)
            c = xyz[first]
            nrm = None if normals is None else normals[i]
            nrm = torch.zeros_like(c) if nrm is None else nrm[first]
            feats.append(torch.cat([c, torch.zeros_like(c), nrm], 1))  # coord, color, normal
            if extras is not None:
                e = torch.zeros(len(c), extras[i].shape[1], device=c.device, dtype=extras[i].dtype)
                e.index_add_(0, inv, extras[i])
                cnt = torch.bincount(inv, minlength=len(c)).clamp(min=1)
                extra.append(e / cnt[:, None].to(e.dtype))
            coords.append(c)
            grids.append(g)
            inverses.append(inv)
            total += len(c)
            offsets.append(total)
        point = {"coord": torch.cat(coords), "grid_coord": torch.cat(grids),
                 "feat": torch.cat(feats), "offset": torch.tensor(offsets, device=clouds[0].device)}
        if extras is not None:
            point["extra"] = torch.cat(extra)
        point = self.net(point)
        for _ in range(self.concat_levels - 1):
            parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point:
            parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        out, start = [], 0
        for c, inv in zip(coords, inverses):
            out.append(point.feat[start:start + len(c)][inv])
            start += len(c)
        return out

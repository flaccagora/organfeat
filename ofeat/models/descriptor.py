"""The learned point descriptor: one shared encoder for partial (geometry + image) and CT (geometry
only) clouds.

  encoder   Sonata / PTv3 (ofeat._vendor.sonata_encoder), pretrained init, fine-tuned. Input: model
            units (mm x 0.005), outward normals, grid 0.02 (4 mm voxels).
  extra     per-point [geometric channels (shape index, log curvedness at 6 / 15 mm) || visual
            channel (DINO-PCA-64, or a learned *missing* vector)] -> MLP added to the embedding
            output. Its last layer is zero-initialised, so training starts from plain Sonata.
  head      MLP(1088 -> 256 -> 32), L2-normalised.

Visual-modality dropout replaces the visual channel of whole samples / single points by the missing
vector (`vis_ok` False). CT clouds always use the missing vector: the CT path is exactly the
geometry-only path of the partial side.
"""

from __future__ import annotations

import torch
from torch import nn

from ofeat._vendor.sonata_encoder import SonataEncoder

SONATA_SCALE = 0.005
SONATA_ENC_CHANNELS = (48, 96, 192, 384, 512)  # 512 + 384 + 192 = the 1088-d frozen-Sonata output


class ExtraEmbedding(nn.Module):
    """Wraps Sonata's Embedding: feat = stem(coord, colour, normal) + adapter(extra)."""

    def __init__(self, base: nn.Module, extra_dim: int) -> None:
        super().__init__()
        self.base = base
        c = base.embed_channels
        self.adapter = nn.Sequential(nn.Linear(extra_dim, c), nn.GELU(), nn.Linear(c, c))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, point):
        point = self.base(point)
        point.feat = point.feat + self.adapter(point["extra"]).to(point.feat.dtype)
        return point


class PointDescriptor(nn.Module):
    def __init__(self, geom_dim: int = 4, vis_dim: int = 64, out_dim: int = 32, pretrained: bool = True,
                 weights_dir: str = "data/weights/sonata", use_extra: bool = True, enc_stages: int = 5,
                 patch_size: int | None = None) -> None:
        """enc_stages < 5 keeps only the first encoder stages; patch_size overrides the serialized-
        attention patch (1024 points) of every block. Both limit the receptive field (S1 diagnosis)."""
        super().__init__()
        self.enc = SonataEncoder(pretrained=pretrained, weights_dir=weights_dir)
        if enc_stages < len(SONATA_ENC_CHANNELS):
            from ofeat._vendor.sonata.module import PointSequential

            kept = PointSequential()
            for name, m in list(self.enc.net.enc.named_children())[:enc_stages]:
                kept.add(m, name)
            self.enc.net.enc = kept
            self.enc.concat_levels = min(self.enc.concat_levels, enc_stages)
            self.enc.out_dim = sum(SONATA_ENC_CHANNELS[:enc_stages][-self.enc.concat_levels:])
        if patch_size:
            from ofeat._vendor.sonata.model import SerializedAttention

            for m in self.enc.net.modules():
                if isinstance(m, SerializedAttention):
                    m.patch_size = patch_size
        self.geom_dim, self.vis_dim, self.use_extra = geom_dim, vis_dim, use_extra
        if use_extra:
            self.enc.net.embedding = ExtraEmbedding(self.enc.net.embedding, geom_dim + vis_dim)
        self.missing = nn.Parameter(torch.zeros(vis_dim))
        self.head = nn.Sequential(nn.Linear(self.enc.out_dim, 256), nn.GELU(), nn.Linear(256, out_dim))

    def extra(self, geom: torch.Tensor, vis: torch.Tensor | None, vis_ok: torch.Tensor | None) -> torch.Tensor:
        n = len(geom)
        if vis is None:
            v = self.missing.expand(n, -1)
        else:
            ok = torch.ones(n, dtype=torch.bool, device=geom.device) if vis_ok is None else vis_ok
            v = torch.where(ok[:, None], vis, self.missing.expand(n, -1))
        return torch.cat([geom, v], 1)

    def forward(self, xyz: list[torch.Tensor], normals: list[torch.Tensor], geom: list[torch.Tensor],
                vis: list[torch.Tensor | None] | None = None,
                vis_ok: list[torch.Tensor | None] | None = None) -> list[torch.Tensor]:
        """B clouds in mm (centred) -> B tensors (N_i, out_dim), L2-normalised."""
        vis = vis or [None] * len(xyz)
        vis_ok = vis_ok or [None] * len(xyz)
        extras = [self.extra(g, v, o) for g, v, o in zip(geom, vis, vis_ok)] if self.use_extra else None
        feats = self.enc([x * SONATA_SCALE for x in xyz], normals, extras=extras)
        return [nn.functional.normalize(self.head(f.float()), dim=1) for f in feats]


def build(model_cfg: dict, pretrained: bool, weights_dir: str) -> nn.Module:
    """The descriptor named by a training config's `model` section (arch: sonata | ppf)."""
    if model_cfg.get("arch", "sonata") == "ppf":
        from ofeat.models.local import PPFDescriptor

        return PPFDescriptor(out_dim=model_cfg["out_dim"], radii=tuple(model_cfg.get("ppf_radii", (10.0, 25.0))),
                             k=model_cfg.get("ppf_k", 64))
    return PointDescriptor(out_dim=model_cfg["out_dim"], pretrained=pretrained, weights_dir=weights_dir,
                           enc_stages=model_cfg.get("enc_stages", 5), patch_size=model_cfg.get("patch_size"))

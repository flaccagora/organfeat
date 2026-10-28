"""Descriptor extractor (deliverable): a trained checkpoint -> per-point 32-d descriptors.

    ex = Extractor("outputs/train/<run>/step_010000.pt")
    f = ex(xyz_mm, normals)                          # CT / geometry-only path
    f = ex(xyz_mm, normals, rgb=rgb_u8, pix=uv)      # partial view with its image (if the model uses images)

Normals must be outward (towards the camera for a partial view). Clouds should be sampled as in
configs/points.yaml (one point per 3 mm voxel).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ofeat.features.geom import geom_features
from ofeat.features.lift import lift
from ofeat.models.descriptor import build
from ofeat.mesh import data_root


class VisualChannel:
    """Frozen DINOv2 tokens -> lifted to pixels -> fixed PCA-64 (fitted on train renders)."""

    def __init__(self, pca_path: str | Path = "outputs/pca/dino64.npz", device: str = "cuda") -> None:
        from ofeat.features.dino import DinoPatches

        self.dino = DinoPatches().to(device).eval()
        z = np.load(pca_path)
        self.mean = torch.as_tensor(z["mean"], device=device)
        self.comp = torch.as_tensor(z["components"], device=device)
        self.device = device

    @torch.no_grad()
    def tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb (B, 3, H, W) float in [0, 1] -> (B, h, w, 1024)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.dino(rgb).float()

    @torch.no_grad()
    def project(self, tok: torch.Tensor, pix: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return (lift(tok, pix, size) - self.mean) @ self.comp.T


class Extractor:
    def __init__(self, ckpt: str | Path, device: str = "cuda") -> None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = state["config"]
        m = self.cfg["model"]
        self.model = build(m, pretrained=False, weights_dir=str(data_root() / "weights" / "sonata")).to(device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.radii = tuple(m["geom_radii"])
        self.uses_images = bool(self.cfg["loss"]["use_vis"])
        self.visual = VisualChannel(device=device) if self.uses_images else None
        self.device = device

    @torch.no_grad()
    def __call__(self, xyz: np.ndarray, normals: np.ndarray, rgb: np.ndarray | None = None,
                 pix: np.ndarray | None = None, vis: np.ndarray | None = None,
                 vis_ok: np.ndarray | None = None) -> np.ndarray:
        """Image features either from one frame (rgb + pix) or precomputed per point (vis: (N, 64)
        DINO-PCA-64, e.g. averaged over several views; vis_ok: (N,) False -> the missing vector)."""
        x = torch.as_tensor(xyz, dtype=torch.float32, device=self.device)
        x = x - x.mean(0)
        n = torch.as_tensor(normals, dtype=torch.float32, device=self.device)
        g = geom_features(x, n, self.radii)
        if rgb is not None and self.visual is not None:
            img = torch.as_tensor(rgb, device=self.device).permute(2, 0, 1)[None].float() / 255
            tok = self.visual.tokens(img)[0]
            H, W = rgb.shape[:2]
            vis = self.visual.project(tok, torch.as_tensor(pix, device=self.device), (W, H))
        elif vis is not None and self.visual is not None:
            vis = torch.as_tensor(vis, dtype=torch.float32, device=self.device)
        else:
            vis = None
        ok = None if vis_ok is None or vis is None else torch.as_tensor(vis_ok, dtype=torch.bool, device=self.device)
        torch.manual_seed(0)  # PTv3 draws its serialization-order shuffle from the global RNG
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg["optim"].get("amp", True)):
            z = self.model([x], [n], [g], [vis], [ok])[0]
        return z.float().cpu().numpy()

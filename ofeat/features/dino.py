"""Frozen DINOv2 (ViT-L/14 with registers) patch tokens.

Weights from Hugging Face `facebook/dinov2-with-registers-large`, pre-fetched into $HF_HOME on a
node with internet (compute nodes are offline: HF_HUB_OFFLINE=1).

Copied from correspondences/corr/features/dino.py.
"""

from __future__ import annotations

import torch
from torch import nn

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DinoPatches(nn.Module):
    def __init__(self, name: str = "facebook/dinov2-with-registers-large") -> None:
        super().__init__()
        from transformers import AutoModel

        self.net = AutoModel.from_pretrained(name).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.patch = self.net.config.patch_size
        self.n_reg = getattr(self.net.config, "num_register_tokens", 0)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb (B, 3, H, W) in [0, 1], H and W multiples of the patch size ->
        patch tokens (B, H/p, W/p, C), L2-normalised."""
        B, _, H, W = rgb.shape
        h, w = H // self.patch, W // self.patch
        x = self.net(pixel_values=(rgb - self.mean) / self.std).last_hidden_state
        tok = x[:, 1 + self.n_reg:].reshape(B, h, w, -1)
        return nn.functional.normalize(tok, dim=-1)

    @torch.no_grad()
    def layers(self, rgb: torch.Tensor, which: tuple[int, ...]) -> dict[int, torch.Tensor]:
        """Patch tokens of intermediate transformer blocks (hidden_states[i], 0 = embeddings),
        each (B, H/p, W/p, C) and L2-normalised; the last block is `forward`'s input before the
        final layer norm."""
        B, _, H, W = rgb.shape
        h, w = H // self.patch, W // self.patch
        hs = self.net(pixel_values=(rgb - self.mean) / self.std, output_hidden_states=True).hidden_states
        return {i: nn.functional.normalize(hs[i][:, 1 + self.n_reg:].reshape(B, h, w, -1), dim=-1)
                for i in which}

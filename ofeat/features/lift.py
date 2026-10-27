"""Image -> point lifting shared by synthetic renders and real frames.

  backproject       depth map (mm) -> camera-frame points at pixel centres (u + 0.5, v + 0.5),
                    the convention of both render backends (tests/test_points.py)
  specular_mask     glare pixels (V > 240 and S < 40 on 8-bit HSV), excluded from the visible surface
                    on synthetic and real frames alike (same detector as correspondences' covariates)
  lift              bilinear sample of a (h, w, C) patch-token grid at pixel positions
"""

from __future__ import annotations

import numpy as np
import torch


def pixel_grid(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0:H, 0:W]
    return u, v


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W, 3) camera-frame points; rows with depth 0 are 0."""
    H, W = depth.shape
    u, v = pixel_grid(H, W)
    z = depth.astype(np.float64)
    return np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], -1)


def specular_mask(rgb_u8: np.ndarray, v_min: int = 240, s_max: int = 40, dilate: int = 2) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)).astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, np.ones((2 * dilate + 1,) * 2, np.uint8))
    return m.astype(bool)


def lift(tokens: torch.Tensor, uv: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """tokens (h, w, C) over an image of size (W, H); uv (N, 2) pixel indices (u, v) ->
    (N, C) bilinear samples at the pixel centres (token centres sit at their patch centres)."""
    W, H = size
    g = torch.stack([(uv[:, 0].float() + 0.5) / W * 2 - 1, (uv[:, 1].float() + 0.5) / H * 2 - 1], -1)
    out = torch.nn.functional.grid_sample(tokens.permute(2, 0, 1)[None].float(), g[None, None],
                                          mode="bilinear", padding_mode="border", align_corners=False)
    return out[0, :, 0].T

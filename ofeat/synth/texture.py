"""Random procedural liver-like albedo, redrawn per rendered sample.

Texture is a nuisance here, not a label: every sample draws a fresh texture from its own seed, so
no texture pattern is tied to a vertex id, a subject or a position on the organ (the descriptor
cannot learn point identity from texture). The CT side is never textured.

Families, all 3-D solid textures evaluated at the rest-pose surface point (so the pattern moves
with the tissue under deformation):
  training   fbm      multi-octave random plane waves (mottled parenchyma)
             blobs    Gaussian spots of an accent colour (bruising, fat, haematoma)
             vessels  thin ridges of band-limited noise (subcapsular vessels)
             smooth   one low-frequency octave at low contrast (near-uniform capsule)
  held out   stripes  plane-wave stripes
  (stress)   cells    Voronoi cell borders (lobular pattern)
             flat     uniform albedo (untextured)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRAIN_FAMILIES = ("fbm", "blobs", "vessels", "smooth")
HELD_OUT_FAMILIES = ("stripes", "cells")
FAMILIES = TRAIN_FAMILIES + HELD_OUT_FAMILIES + ("flat",)  # flat: untextured stress setting


@dataclass
class Texture:
    family: str
    base: np.ndarray  # (3,) linear RGB
    accent: np.ndarray  # (3,)
    waves: np.ndarray  # (K, 5): direction xyz, frequency (1/mm), phase
    weights: np.ndarray  # (K,) amplitude per wave
    centres: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # blobs / cells
    radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    contrast: float = 1.0

    def __call__(self, p: np.ndarray) -> np.ndarray:
        """Albedo (M, 3) at rest-pose points p (M, 3), mm."""
        p = np.asarray(p, np.float64)
        if len(p) == 0:
            return np.zeros((0, 3))
        f = self.family
        if f == "flat":
            return np.broadcast_to(self.base, p.shape).copy()
        if f in ("fbm", "smooth", "vessels", "stripes"):
            w = p @ self.waves[:, :3].T * (2 * np.pi * self.waves[:, 3]) + self.waves[:, 4]
            s = np.sin(w) @ self.weights / max(np.abs(self.weights).sum(), 1e-9)  # in [-1, 1]
            if f == "vessels":
                a = np.clip(1 - np.abs(s) * 6.0, 0, 1) ** 2  # thin ridges at the zero crossings
            elif f == "stripes":
                a = (s > 0).astype(float)
            else:
                a = 0.5 + 0.5 * s / max(np.abs(s).max(), 1e-9)
        elif f == "blobs":
            d2 = ((p[:, None] - self.centres[None]) ** 2).sum(-1)
            a = np.exp(-d2 / (2 * self.radii[None] ** 2)).max(1)
        elif f == "cells":
            from scipy.spatial import cKDTree

            d = cKDTree(self.centres).query(p, k=2)[0]
            a = np.clip(1 - (d[:, 1] - d[:, 0]) / 1.5, 0, 1)  # ~1.5 mm wide cell borders
        else:
            raise ValueError(f)
        a = np.clip(self.contrast * a, 0, 1)[:, None]
        return (1 - a) * self.base + a * self.accent


def random_background(rng: np.random.Generator, H: int, W: int) -> np.ndarray:
    """(H, W, 3) linear RGB stand-in for surrounding tissue (fat, bowel, abdominal wall): a smooth
    random blend of 2-3 dim tissue colours."""
    from scipy.ndimage import zoom

    cols = np.array([[0.35, 0.22, 0.05], [0.30, 0.06, 0.05], [0.25, 0.12, 0.10], [0.12, 0.03, 0.02]])
    k = rng.choice(len(cols), int(rng.integers(2, 4)), replace=False)
    # cubic upsampling overshoots [0, 1]: clip before sharpening, keep weights strictly positive
    w = np.stack([np.clip(zoom(rng.random((6, 6)), (H / 6, W / 6), order=3)[:H, :W], 0, 1) for _ in k], -1)
    w = w ** 3 + 1e-3
    w /= w.sum(-1, keepdims=True)
    return (w @ cols[k]) * rng.uniform(0.3, 1.0)


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def random_texture(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray,
                   families: tuple[str, ...] = TRAIN_FAMILIES) -> Texture:
    """A fresh texture over the bounding box [lo, hi] (mm) of the rest-pose organ."""
    family = families[int(rng.integers(len(families)))]
    # liver palette, linear RGB (the renderer applies gamma 1/2.2): dark red-brown base, sRGB ~
    # (0.45-0.75, 0.13-0.33, 0.10-0.28); accent lighter/yellower or darker
    base = np.array([rng.uniform(0.15, 0.50), rng.uniform(0.015, 0.09), rng.uniform(0.01, 0.065)])
    if rng.random() < 0.5:
        accent = base * rng.uniform(1.2, 1.9, 3) + np.array([0.05, 0.05, 0.0]) * rng.random()
    else:
        accent = base * rng.uniform(0.35, 0.75, 3)
    accent = accent.clip(0.02, 0.95)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    k = {"fbm": 24, "smooth": 4, "vessels": 12, "stripes": 1}.get(family, 1)
    waves = np.c_[_unit(rng, k), np.zeros(k), rng.uniform(0, 2 * np.pi, k)]
    weights = np.ones(k)
    if family == "fbm":  # 4 octaves x 6 waves, base frequency 0.02-0.06 / mm
        f0 = rng.uniform(0.02, 0.06)
        octave = np.repeat(np.arange(4), 6)
        waves[:, 3] = f0 * 2.0 ** octave
        weights = 0.5 ** octave
    elif family == "smooth":
        waves[:, 3] = rng.uniform(0.005, 0.02, k)
    elif family == "vessels":
        waves[:, 3] = rng.uniform(0.02, 0.08, k)
    elif family == "stripes":
        waves[:, 3] = rng.uniform(0.08, 0.3)
    tex = Texture(family=family, base=base, accent=accent, waves=waves, weights=weights,
                  contrast={"smooth": rng.uniform(0.1, 0.4)}.get(family, rng.uniform(0.6, 1.0)))
    if family == "blobs":
        n = int(rng.integers(15, 60))
        tex.centres = lo + rng.random((n, 3)) * (hi - lo)
        tex.radii = rng.uniform(3.0, 15.0, n)
    elif family == "cells":
        size = rng.uniform(6.0, 15.0)  # mean cell diameter, mm
        n = int(np.prod((hi - lo) / size).clip(64, 20000))
        tex.centres = lo + rng.random((n, 3)) * (hi - lo)
    return tex

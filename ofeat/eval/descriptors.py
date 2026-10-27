"""Baseline descriptors, all through the same interface as ours:

    describe(name, pair, side) -> (N, D) float32, side "src" (partial) or "tgt" (CT)

  fpfh_r<R>          FPFH at radius R mm (open3d, outward normals)
  sonata             frozen Sonata (PTv3) 1088-d features (mm x 0.005, normals, grid 0.02)
  sonata_pca32       the same, PCA-32 fitted on train CT clouds
  <geom>+dino        concatenation [geom block || DINO-PCA-64 block], each L2-normalised, equal
                     weight; the CT side has no image, its visual block is zero
  learned:<ckpt>[:geo]  a trained ofeat descriptor (ofeat.models.infer.Extractor)
  random             i.i.d. Gaussian descriptors: a null reference for calibrating metrics
DINO-PCA: DINOv2 ViT-L/14-reg last-layer tokens lifted to the partial points, PCA-64 fitted on train
renders (scripts/bench/fit_pca.py). PCA files live in outputs/pca/.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import torch

from ofeat._vendor.geometry import fpfh
from ofeat.eval.pairs import Pair
from ofeat.features.lift import lift

PCA_DIR = Path("outputs/pca")
SONATA_SCALE = 0.005  # model units per mm (Sonata pretrained at object scale ~1 unit for 200 mm)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


@cache
def sonata():
    from ofeat._vendor.sonata_encoder import SonataEncoder

    from ofeat.mesh import data_root

    return SonataEncoder(weights_dir=str(data_root() / "weights" / "sonata"), freeze=True).cuda().eval()


@cache
def dino():
    from ofeat.features.dino import DinoPatches

    return DinoPatches().cuda().eval()


@cache
def pca(name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(PCA_DIR / f"{name}.npz")
    return z["mean"], z["components"]


@torch.no_grad()
def sonata_features(x: np.ndarray, n: np.ndarray) -> np.ndarray:
    xs = torch.as_tensor(x * SONATA_SCALE, dtype=torch.float32, device="cuda")
    ns = torch.as_tensor(n, dtype=torch.float32, device="cuda")
    torch.manual_seed(0)  # PTv3 draws its serialization-order shuffle from the global RNG
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = sonata()([xs], [ns])[0]
    return f.float().cpu().numpy()


@torch.no_grad()
def dino_tokens(rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb[None]).cuda().permute(0, 3, 1, 2).float() / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return dino()(x)[0].float()


def dino_lifted(pair: Pair) -> np.ndarray:
    """(N, 1024) DINOv2 features at the partial points."""
    tok = dino_tokens(pair.rgb)
    H, W = pair.rgb.shape[:2]
    return lift(tok, torch.as_tensor(pair.pix, device=tok.device), (W, H)).cpu().numpy()


def _project(x: np.ndarray, name: str) -> np.ndarray:
    mu, comp = pca(name)
    return (x - mu) @ comp.T


@cache
def extractor(ckpt: str):
    from ofeat.models.infer import Extractor

    return Extractor(ckpt)


def describe(name: str, pair: Pair, side: str) -> np.ndarray:
    """`learned:<ckpt>` uses the partial view's image if the model was trained with images;
    `learned:<ckpt>:geo` forces geometry-only inference (same path as the CT side)."""
    x, n = (pair.P, pair.nP) if side == "src" else (pair.Q, pair.nQ)
    if name == "random":  # null reference: matches carry no information (metric calibration only)
        rng = np.random.default_rng(hash((pair.subject, pair.view, side)) % 2 ** 32)
        return rng.normal(size=(len(x), 32)).astype(np.float32)
    if name.startswith("learned:"):
        ckpt, _, mode = name[len("learned:"):].partition(":")
        ex = extractor(ckpt)
        if side == "src" and mode != "geo" and ex.uses_images:
            return ex(x, n, rgb=pair.rgb, pix=pair.pix)
        return ex(x, n)
    if "+dino" in name:
        geom = _l2(describe(name.replace("+dino", ""), pair, side))
        if side == "src":
            vis = _l2(_project(dino_lifted(pair), "dino64"))
        else:
            vis = np.zeros((len(x), 64), np.float32)
        return np.concatenate([geom, vis], 1).astype(np.float32) / np.sqrt(2)
    if name.startswith("fpfh_r"):
        return fpfh(x, n, float(name[len("fpfh_r"):]))
    if name == "sonata":
        return sonata_features(x, n)
    if name == "sonata_pca32":
        return _project(sonata_features(x, n), "sonata32").astype(np.float32)
    raise ValueError(name)

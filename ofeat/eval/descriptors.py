"""Baseline descriptors, all through the same interface as ours:

    describe(name, pair, side) -> (N, D) float32, side "src" (partial) or "tgt" (CT)

  fpfh_r<R>          FPFH at radius R mm (open3d, outward normals)
  sonata             frozen Sonata (PTv3) 1088-d features (mm x 0.005, normals, grid 0.02)
  sonata_pca32       the same, PCA-32 fitted on train CT clouds
  <geom>+dino        concatenation [geom block || DINO-PCA-64 block], each L2-normalised, equal
                     weight; the CT side has no image, its visual block is zero
  learned:<ckpt>[:geo]  a trained ofeat descriptor (ofeat.models.infer.Extractor)
  random             i.i.d. Gaussian descriptors: a null reference for calibrating metrics
  dinolift           lifting baseline (R4, docs/protocol.md 2026-10-28): DINO-PCA-64 on both sides; the
                     partial lifted from its frame, the CT from 32 flat-albedo renders of its mesh
                     (training camera model), averaged per point (ct_dinolift; cached per liver)
  <geom>+dinolift    [geom block || dinolift block], each L2-normalised, equal weight, on both sides
DINO-PCA: DINOv2 ViT-L/14-reg last-layer tokens lifted to the partial points, PCA-64 fitted on train
renders (scripts/bench/fit_pca.py). PCA files live in outputs/pca/.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import torch

from ofeat._vendor.geometry import fpfh
from ofeat.eval.pairs import Pair, _seed
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


DINOLIFT_VIEWS = 32
DINOLIFT_CACHE = Path("outputs/cache/dinolift")


def ct_dinolift(subject: str) -> np.ndarray:
    """(N_ct, 64) DINO-PCA-64 of the liver's CT cloud (ofeat.eval.pairs.ct_for order), lifted from
    DINOLIFT_VIEWS renders of the untextured mesh (flat albedo, default lighting, no tools or smoke)
    with the training camera model, averaged over the views that see each point (depth test 3 mm);
    unseen points get zeros. Cached in outputs/cache/dinolift/<subject>.npz."""
    import zlib

    f = DINOLIFT_CACHE / f"{subject}.npz"
    if f.exists():
        return np.load(f)["feat"]
    from ofeat.data import liver_meshes
    from ofeat.data.points import load_config as points_config
    from ofeat.eval.pairs import ct_for
    from ofeat.synth.render import Camera, Nuisance, intrinsics, look_at, render
    from ofeat.synth.texture import random_background, random_texture
    from ofeat.synth.views import FRAME_SIZE, ViewConfig, _cone, _unit

    mesh = liver_meshes.load(subject)
    x = ct_for(subject, points_config()).xyz.astype(np.float64)  # world frame, pairs.py order
    rng = np.random.default_rng(zlib.crc32(f"dinolift/{subject}".encode()))
    vcfg, (W, H) = ViewConfig(), FRAME_SIZE
    K, ant = intrinsics(W, H, vcfg.fov), np.array([0.0, 1.0, 0.0])
    tex = random_texture(rng, mesh.verts.min(0), mesh.verts.max(0), ("flat",))
    acc, cnt = np.zeros((len(x), 64)), np.zeros(len(x))
    for _ in range(DINOLIFT_VIEWS):
        d = _cone(rng, ant, vcfg.cone_deg)
        facing = np.flatnonzero(mesh.normals @ d > np.cos(np.radians(60)))
        cand = facing if len(facing) else np.arange(len(mesh.verts))
        focus = mesh.verts[cand[np.argmax(mesh.verts[cand] @ d + rng.normal(0, 15.0, len(cand)))]]
        cam = Camera(K=K, E=look_at(focus + rng.uniform(*vcfg.dist_mm) * d, focus, np.cross(d, _unit(rng))),
                     width=W, height=H)
        fr = render(mesh, tex, cam, Nuisance(), rng, background=random_background(rng, H, W))
        xc = x @ cam.E[:3, :3].T + cam.E[:3, 3]
        z = xc[:, 2]
        uv = xc @ K.T
        u = np.round(uv[:, 0] / np.maximum(z, 1e-6)).astype(int)
        v = np.round(uv[:, 1] / np.maximum(z, 1e-6)).astype(int)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ok[ok] &= np.abs(fr.depth[v[ok], u[ok]] - z[ok]) < 3.0
        if not ok.any():
            continue
        tok = dino_tokens((fr.rgb * 255).round().astype(np.uint8))
        feat = lift(tok, torch.as_tensor(np.stack([u[ok], v[ok]], 1), device=tok.device), (W, H)).cpu().numpy()
        acc[ok] += _project(feat, "dino64")
        cnt[ok] += 1
    seen = cnt > 0
    acc[seen] /= cnt[seen, None]
    DINOLIFT_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, feat=acc.astype(np.float32), seen=seen)
    return acc.astype(np.float32)


def _dinolift(pair: Pair, side: str) -> np.ndarray:
    if side == "src":
        return _l2(_project(dino_lifted(pair), "dino64")).astype(np.float32)
    feat = ct_dinolift(pair.subject)
    if len(feat) != len(pair.Q):
        raise ValueError(f"{pair.subject}: dinolift cache has {len(feat)} CT points, the pair {len(pair.Q)}")
    return _l2(feat).astype(np.float32)


def describe(name: str, pair: Pair, side: str) -> np.ndarray:
    """`learned:<ckpt>` uses the partial view's image if the model was trained with images;
    `learned:<ckpt>:geo` forces geometry-only inference (same path as the CT side)."""
    x, n = (pair.P, pair.nP) if side == "src" else (pair.Q, pair.nQ)
    if name == "random":  # null reference: matches carry no information (metric calibration only)
        rng = np.random.default_rng(_seed(pair.subject, pair.view, side))  # crc32: stable across processes
        return rng.normal(size=(len(x), 32)).astype(np.float32)
    if name == "dinolift":
        return _dinolift(pair, side)
    if name.endswith("+dinolift"):
        geom = _l2(describe(name[: -len("+dinolift")], pair, side))
        return np.concatenate([geom, _dinolift(pair, side)], 1).astype(np.float32) / np.sqrt(2)
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

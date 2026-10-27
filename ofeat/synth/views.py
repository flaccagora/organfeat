"""Synthetic laparoscopic views of TotalSeg livers, cached per split before any training.

Per liver, `n_groups` groups of `group_size` views. A group shares one random texture (redrawn per
group) and one focus region, and its views orbit around it, so views within a group overlap (pairs
for partial <-> partial consistency). Cameras: direction within `cone_deg` of anterior (RAS +y),
random roll, `dist_mm` from the focus point, `fov` horizontal. Nuisances (light, specular, smoke,
tools, noise) are drawn per view. Textures and tool layouts use their own random streams, so stress
sets that change only the texture family or the number of tools keep the same cameras (paired).

Stored per liver (npz, data/processed/views/<set>/<subject>.npz): rgb (V, H, W, 3) u8, depth (V, H, W)
f16 (mm, 0 = background or tool), tool (V, H, W) bool, K (3, 3), E (V, 4, 4), group (V,), family (V,),
vis_frac (V,) fraction of the surface area seen, overlap (V, V) fraction of view i's visible points
within 3 mm of view j's (same group only, else -1).
  Rigid views: the true surface point of a pixel is E^-1 backproject(depth); the CT mesh is the
  TotalSeg PLY.
  Deformed views (`deform_mm`, stress only): organ_sim ARAP; also stored: ct_verts / ct_faces (the
  rest-pose mesh = CT side) and, per view, the rest-pose point of every organ pixel (gt_idx flat pixel
  index, gt_xyz float32).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from ofeat.features.lift import backproject
from ofeat.mesh import Mesh
from ofeat.synth.render import Camera, Nuisance, intrinsics, look_at, render, surface_points
from ofeat.synth.texture import TRAIN_FAMILIES, random_background, random_texture

FRAME_SIZE = (644, 518)  # (W, H): 46 x 37 DINOv2 patches, same as the real frames


@dataclass(frozen=True)
class ViewConfig:
    n_groups: int = 16
    group_size: int = 3
    cone_deg: float = 60.0
    dist_mm: tuple[float, float] = (70.0, 160.0)
    orbit_deg: float = 20.0
    focus_jitter_mm: float = 15.0
    fov: float = 70.0
    min_px: int = 4000  # organ pixels needed for a view (else redraw)
    families: tuple[str, ...] = TRAIN_FAMILIES
    p_tools: float = 0.6
    p_smoke: float = 0.4
    n_tools: int | None = None  # stress: fixed number of tools in every view
    deform_mm: tuple[float, float] | None = None  # stress: ARAP handle displacement range


def _unit(rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=3)
    return x / np.linalg.norm(x)


def _cone(rng: np.random.Generator, axis: np.ndarray, deg: float) -> np.ndarray:
    while True:
        d = _unit(rng)
        if np.degrees(np.arccos(np.clip(d @ axis, -1, 1))) <= deg:
            return d


def _rotate_towards(d: np.ndarray, rng: np.random.Generator, deg: float) -> np.ndarray:
    """A direction within `deg` of d (uniform in angle)."""
    a = np.radians(rng.uniform(0, deg))
    p = np.cross(d, _unit(rng))
    p /= np.linalg.norm(p)
    out = np.cos(a) * d + np.sin(a) * p
    return out / np.linalg.norm(out)


def nuisance(rng: np.random.Generator, cfg: ViewConfig) -> Nuisance:
    n = Nuisance(light=rng.uniform(0.7, 1.3), specular=rng.uniform(0.0, 0.6),
                 shininess=rng.uniform(10, 80), vignette=rng.uniform(0.2, 0.7),
                 smoke=rng.uniform(0, 0.4) if rng.random() < cfg.p_smoke else 0.0,
                 n_tools=int(rng.integers(1, 3)) if rng.random() < cfg.p_tools else 0,
                 noise=rng.uniform(0.005, 0.02))
    if cfg.n_tools is not None:
        n.n_tools = cfg.n_tools
    return n


def _visible_area_frac(mesh: Mesh, tri: np.ndarray) -> float:
    v, f = mesh.verts, mesh.faces
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    seen = np.unique(tri[tri >= 0])
    return float(area[seen].sum() / area.sum())


def _overlap(xa: np.ndarray, xb: np.ndarray, tol: float = 3.0) -> float:
    from scipy.spatial import cKDTree

    if len(xa) == 0 or len(xb) == 0:
        return 0.0
    return float((cKDTree(xb).query(xa, distance_upper_bound=tol)[0] < tol).mean())


def render_liver(mesh: Mesh, cfg: ViewConfig, seed: int, backend: str = "auto") -> dict:
    """All views of one liver (see module docstring for the fields)."""
    rng = np.random.default_rng(seed)
    W, H = FRAME_SIZE
    K = intrinsics(W, H, cfg.fov)
    ant = np.array([0.0, 1.0, 0.0])
    rest, shown, extra = mesh, mesh, {}
    if cfg.deform_mm is not None:
        from ofeat._vendor.organ_sim import SimConfig, simulate

        sim = simulate(mesh.verts, mesh.faces, seed, SimConfig(n_verts=len(mesh.verts), disp_mm=cfg.deform_mm))
        rest, shown = sim.source, sim.deformed
        extra = {"ct_verts": rest.verts, "ct_faces": rest.faces,
                 "deformation_mm": sim.meta["deformation_mm"], "gt_idx": [], "gt_xyz": []}
    lo, hi = rest.verts.min(0), rest.verts.max(0)
    out = {k: [] for k in ("rgb", "depth", "tool", "E", "group", "family", "vis_frac", "nuisance")}
    world_pts = []
    for g in range(cfg.n_groups):
        tex = random_texture(np.random.default_rng([seed, 1, g]), lo, hi, cfg.families)
        d0 = _cone(rng, ant, cfg.cone_deg)
        # focus: an anterior-facing surface point roughly along d0 from the centre
        facing = np.flatnonzero(shown.normals @ d0 > np.cos(np.radians(60)))
        cand = facing if len(facing) else np.arange(len(shown.verts))
        focus = shown.verts[cand[np.argmax(shown.verts[cand] @ d0 + rng.normal(0, 15.0, len(cand)))]]
        n_ok, tries = 0, 0
        while n_ok < cfg.group_size and tries < 20 * cfg.group_size:
            tries += 1
            d = _rotate_towards(d0, rng, cfg.orbit_deg)
            target = focus + rng.normal(0, cfg.focus_jitter_mm / np.sqrt(3), 3)
            eye = target + rng.uniform(*cfg.dist_mm) * d
            up = np.cross(d, _unit(rng))
            cam = Camera(K=K, E=look_at(eye, target, up), width=W, height=H)
            nuis = nuisance(rng, cfg)
            bg = random_background(rng, H, W)
            fr = render(shown, tex, cam, nuis, rng, rest=rest, backend=backend, background=bg,
                        tool_rng=np.random.default_rng([seed, 2, g, tries]))
            if (fr.depth > 0).sum() < cfg.min_px:
                continue
            out["rgb"].append((fr.rgb * 255).round().astype(np.uint8))
            out["depth"].append(fr.depth.astype(np.float16))
            out["tool"].append(fr.tool)
            out["E"].append(cam.E)
            out["group"].append(g)
            out["family"].append(tex.family)
            out["vis_frac"].append(_visible_area_frac(shown, fr.tri))
            out["nuisance"].append(json.dumps(asdict(nuis)))
            if cfg.deform_mm is not None:
                idx = np.flatnonzero(fr.tri.ravel() >= 0)
                gt = surface_points(rest.verts, rest.faces, fr.tri, fr.bary).reshape(-1, 3)[idx]
                extra["gt_idx"].append(idx.astype(np.int32))
                extra["gt_xyz"].append(gt.astype(np.float32))
                world_pts.append(gt[::7])
            else:
                Einv = np.linalg.inv(cam.E)
                x = backproject(fr.depth, K)[fr.depth > 0][::7]
                world_pts.append(x @ Einv[:3, :3].T + Einv[:3, 3])
            n_ok += 1
    V = len(out["E"])
    overlap = -np.ones((V, V), np.float32)
    group = np.array(out["group"])
    for i in range(V):
        for j in range(V):
            if group[i] == group[j]:
                overlap[i, j] = _overlap(world_pts[i], world_pts[j])
    if extra:  # ragged per-view arrays -> concatenated + offsets
        n = np.cumsum([0] + [len(i) for i in extra["gt_idx"]])
        extra["gt_offsets"] = n
        extra["gt_idx"] = np.concatenate(extra["gt_idx"])
        extra["gt_xyz"] = np.concatenate(extra["gt_xyz"])
    return {"rgb": np.stack(out["rgb"]), "depth": np.stack(out["depth"]), "tool": np.stack(out["tool"]),
            "K": K, "E": np.stack(out["E"]), "group": group, "family": np.array(out["family"]),
            "vis_frac": np.array(out["vis_frac"], np.float32), "overlap": overlap,
            "nuisance": np.array(out["nuisance"]), "seed": seed, "config": json.dumps(asdict(cfg)), **extra}

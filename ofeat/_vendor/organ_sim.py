"""Procedural deformation + laparoscope-like view of an organ mesh. Copied from
defreg/generate/organ_sim.py, adapted to return a `SimResult` (ofeat.mesh.Mesh) instead of a defreg
Sample, and to also return where the camera looks (for the renderer in ofeat.synth).

  1. decimate to ~`n_verts` vertices (source = complete organ surface, preoperative model)
  2. ARAP deformation (open3d): `n_anchor` surface patches fixed, `n_handle` patches displaced by
     a random vector of magnitude U(disp_mm) (direction biased along the patch normal)
  3. laparoscopic-like visibility: camera on a random direction within `cam_cone_deg` of the
     anterior axis, hidden-point removal, then a geodesic patch keeping `visible_frac` of the surface
  4. target = `n_tgt` points sampled uniformly on the visible deformed faces + Gaussian noise

Deterministic given (mesh, seed).

Copied into organfeat from correspondences/corr/_vendor/organ_sim.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ofeat.mesh import Mesh


@dataclass
class SimConfig:
    n_verts: int = 4000
    n_tgt: int = 2048
    n_anchor: tuple[int, int] = (2, 4)
    n_handle: tuple[int, int] = (1, 3)
    patch_frac: float = 0.12  # patch radius as a fraction of the organ bbox diagonal
    disp_mm: tuple[float, float] = (5.0, 30.0)
    normal_bias: float = 0.5  # 0 = isotropic direction, 1 = along the patch normal
    arap_iters: int = 50
    anterior: tuple[float, float, float] = (0.0, 1.0, 0.0)  # RAS+: +y is anterior
    cam_cone_deg: float = 60.0
    visible_frac: tuple[float, float] = (0.15, 0.45)
    noise_mm: float = 0.5
    global_mm: float = 0.0  # max amplitude of a smooth whole-organ field (insufflation/gravity-like)
    global_bumps: int = 3  # Gaussian bumps of width global_width * bbox diagonal
    global_width: float = 0.5
    p_view_handle: float = 0.0  # prob. the camera looks at a displaced patch (tool contact)
    facing_test: bool = True


@dataclass
class SimResult:
    source: Mesh  # decimated preoperative surface
    deformed: Mesh  # same connectivity, deformed vertices
    camera: np.ndarray  # (3,) camera centre, mm
    look_at: np.ndarray  # (3,) point the camera looks at (centroid of the visible patch)
    visible: np.ndarray  # (N,) bool, source vertices in the observed patch
    target: np.ndarray  # (n_tgt, 3) noisy samples on the visible deformed surface
    src_of_tgt: np.ndarray  # (n_tgt,) nearest-corner source vertex of each target sample
    meta: dict


def _decimate(verts: np.ndarray, faces: np.ndarray, n: int):
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                  o3d.utility.Vector3iVector(faces))
    if len(verts) > n:
        m = m.simplify_quadric_decimation(int(2 * n))  # ~2 faces per vertex
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    m.remove_non_manifold_edges()
    tri_cluster, counts, _ = m.cluster_connected_triangles()  # keep the largest component
    tri_cluster = np.asarray(tri_cluster)
    if len(counts) > 1:
        m.remove_triangles_by_mask(tri_cluster != int(np.argmax(counts)))
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


def _patch(v: np.ndarray, center: int, radius: float) -> np.ndarray:
    return np.flatnonzero(np.linalg.norm(v - v[center], axis=1) < radius)


def _geodesic_patch(v: np.ndarray, f: np.ndarray, allowed: np.ndarray, seed_vertex: int,
                    k: int) -> np.ndarray:
    """The k allowed vertices geodesically closest to seed_vertex (edges between allowed only)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    ok = np.zeros(len(v), bool)
    ok[allowed] = True
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = e[ok[e].all(1)]
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    g = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(v), len(v))).tocsr()
    dist = dijkstra(g, directed=False, indices=seed_vertex)
    reach = np.flatnonzero(np.isfinite(dist) & ok)
    return reach[np.argsort(dist[reach])[:k]]


def _unit(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def simulate(verts: np.ndarray, faces: np.ndarray, seed: int, cfg: SimConfig | None = None,
             mesh_id: str = "sim") -> SimResult:
    import open3d as o3d

    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    mesh = _decimate(np.asarray(verts, np.float64), np.asarray(faces, np.int64), cfg.n_verts)
    v = np.asarray(mesh.vertices).copy()
    f = np.asarray(mesh.triangles).copy()
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    radius = cfg.patch_frac * diag
    src = Mesh(v, f, mesh_id=mesh_id)
    nrm = src.normals

    # 2. ARAP: handles first, anchors chosen far from all handles (redraw if the solver fails)
    for attempt in range(4):
        try:
            n_h = rng.integers(cfg.n_handle[0], cfg.n_handle[1] + 1)
            n_a = rng.integers(cfg.n_anchor[0], cfg.n_anchor[1] + 1)
            handles = rng.choice(len(v), n_h, replace=False)
            d_to_h = np.min(np.linalg.norm(v[:, None] - v[handles][None], axis=2), axis=1)
            far = np.flatnonzero(d_to_h > 3 * radius)
            anchors = rng.choice(far if len(far) >= n_a else np.argsort(-d_to_h)[: 10 * n_a], n_a,
                                 replace=False)
            ids, pos = [], []
            disp_total = 0.0
            for c in handles:
                p = _patch(v, c, radius)
                direction = cfg.normal_bias * nrm[c] + (1 - cfg.normal_bias) * _unit(rng)[0]
                direction /= np.linalg.norm(direction)
                mag = rng.uniform(*cfg.disp_mm)
                disp_total = max(disp_total, mag)
                ids.append(p)
                pos.append(v[p] + mag * direction)
            for c in anchors:
                p = _patch(v, c, radius)
                ids.append(p)
                pos.append(v[p])
            ids_all = np.concatenate(ids)
            pos_all = np.concatenate(pos)
            # a vertex in two patches: keep the first assignment
            ids_all, first = np.unique(ids_all, return_index=True)
            pos_all = pos_all[first]
            deformed = mesh.deform_as_rigid_as_possible(o3d.utility.IntVector(ids_all.tolist()),
                                                        o3d.utility.Vector3dVector(pos_all),
                                                        max_iter=cfg.arap_iters)
            vd = np.asarray(deformed.vertices).copy()
            if not np.isfinite(vd).all():
                raise RuntimeError(f"{mesh_id}: ARAP diverged")
            break
        except RuntimeError:
            if attempt == 3:
                raise

    # 2b. smooth global field: sum of wide Gaussian bumps with random amplitude/direction
    rng2 = np.random.default_rng(seed + 7919)
    if cfg.global_mm > 0:
        centers = v[rng2.choice(len(v), cfg.global_bumps, replace=False)]
        amp = rng2.uniform(0, cfg.global_mm, cfg.global_bumps)[:, None] * _unit(rng2,
                                                                            cfg.global_bumps)
        w = np.exp(-np.linalg.norm(v[:, None] - centers[None], axis=2) ** 2
                   / (2 * (cfg.global_width * diag) ** 2))
        vd = vd + w @ amp
    view_handle = cfg.p_view_handle > 0 and rng2.random() < cfg.p_view_handle
    dmesh = Mesh(vd, f, mesh_id=mesh_id)

    # 3. visibility from a laparoscope-like camera (redraw until enough surface faces it)
    ant = np.asarray(cfg.anterior, float)
    nd = dmesh.normals
    center = vd.mean(0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vd))
    frac = rng.uniform(*cfg.visible_frac)
    best = np.zeros(0, np.int64)
    best_cam = None
    focus = int(handles[rng2.integers(len(handles))]) if view_handle else None
    for attempt in range(40):  # 20 tries inside the anterior cone, then any direction
        if focus is not None and attempt < 20:  # look at the displaced patch
            d = nd[focus] + 0.3 * _unit(rng2)[0]
            d /= np.linalg.norm(d)
        else:
            d = _unit(rng)[0]
            if attempt < 20 and focus is None and \
                    np.degrees(np.arccos(np.clip(d @ ant, -1, 1))) > cfg.cam_cone_deg:
                continue
        cam = center + 3.0 * diag * d
        _, vis = pcd.hidden_point_removal(cam, 100.0 * diag)
        vis = np.asarray(vis, np.int64)
        to_cam = (cam - vd[vis]) / np.linalg.norm(cam - vd[vis], axis=1, keepdims=True)
        if cfg.facing_test:
            vis = vis[(nd[vis] * to_cam).sum(1) > np.cos(np.radians(80))]  # facing the camera
        if len(vis) > len(best):
            best, best_cam = vis, cam
        if len(vis) >= frac * len(v):
            break
    vis = best
    if len(vis) < 16:
        raise RuntimeError(f"{mesh_id}: no usable camera view")
    k = min(len(vis), max(16, int(frac * len(v))))
    # contiguous patch: geodesic growth over the mesh graph from a camera-facing visible vertex
    if focus is not None:
        start = int(vis[np.argmin(np.linalg.norm(vd[vis] - vd[focus], axis=1))])
    else:
        start = int(rng.choice(vis))
    visible = _geodesic_patch(vd, f, vis, start, k)
    vis_mask = np.zeros(len(v), bool)
    vis_mask[visible] = True

    # 4. target samples on visible deformed faces
    face_vis = vis_mask[f].all(1)
    fv = f[face_vis] if face_vis.any() else f[vis_mask[f].any(1)]
    area = np.linalg.norm(np.cross(vd[fv[:, 1]] - vd[fv[:, 0]], vd[fv[:, 2]] - vd[fv[:, 0]]),
                          axis=1)
    pick = rng.choice(len(fv), cfg.n_tgt, p=area / area.sum())
    bary = rng.dirichlet(np.ones(3), cfg.n_tgt)
    tri = fv[pick]
    tgt = np.einsum("nk,nkd->nd", bary, vd[tri]) + rng.normal(0, cfg.noise_mm, (cfg.n_tgt, 3))
    src_of_tgt = tri[np.arange(cfg.n_tgt), bary.argmax(1)]

    return SimResult(
        source=src, deformed=dmesh, camera=best_cam, look_at=vd[visible].mean(0), visible=vis_mask,
        target=tgt, src_of_tgt=src_of_tgt,
        meta={"seed": seed, "max_handle_disp_mm": disp_total, "view_handle": bool(view_handle),
              "visible_frac": float(vis_mask.mean()), "n_handles": int(n_h), "n_anchors": int(n_a),
              "deformation_mm": float(np.linalg.norm(vd - v, axis=1).mean())},
    )

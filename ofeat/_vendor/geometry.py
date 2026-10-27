"""Geometry helpers. Copied from defreg: core/area.py (vertex_area, knn_area), core/visibility.py
(hpr, camera_mask), methods/rigid.py (kabsch), methods/features.py (_fpfh, adapted to one surface
with known outward normals).

Copied into organfeat from correspondences/corr/_vendor/geometry.py.
"""

from __future__ import annotations

import numpy as np


def knn_area(x: np.ndarray, k: int = 10) -> np.ndarray:
    """Area per point from the local 2-D density: pi r_k^2 / k, r_k the distance to the k-th neighbour."""
    from scipy.spatial import cKDTree

    r = cKDTree(x).query(x, k=min(k, len(x) - 1) + 1)[0][:, -1]
    return np.pi * r ** 2 / k


def vertex_area(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """One third of the area of every incident triangle."""
    a = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    out = np.zeros(len(v))
    np.add.at(out, f.ravel(), np.repeat(a / 3.0, 3))
    return out


def hpr(points: np.ndarray, camera: np.ndarray, gamma: float = 1.8) -> np.ndarray:
    """Indices of `points` visible from `camera`: spherical-flip HPR
    (exponential flip p + 2 (R - |p|)^gamma p/|p|, then convex hull with the viewpoint)."""
    from scipy.spatial import ConvexHull

    p = np.asarray(points, np.float64) - camera
    n = np.linalg.norm(p, axis=1)
    R = n.max()
    flipped = p + 2 * ((R - n) ** gamma)[:, None] * (p / n[:, None])
    hull = ConvexHull(np.vstack([flipped, np.zeros(3)]))
    return np.sort(hull.vertices[hull.vertices < len(points)])


def camera_mask(points: np.ndarray, normals: np.ndarray | None, camera: np.ndarray,
                facing_deg: float = 80.0) -> np.ndarray:
    """Boolean mask of `points` seen from `camera`: open3d hidden-point removal with a radius of
    100 x the cloud's diagonal and, given outward normals, the facing test (normal within
    `facing_deg` of the direction to the camera)."""
    import open3d as o3d

    x = np.asarray(points, np.float64)
    cam = np.asarray(camera, np.float64)
    diag = float(np.linalg.norm(x.max(0) - x.min(0)))
    _, vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x)).hidden_point_removal(cam, 100.0 * diag)
    m = np.zeros(len(x), bool)
    m[np.asarray(vis, np.int64)] = True
    if normals is not None:
        to_cam = (cam - x) / np.maximum(np.linalg.norm(cam - x, axis=1, keepdims=True), 1e-12)
        m &= (normals * to_cam).sum(1) > np.cos(np.radians(facing_deg))
    return m


def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Least-squares rigid T (4x4) with T(P) ~ Q."""
    w = np.ones(len(P)) if w is None else np.asarray(w, np.float64)
    w = w / w.sum()
    cp, cq = w @ P, w @ Q
    H = (P - cp).T @ ((Q - cq) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    T = np.eye(4)
    T[:3, :3] = Vt.T @ D @ U.T
    T[:3, 3] = cq - T[:3, :3] @ cp
    return T


def fpfh(points: np.ndarray, normals: np.ndarray, radius_mm: float) -> np.ndarray:
    """FPFH (Rusu et al., ICRA 2009; Open3D) over `radius_mm`, given oriented normals.
    33-D histograms, L2-normalised. Rotation-invariant."""
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))
    pc.normals = o3d.utility.Vector3dVector(np.asarray(normals, np.float64))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        pc, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_mm, max_nn=100)).data.T
    return (f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)).astype(np.float32)

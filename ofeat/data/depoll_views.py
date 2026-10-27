"""DePoLL partial views: the coloured stereo reconstruction of a state, Poisson-meshed and rendered from
virtual laparoscope views (DePoLL ships no calibration or poses for stereo.avi). LOCKED: every entry
point goes through ofeat.data.depoll (require_unlocked).

Copied from correspondences/scripts/stage5/depoll_eval.py (coloured_mesh, render_colours, views),
adapted to return camera-frame depth for ofeat.data.points.partial_cloud.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ofeat._vendor.io import read_ply
from ofeat.data import depoll
from ofeat.data.liver_probe import FRAME_SIZE
from ofeat.mesh import Mesh
from ofeat.synth.render import Camera, intrinsics, look_at, rasterize

N_VIEWS = 24


def coloured_mesh(state: str) -> tuple[Mesh, np.ndarray]:
    import open3d as o3d
    from plyfile import PlyData

    path = depoll.root() / state / "surfaces" / "video_reconstruction.ply"
    depoll.require_unlocked("depoll")
    pts, nrm, _ = read_ply(path)
    v = PlyData.read(str(path))["vertex"].data
    col = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64) / 255
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    if nrm is not None and np.linalg.norm(nrm, axis=1).min() > 1e-6:
        pc.normals = o3d.utility.Vector3dVector(nrm.astype(np.float64))
    else:  # states 09-13 ship all-zero normals: estimate, orient towards a far point on the thinnest axis
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        c = pts.mean(0).astype(np.float64)
        axis = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
        pc.orient_normals_towards_camera_location(c + 1000.0 * axis)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
    mv = np.asarray(mesh.vertices)
    d, j = cKDTree(pts).query(mv)
    keep = d < 3.0  # drop Poisson extrapolation far from observed points
    m = Mesh(mv, np.asarray(mesh.triangles))
    faces = m.faces[keep[m.faces].all(1)]
    return Mesh(mv, faces), col[j]


def render_colours(mesh: Mesh, colours: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """-> rgb (H, W, 3) u8 (vertex colours, no added shading), depth (H, W) mm (0 = miss)."""
    tri, bary = rasterize(mesh.verts, mesh.faces, cam, "auto")
    hit = tri >= 0
    f = mesh.faces[np.where(hit, tri, 0)]
    b = np.where(hit[..., None], bary, 0.0)
    rgb = np.einsum("hwk,hwkd->hwd", b, colours[f])
    xyz = np.einsum("hwk,hwkd->hwd", b, mesh.verts[f])
    depth = np.where(hit, xyz @ cam.E[2, :3] + cam.E[2, 3], 0.0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth.astype(np.float32)


def views(mesh: Mesh, rng: np.random.Generator, n: int = N_VIEWS) -> list[Camera]:
    W, H = FRAME_SIZE
    K = intrinsics(W, H, 70.0)
    n_mean = mesh.normals.mean(0)
    n_mean /= np.linalg.norm(n_mean)
    cams = []
    while len(cams) < n:
        target = mesh.verts[rng.integers(len(mesh.verts))]
        d = n_mean + 0.4 * rng.normal(size=3)
        d /= np.linalg.norm(d)
        cams.append(Camera(K=K, E=look_at(target + rng.uniform(70, 130) * d, target, np.array([0, 0, 1.0])),
                           width=W, height=H))
    return cams

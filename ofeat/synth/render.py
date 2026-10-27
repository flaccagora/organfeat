"""Laparoscope-like rendering of a textured organ mesh with exact per-pixel ground truth.

Geometry pass (`rasterize`): triangle id + barycentrics per pixel, by nvdiffrast on the GPU or by
open3d raycasting on the CPU (same outputs; tests and login nodes use the CPU path). Everything
after it is shared numpy: rest-pose surface point -> albedo (ofeat.synth.texture), a point light
at the camera (endoscope) with 1/d^2 falloff, Lambert + wet specular, vignetting, smoke, tool
occluders, sensor noise.

Camera convention: OpenCV (x right, y down, z forward), world -> camera extrinsic E (4x4),
intrinsics K (3x3). Ground truth per pixel: triangle id + barycentrics (exact surface point on the
rendered mesh and on its rest pose), nearest-corner vertex id (-1 background or tool), depth z (mm).

Copied from correspondences/corr/synth/render.py. Changes: the albedo is any callable of the
rest-pose point (no motif ids), and the frame also returns tri / bary / rest-pose points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ofeat.mesh import Mesh

Albedo = Callable[[np.ndarray], np.ndarray]  # rest-pose points (M, 3) -> linear RGB (M, 3)


@dataclass
class Camera:
    K: np.ndarray  # (3, 3)
    E: np.ndarray  # (4, 4) world -> camera
    width: int
    height: int

    @property
    def centre(self) -> np.ndarray:
        return -self.E[:3, :3].T @ self.E[:3, 3]


@dataclass
class Frame:
    rgb: np.ndarray  # (H, W, 3) float32 in [0, 1]
    depth: np.ndarray  # (H, W) float32 mm, 0 where no organ (background or tool)
    vid: np.ndarray  # (H, W) int32 nearest-corner vertex, -1 background / tool
    tri: np.ndarray  # (H, W) int32 triangle id, -1 background / tool
    bary: np.ndarray  # (H, W, 3) float32 barycentrics of the triangle's corners
    tool: np.ndarray  # (H, W) bool
    camera: Camera


@dataclass
class Nuisance:
    light: float = 1.0  # light intensity scale
    specular: float = 0.4  # wet-tissue highlight strength
    shininess: float = 40.0
    vignette: float = 0.5
    smoke: float = 0.0  # 0..1 haze density
    n_tools: int = 0
    noise: float = 0.01
    extra: dict = field(default_factory=dict)


def intrinsics(width: int, height: int, fov_deg: float = 70.0) -> np.ndarray:
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World -> camera extrinsic (OpenCV axes) for a camera at `eye` looking at `target`."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ eye
    return E


def surface_points(verts: np.ndarray, faces: np.ndarray, tri: np.ndarray,
                   bary: np.ndarray) -> np.ndarray:
    """Points on the mesh (..., 3) from triangle ids (...) and barycentrics (..., 3); 0 where tri < 0."""
    hit = tri >= 0
    f = faces[np.where(hit, tri, 0)]
    b = np.where(hit[..., None], bary, 0.0)
    return np.einsum("...k,...kd->...d", b, verts[f])


# -- geometry pass --------------------------------------------------------------------------------
def rasterize(verts: np.ndarray, faces: np.ndarray, cam: Camera,
              backend: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """-> tri (H, W) int (-1 = miss), bary (H, W, 3) weights of the triangle's 3 corners."""
    if backend == "auto":
        backend = "nvdiffrast" if _has_cuda_nvdiffrast() else "raycast"
    if backend == "nvdiffrast":
        return _rasterize_nvdiffrast(verts, faces, cam)
    if backend == "raycast":
        return _rasterize_raycast(verts, faces, cam)
    raise ValueError(backend)


def _has_cuda_nvdiffrast() -> bool:
    try:
        import nvdiffrast.torch  # noqa: F401
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _rasterize_raycast(verts, faces, cam):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(verts.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(cam.K), o3d.core.Tensor(cam.E), cam.width, cam.height)
    hit = scene.cast_rays(rays)
    tri = hit["primitive_ids"].numpy().astype(np.int64)
    tri[tri == o3d.t.geometry.RaycastingScene.INVALID_ID] = -1
    uv = hit["primitive_uvs"].numpy()  # point = (1-u-v) v0 + u v1 + v v2
    bary = np.stack([1 - uv[..., 0] - uv[..., 1], uv[..., 0], uv[..., 1]], -1)
    return tri, bary


def _rasterize_nvdiffrast(verts, faces, cam, near: float = 1.0, far: float = 2000.0):
    import nvdiffrast.torch as dr
    import torch

    W, H, K = cam.width, cam.height, cam.K
    # OpenCV camera -> OpenGL clip space (y up, z towards the viewer); pixel centres match K
    P = np.array([
        [2 * K[0, 0] / W, 0, 2 * K[0, 2] / W - 1, 0],
        [0, 2 * K[1, 1] / H, 2 * K[1, 2] / H - 1, 0],
        [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, 1, 0]])
    flip = np.diag([1.0, -1.0, 1.0, 1.0])  # OpenCV y down -> clip y up
    pos = np.c_[verts, np.ones(len(verts))] @ (flip @ P @ cam.E).T
    dev = torch.device("cuda")
    glctx = _GLCTX.get("ctx") or _GLCTX.setdefault("ctx", dr.RasterizeCudaContext(device=dev))
    rast, _ = dr.rasterize(glctx, torch.tensor(pos, dtype=torch.float32, device=dev)[None],
                           torch.tensor(faces, dtype=torch.int32, device=dev), (H, W))
    rast = rast[0].flip(0).cpu().numpy()  # nvdiffrast row 0 = bottom
    tri = rast[..., 3].astype(np.int64) - 1
    u, v = rast[..., 0], rast[..., 1]  # point = u v0 + v v1 + (1-u-v) v2
    return tri, np.stack([u, v, 1 - u - v], -1)


_GLCTX: dict = {}


# -- shading --------------------------------------------------------------------------------------
def render(mesh: Mesh, albedo: Albedo, cam: Camera, nuis: Nuisance, rng: np.random.Generator,
           rest: Mesh | None = None, backend: str = "auto",
           tool_rng: np.random.Generator | None = None, background: np.ndarray | None = None) -> Frame:
    """Render `mesh` with albedo evaluated at the rest-pose surface point (`rest`, same
    connectivity, default `mesh`), so texture moves with the tissue under deformation.
    `tool_rng` (default `rng`) draws the tool layout. `background` (H, W, 3) linear RGB fills the
    pixels the organ does not cover (other tissue; it never yields depth or ground truth)."""
    H, W = cam.height, cam.width
    rest = rest or mesh
    tri, bary = rasterize(mesh.verts, mesh.faces, cam, backend)
    hit = tri >= 0
    f = mesh.faces[np.where(hit, tri, 0)]  # (H, W, 3)
    b = np.where(hit[..., None], bary, 0.0)
    p_def = np.einsum("hwk,hwkd->hwd", b, mesh.verts[f])
    p_rest = np.einsum("hwk,hwkd->hwd", b, rest.verts[f])
    n = np.einsum("hwk,hwkd->hwd", b, mesh.normals[f])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    vid = np.where(hit, f[np.arange(H)[:, None], np.arange(W)[None], b.argmax(-1)], -1)

    alb = albedo(p_rest[hit])
    to_eye = cam.centre - p_def[hit]
    d = np.linalg.norm(to_eye, axis=-1, keepdims=True)
    lv = to_eye / d
    ndl = np.clip((n[hit] * lv).sum(-1, keepdims=True), 0, 1)
    d0 = np.median(d) if len(d) else 1.0
    fall = nuis.light * (d0 / d) ** 2
    spec = nuis.specular * ndl ** nuis.shininess  # light at the eye: half-vector = view direction
    rgb = np.zeros((H, W, 3)) if background is None else np.array(background, np.float64)
    rgb[hit] = fall * (alb * (0.15 + 0.85 * ndl) + spec)

    depth = np.zeros((H, W))
    depth[hit] = (p_def[hit] @ cam.E[2, :3] + cam.E[2, 3])

    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (0.5 * max(H, W)) ** 2
    rgb *= (1 - nuis.vignette * np.clip(r2, 0, 1.5) / 1.5)[..., None]

    tool = _tools(H, W, nuis.n_tools, tool_rng or rng)
    if tool.any():
        g = 0.35 + 0.4 * np.abs(np.sin((xx + yy) * 0.05))[..., None]  # brushed-metal shading
        rgb = np.where(tool[..., None], g * np.array([0.8, 0.82, 0.85]), rgb)
        vid = np.where(tool, -1, vid)
        tri = np.where(tool, -1, tri)
        depth = np.where(tool, 0.0, depth)
    if nuis.smoke > 0:
        haze = nuis.smoke * _fbm(H, W, rng)[..., None]
        rgb = (1 - haze) * rgb + haze * 0.85
    rgb = np.clip(rgb + rng.normal(0, nuis.noise, rgb.shape), 0, 1) ** (1 / 2.2)
    return Frame(rgb=rgb.astype(np.float32), depth=depth.astype(np.float32),
                 vid=vid.astype(np.int32), tri=tri.astype(np.int32), bary=bary.astype(np.float32),
                 tool=tool, camera=cam)


def _tools(H: int, W: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Instrument shafts: thick bands entering from the image border toward the centre."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    m = np.zeros((H, W), bool)
    for _ in range(n):
        a = rng.uniform(0, 2 * np.pi)
        start = np.array([W / 2 + 0.75 * W * np.cos(a), H / 2 + 0.75 * H * np.sin(a)])
        tip = np.array([W / 2, H / 2]) + rng.uniform(-0.25, 0.25, 2) * [W, H]
        d = tip - start
        L = np.linalg.norm(d)
        d /= L
        rel = np.stack([xx - start[0], yy - start[1]], -1)
        t = rel @ d
        perp = np.abs(rel @ np.array([-d[1], d[0]]))
        m |= (t > 0) & (t < L) & (perp < rng.uniform(0.04, 0.08) * W)
    return m


def _fbm(H: int, W: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Smooth [0, 1] noise from upsampled random grids."""
    from scipy.ndimage import zoom

    out = np.zeros((H, W))
    for o in range(octaves):
        s = 2 ** (o + 2)
        g = rng.random((s, s))
        out += zoom(g, (H / s, W / s), order=3)[:H, :W] / 2 ** o
    out -= out.min()
    return out / max(out.max(), 1e-9)

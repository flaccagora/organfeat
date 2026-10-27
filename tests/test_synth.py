"""Renderer ground truth is geometrically consistent; textures are fresh per seed and never tied to
vertex ids. (Render tests copied from correspondences/tests/test_synth.py.)"""

import numpy as np
import pytest

from ofeat.features.lift import backproject
from ofeat.mesh import Mesh
from ofeat.synth.render import Camera, Nuisance, intrinsics, look_at, render, surface_points
from ofeat.synth.texture import FAMILIES, HELD_OUT_FAMILIES, TRAIN_FAMILIES, random_texture

pytest.importorskip("open3d")


def _sphere(subdiv=4, r=60.0):
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=subdiv, radius=r)
    return Mesh(s.vertices, s.faces)


def _cam(eye=(0, 0, 200.0), W=160, H=128):
    return Camera(K=intrinsics(W, H), E=look_at(np.array(eye), np.zeros(3), np.array([0, 1.0, 0])),
                  width=W, height=H)


def _flat(p):
    return np.full((len(p), 3), 0.5)


def _gpu_or_skip(backend):
    if backend == "nvdiffrast":
        torch = pytest.importorskip("torch")
        pytest.importorskip("nvdiffrast")
        if not torch.cuda.is_available():
            pytest.skip("no GPU")


@pytest.mark.parametrize("backend", ["raycast", "nvdiffrast"])
def test_gt_vertex_projects_to_its_pixel(backend):
    _gpu_or_skip(backend)
    m = _sphere()
    cam = _cam()
    fr = render(m, _flat, cam, Nuisance(noise=0), np.random.default_rng(0), backend=backend)
    ys, xs = np.nonzero(fr.vid >= 0)
    assert 0.15 < len(ys) / fr.vid.size < 0.25  # analytic disc: 0.199
    pc = m.verts[fr.vid[ys, xs]] @ cam.E[:3, :3].T + cam.E[:3, 3]
    uv = pc @ cam.K.T
    uv = uv[:, :2] / uv[:, 2:]
    err = np.linalg.norm(uv - np.c_[xs + 0.5, ys + 0.5], axis=1)
    assert np.median(err) < 2.0
    np.testing.assert_allclose(fr.depth[ys, xs], pc[:, 2], atol=3.0)


@pytest.mark.parametrize("backend", ["raycast", "nvdiffrast"])
def test_depth_backprojects_to_the_exact_surface_point(backend):
    """Pixel-centre convention: backproject(depth) equals the rasterised surface point (tri/bary).
    Synthetic ground truth relies on this (views store depth only)."""
    _gpu_or_skip(backend)
    m = _sphere(5)
    cam = _cam(eye=(30.0, 20.0, 180.0))
    fr = render(m, _flat, cam, Nuisance(noise=0), np.random.default_rng(0), backend=backend)
    hit = fr.tri >= 0
    exact = surface_points(m.verts, m.faces, fr.tri, fr.bary)[hit]
    exact_cam = exact @ cam.E[:3, :3].T + cam.E[:3, 3]
    err = np.linalg.norm(backproject(fr.depth, cam.K)[hit] - exact_cam, axis=1)
    assert np.median(err) < 0.05 and np.quantile(err, 0.99) < 0.5


def test_tools_hide_ground_truth():
    m = _sphere()
    fr = render(m, _flat, _cam(), Nuisance(n_tools=2), np.random.default_rng(3), backend="raycast")
    assert fr.tool.any() and (fr.vid[fr.tool] == -1).all() and (fr.depth[fr.tool] == 0).all()


def test_textures_vary_by_seed_and_cover_families():
    lo, hi = -np.full(3, 60.0), np.full(3, 60.0)
    p = np.random.default_rng(0).uniform(-60, 60, (500, 3))
    a = random_texture(np.random.default_rng(1), lo, hi)(p)
    b = random_texture(np.random.default_rng(2), lo, hi)(p)
    assert a.shape == (500, 3) and np.abs(a - b).mean() > 1e-3
    assert set(TRAIN_FAMILIES).isdisjoint(HELD_OUT_FAMILIES)
    for fam in FAMILIES:
        t = random_texture(np.random.default_rng(3), lo, hi, families=(fam,))
        c = t(p)
        assert t.family == fam and np.isfinite(c).all() and c.min() >= 0 and c.max() <= 1


def test_backends_agree():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nvdiffrast")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from ofeat.synth.render import rasterize

    m = _sphere()
    cam = _cam(eye=(30.0, 20.0, 180.0))
    t_ray, b_ray = rasterize(m.verts, m.faces, cam, "raycast")
    t_gpu, b_gpu = rasterize(m.verts, m.faces, cam, "nvdiffrast")
    both = (t_ray >= 0) & (t_gpu >= 0)
    assert ((t_ray >= 0) != (t_gpu >= 0)).mean() < 0.01
    assert (t_ray[both] == t_gpu[both]).mean() > 0.97
    same = both & (t_ray == t_gpu)
    assert np.abs(b_ray[same] - b_gpu[same]).max() < 0.05

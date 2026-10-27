"""Point clouds: common sampling, outward normals, exact ground truth, rigid-invariant FPFH, lifting."""

import numpy as np
import pytest
import torch

from ofeat.data.points import (PointsConfig, ct_cloud, load_config, partial_cloud, random_rigid,
                               to_input, voxel_pick)
from ofeat.features.geom import curvature, normals_pca, shape_index
from ofeat.features.lift import lift
from ofeat.mesh import Mesh

pytest.importorskip("open3d")


def _sphere(subdiv=5, r=60.0):
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=subdiv, radius=r)
    return Mesh(s.vertices, s.faces)


def _view(m, eye=(0.0, 0.0, 200.0)):
    from ofeat.synth.render import Camera, Nuisance, intrinsics, look_at, render

    cam = Camera(K=intrinsics(320, 256), E=look_at(np.array(eye), np.zeros(3), np.array([0, 1.0, 0])),
                 width=320, height=256)
    fr = render(m, lambda p: np.full((len(p), 3), 0.5), cam, Nuisance(noise=0, specular=0),
                np.random.default_rng(0), backend="raycast")
    return (fr.rgb * 255).astype(np.uint8), fr.depth, fr.tool, cam


def test_config_file_matches_defaults():
    assert load_config() == PointsConfig()


def test_voxel_pick_one_per_voxel():
    x = np.random.default_rng(0).uniform(0, 30, (5000, 3))
    idx = voxel_pick(x, 3.0, np.random.default_rng(1))
    off = np.random.default_rng(1).uniform(0, 3.0, 3)  # the grid origin voxel_pick draws first
    g_all = np.floor((x + off) / 3.0).astype(int)
    assert len(np.unique(g_all[idx], axis=0)) == len(idx) == len(np.unique(g_all, axis=0))


def test_partial_cloud_gt_and_normals():
    m = _sphere()
    rgb, depth, tool, cam = _view(m)
    c = partial_cloud(rgb, depth, tool, cam.K, cam.E, PointsConfig(), np.random.default_rng(0))
    assert len(c) > 300
    np.testing.assert_allclose(np.linalg.norm(c.gt, axis=1), 60.0, atol=0.2)  # true points on the sphere
    Einv = np.linalg.inv(cam.E)
    world = c.xyz @ Einv[:3, :3].T + Einv[:3, 3]
    np.testing.assert_allclose(world, c.gt, atol=1e-3)  # no noise: observed = exact
    n_world = c.normals @ Einv[:3, :3].T
    assert ((n_world * c.gt).sum(1) > 0).mean() > 0.99  # outward (facing the camera)


def test_ct_cloud_density_and_outward_normals():
    m = _sphere()
    c = ct_cloud(m, PointsConfig(), np.random.default_rng(0))
    assert ((c.normals * c.xyz).sum(1) > 0).all()
    from scipy.spatial import cKDTree

    nn = cKDTree(c.xyz).query(c.xyz, k=2)[0][:, 1]
    assert 1.8 < np.median(nn) < 3.5  # ~ one point per 3 mm voxel


def test_to_input_centres_and_preserves_geometry():
    m = _sphere(4)
    c = ct_cloud(m, PointsConfig(), np.random.default_rng(0))
    ci, A = to_input(c, random_rigid(np.random.default_rng(1)))
    np.testing.assert_allclose(ci.xyz.mean(0), 0, atol=1e-3)
    np.testing.assert_allclose(c.xyz @ A[:3, :3].T + A[:3, 3], ci.xyz, atol=1e-3)


def test_fpfh_invariant_to_rigid_input_transform():
    from ofeat._vendor.geometry import fpfh

    m = _sphere(4)
    c = ct_cloud(m, PointsConfig(), np.random.default_rng(0))
    ci, _ = to_input(c, random_rigid(np.random.default_rng(1)))
    diff = np.abs(fpfh(c.xyz, c.normals, 15.0) - fpfh(ci.xyz, ci.normals, 15.0))
    assert diff.max() < 1e-2 and (diff > 1e-3).mean() < 1e-4  # float32 ties at the radius only


def test_sphere_curvature_on_points():
    m = _sphere()
    c = ct_cloud(m, PointsConfig(), np.random.default_rng(0))
    x, n = torch.from_numpy(c.xyz), torch.from_numpy(c.normals)
    k = curvature(x, n, 10.0)
    np.testing.assert_allclose(k.median(0).values.numpy(), [1 / 60.0, 1 / 60.0], rtol=0.1)
    assert shape_index(k).median() > 0.9  # convex cap w.r.t. outward normals
    n2 = normals_pca(x, 16, reference=n)
    assert ((n2 * n).sum(1) > 0.99).float().mean() > 0.99


def test_lift_samples_token_at_pixel_centre():
    """Token (i, j) of an 11-px patch grid is centred on pixel (11 j + 5, 11 i + 5); in between,
    lifting is bilinear in the pixel-centre coordinate."""
    h, w = 4, 5
    W, H = 11 * w, 11 * h
    jj, ii = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    tok = torch.stack([jj + 10.0 * ii, -jj.float()], -1).float()  # (h, w, 2), linear in i, j
    assert torch.allclose(lift(tok, torch.tensor([[11 * 1 + 5, 11 * 2 + 5]]), (W, H))[0], tok[2, 1])
    uv = torch.tensor([[13, 20], [30, 9], [40, 33]])  # interior pixels
    expect = (uv[:, 0] + 0.5) / 11 - 0.5 + 10 * ((uv[:, 1] + 0.5) / 11 - 0.5)
    assert torch.allclose(lift(tok, uv, (W, H))[:, 0], expect, atol=1e-5)

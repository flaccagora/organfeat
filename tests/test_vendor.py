"""Copied helpers still behave (copied from correspondences/tests/test_vendor.py)."""

import numpy as np
import pytest

from ofeat._vendor.geometry import hpr, kabsch, vertex_area
from ofeat.mesh import Mesh


def _sphere(subdiv: int = 3, r: float = 50.0) -> Mesh:
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=subdiv, radius=r)
    return Mesh(s.vertices, s.faces)


def test_kabsch_recovers_rigid():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(100, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    Q = P @ q.T + [1.0, 2.0, 3.0]
    T = kabsch(P, Q)
    np.testing.assert_allclose(P @ T[:3, :3].T + T[:3, 3], Q, atol=1e-8)


def test_mesh_normals_outward_and_area():
    m = _sphere()
    assert ((m.verts - m.verts.mean(0)) * m.normals).sum(1).min() > 0
    assert vertex_area(m.verts, m.faces).sum() == pytest.approx(4 * np.pi * 50.0 ** 2, rel=0.02)


def test_hpr_sees_about_half_a_sphere():
    m = _sphere()
    vis = hpr(m.verts, np.array([0.0, 0.0, 500.0]))
    assert 0.3 < len(vis) / len(m.verts) < 0.6
    assert m.verts[vis, 2].min() > -10.0


def test_organ_sim_on_sphere():
    pytest.importorskip("open3d")
    from ofeat._vendor.organ_sim import SimConfig, simulate

    m = _sphere(4)
    r = simulate(m.verts, m.faces, seed=1, cfg=SimConfig(n_verts=1500, n_tgt=256))
    assert r.deformed.verts.shape == r.source.verts.shape
    assert r.target.shape == (256, 3) and r.visible.any()
    assert r.visible[r.src_of_tgt].all()

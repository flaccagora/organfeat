"""The fixed matching pipeline and the protocol metrics on controlled inputs."""

import numpy as np
import pytest

from ofeat.eval.metrics import coverage_curve, pair_metrics, rre_deg
from ofeat.match import MatchConfig, icp, load_config, match, ransac, register

pytest.importorskip("open3d")


def _surface(n=3000, seed=0):
    """A bumpy open surface patch (mm) with normals."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-60, 60, (n, 2))
    z = 8 * np.sin(xy[:, 0] / 15) * np.cos(xy[:, 1] / 20)
    dz = np.stack([8 / 15 * np.cos(xy[:, 0] / 15) * np.cos(xy[:, 1] / 20),
                   -8 / 20 * np.sin(xy[:, 0] / 15) * np.sin(xy[:, 1] / 20)], 1)
    nrm = np.c_[-dz, np.ones(n)]
    return np.c_[xy, z], nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def _rigid(seed=1):
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=seed).as_matrix()
    T[:3, 3] = [30.0, -20.0, 50.0]
    return T


def test_config_file_matches_defaults():
    assert load_config() == MatchConfig()


def test_mutual_nn_identity_descriptors():
    f = np.random.default_rng(0).normal(size=(500, 32))
    m = match(f, f[::-1].copy())
    assert len(m.src) == 500 and (m.tgt == 499 - m.src).all() and (m.nn == 499 - np.arange(500)).all()


def test_ransac_icp_recover_pose_with_90pc_outliers():
    Q, nq = _surface()
    T = _rigid()
    Tinv = np.linalg.inv(T)
    src_idx = np.arange(0, 3000, 3)
    P = Q[src_idx] @ Tinv[:3, :3].T + Tinv[:3, 3]  # source points; T maps them onto Q
    rng = np.random.default_rng(0)
    tgt = src_idx.copy()
    out = rng.random(len(tgt)) < 0.9
    tgt[out] = rng.integers(0, len(Q), out.sum())
    cfg = MatchConfig()
    T_r, n_in = ransac(P, Q, np.arange(len(P)), tgt, cfg)
    assert n_in >= 0.05 * len(P)
    T_f = icp(P, Q, nq, T_r, cfg)
    assert rre_deg(T_f, T) < 1.0 and np.linalg.norm(T_f[:3, 3] - T[:3, 3]) < 2.0


def test_register_and_metrics_on_perfect_descriptors():
    Q, nq = _surface()
    T = _rigid(3)
    Tinv = np.linalg.inv(T)
    src_idx = np.arange(0, 3000, 2)
    P = Q[src_idx] @ Tinv[:3, :3].T + Tinv[:3, 3]
    G = Q[src_idx]
    f = np.random.default_rng(1).normal(size=(3000, 16))
    reg = register(P, f[src_idx], Q, f, nq)
    m = pair_metrics(P, G, Q, reg)
    assert m["point_recall@5"] == 1.0 and m["inlier_ratio@10"] == 1.0 and m["fmr"] == 1.0
    assert m["success"] == 1.0 and m["pose_rmse_icp"] < 0.5 and m["rmse_oracle"] < 1e-6


def test_random_descriptors_fail():
    Q, nq = _surface()
    T = _rigid(4)
    Tinv = np.linalg.inv(T)
    src_idx = np.arange(0, 3000, 2)
    P = Q[src_idx] @ Tinv[:3, :3].T + Tinv[:3, 3]
    rng = np.random.default_rng(2)
    reg = register(P, rng.normal(size=(len(P), 16)), Q, rng.normal(size=(3000, 16)), nq)
    m = pair_metrics(P, Q[src_idx], Q, reg)
    assert m["inlier_ratio@10"] < 0.1


def test_coverage_curve_prefers_confident_inliers():
    ok = np.r_[np.ones(10, bool), np.zeros(90, bool)]
    good = coverage_curve(ok, np.r_[np.ones(10), np.zeros(90)])
    bad = coverage_curve(ok, np.r_[np.zeros(10), np.ones(90)])
    assert good[0] == 1.0 and bad[0] == 0.0 and good[-1] == bad[-1] == pytest.approx(0.1)

"""Frame and unit conventions of the post-hoc external benchmarks (deformed IRCAD, P2I-LReg synthetic)."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ofeat.data.deformed_ircad import IrcadPair, paper_metrics
from ofeat.data.points import random_rigid


def _script(name):
    p = Path(__file__).parents[1] / "scripts" / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    R = Rotation.from_euler("xyz", rng.uniform(0, 45, 3), degrees=True).as_matrix()
    gt = np.c_[R, rng.uniform(-0.5, 0.5, 3)]
    src = rng.uniform(-1, 1, (200, 3))
    tre_src = rng.uniform(-1, 1, (80, 3))
    disp = rng.normal(0, 0.02, (80, 3))  # residual "deformation" at the landmarks
    return IrcadPair(crop=15, idx=0, liver=1, src=src, src_normals=np.tile([0, 0, 1.0], (200, 1)),
                     tgt=src[:50] @ R.T + gt[:, 3], gt_T=gt, corr=np.arange(50), tre_src=tre_src,
                     tre_tgt=tre_src @ R.T + gt[:, 3] + disp, half=np.array([95.0, 80.0, 90.0]), pov=np.zeros(3))


def test_paper_metrics_gt_is_residual_only():
    p = _pair()
    m = paper_metrics(p.gt_T[:, :3], p.gt_T[:, 3], p)
    res = np.linalg.norm((p.tre_tgt - p.tre_src @ p.gt_T[:, :3].T - p.gt_T[:, 3]) * p.half, axis=1)
    assert m["mae_r_deg"] < 1e-9 and m["mae_t_mm"] < 1e-9 and m["rre_gt_deg"] < 1e-4  # arccos near 1
    assert np.isclose(m["tre_mm"], res.mean())


def test_paper_metrics_translation_is_per_axis_mm():
    p = _pair()
    dt = np.array([0.01, 0.0, -0.02])
    m = paper_metrics(p.gt_T[:, :3], p.gt_T[:, 3] + dt, p)
    assert np.isclose(m["mae_t_mm"], np.abs(dt * p.half).mean())
    assert np.isclose(m["rte_gt_mm"], np.linalg.norm(dt * p.half))


def test_ircad_src_to_tgt_recovers_gt_through_input_frames():
    ev = _script("ircad_eval")
    p = _pair(1)
    rng = np.random.default_rng(2)
    A, B, s = random_rigid(rng), random_rigid(rng), 88.0
    # perfect pipeline output: partial input -> CT input, i.e. B . S . gt^-1 . S^-1 . A^-1
    G = np.eye(4)
    G[:3] = p.gt_T
    S = np.diag([s, s, s, 1.0])
    T = B @ S @ np.linalg.inv(G) @ np.linalg.inv(S) @ np.linalg.inv(A)
    R, t = ev.src_to_tgt(T, A, B, s)
    assert np.allclose(R, p.gt_T[:, :3], atol=1e-9) and np.allclose(t, p.gt_T[:, 3], atol=1e-9)
    assert paper_metrics(R, t, p)["mae_r_deg"] < 1e-6


def test_register_prior_recovers_pose_and_respects_the_prior():
    from ofeat.eval.metrics import rre_deg
    from ofeat.match import MatchConfig
    from ofeat.match_ext import register_prior

    rng = np.random.default_rng(0)
    xy = rng.uniform(-60, 60, (1500, 2))
    Q = np.c_[xy, 8 * np.sin(xy[:, 0] / 15) * np.cos(xy[:, 1] / 20)]
    nQ = np.tile([0, 0, 1.0], (len(Q), 1))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [20, 10, 25], degrees=True).as_matrix()  # ~33 deg
    T[:3, 3] = [10.0, -5.0, 20.0]
    sel = rng.choice(len(Q), 200, replace=False)
    P = (Q[sel] - T[:3, 3]) @ T[:3, :3]  # partial = T^-1(CT subset), so T maps partial -> CT
    ft = rng.normal(size=(len(Q), 16))
    fs = ft[sel].copy()
    fs[:120] = rng.normal(size=(120, 16))  # 60 % wrong matches
    cfg = MatchConfig()
    reg = register_prior(P, fs, Q, ft, nQ, cfg, max_angle=75.0)
    assert rre_deg(reg.T_ransac, T) < 1.0
    reg = register_prior(P, fs, Q, ft, nQ, cfg, max_angle=20.0)  # the true pose lies outside this prior
    assert np.degrees(np.arccos(np.clip((np.trace(reg.T_ransac[:3, :3]) - 1) / 2, -1, 1))) <= 20.0 + 1e-6

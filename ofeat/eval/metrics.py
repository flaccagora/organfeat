"""Correspondence and registration metrics (definitions fixed in docs/protocol.md).

Every pair is given as: source (partial) points P (N, 3) in the source input frame, their ground-truth
positions G (N, 3) in the *target (CT) input frame*, and target points Q (M, 3). In the rigid case
G = T_gt(P) exactly; under deformation no single rigid T exists and the oracle rigid transform is the
least-squares fit kabsch(P, G). Registration success is judged on the pose error against that oracle
(RMSE over the source points between T(P) and T_oracle(P)), so deformation and depth noise do not
count as registration failure.
"""

from __future__ import annotations

import numpy as np

from ofeat._vendor.geometry import kabsch
from ofeat.match import Matches, Registration

TAUS_MM = (5.0, 10.0, 20.0)
PRIMARY_TAU_MM = 10.0
SUCCESS_RMSE_MM = 10.0
FMR_IR = 0.05
COVERAGE = np.linspace(0.05, 1.0, 20)


def apply(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ T[:3, :3].T + T[:3, 3]


def correct(G: np.ndarray, Q: np.ndarray, src: np.ndarray, tgt: np.ndarray, tau: float) -> np.ndarray:
    """(K,) bool: the matched target point lies within tau of the source point's true position."""
    return np.linalg.norm(Q[tgt] - G[src], axis=1) < tau


def rre_deg(T: np.ndarray, T_ref: np.ndarray) -> float:
    R = T[:3, :3] @ T_ref[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rte_mm(T: np.ndarray, T_ref: np.ndarray) -> float:
    return float(np.linalg.norm(T[:3, 3] - T_ref[:3, 3]))


def rmse(T: np.ndarray, P: np.ndarray, G: np.ndarray) -> float:
    return float(np.sqrt(((apply(T, P) - G) ** 2).sum(1).mean()))


def coverage_curve(ok: np.ndarray, conf: np.ndarray, grid: np.ndarray = COVERAGE) -> np.ndarray:
    """Inlier ratio of the top-q most confident matches, for each retained fraction q."""
    if len(ok) == 0:
        return np.zeros(len(grid))
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
    return cum[np.clip(np.ceil(grid * len(ok)).astype(int) - 1, 0, len(ok) - 1)]


def pose_metrics(P: np.ndarray, G: np.ndarray, T_ransac: np.ndarray, T_icp: np.ndarray) -> dict:
    """Pose errors against the oracle rigid fit, after RANSAC and after ICP, and success."""
    T_oracle = kabsch(P, G)
    G_oracle = apply(T_oracle, P)
    out = {}
    for name, T in (("ransac", T_ransac), ("icp", T_icp)):
        out[f"rre_{name}"] = rre_deg(T, T_oracle)
        out[f"rte_{name}"] = rte_mm(T, T_oracle)
        out[f"rmse_{name}"] = rmse(T, P, G)  # includes deformation / depth noise
        out[f"pose_rmse_{name}"] = rmse(T, P, G_oracle)  # pose error only (= rmse when rigid, noise-free)
    out["success"] = float(out["pose_rmse_icp"] < SUCCESS_RMSE_MM)
    out["success_rre_rte"] = float(out["rre_icp"] < 10.0 and out["rte_icp"] < 10.0)
    return out


def pair_metrics(P: np.ndarray, G: np.ndarray, Q: np.ndarray, reg: Registration) -> dict:
    """All per-pair numbers of the protocol for one registration."""
    m: Matches = reg.matches
    T_oracle = kabsch(P, G)
    out = {"n_src": len(P), "n_tgt": len(Q), "n_matches": len(m.src),
           "rmse_oracle": rmse(T_oracle, P, G)}
    allpts = np.arange(len(P))
    for tau in TAUS_MM:
        t = int(tau)
        out[f"point_recall@{t}"] = float(correct(G, Q, allpts, m.nn, tau).mean())
        ok = correct(G, Q, m.src, m.tgt, tau)
        out[f"inlier_ratio@{t}"] = float(ok.mean()) if len(ok) else 0.0
    ok = correct(G, Q, m.src, m.tgt, PRIMARY_TAU_MM)
    out["fmr"] = float(out[f"inlier_ratio@{int(PRIMARY_TAU_MM)}"] > FMR_IR)
    curve = coverage_curve(ok, m.conf)
    out["coverage_auc"] = float(curve.mean())
    out["coverage_curve"] = curve.tolist()
    out.update(pose_metrics(P, G, reg.T_ransac, reg.T))
    out["n_ransac_inliers"] = reg.n_ransac_inliers
    out.update({f"sec_{k}": v for k, v in reg.seconds.items()})
    return out

"""Which convention do StereoMIS kinematic poses follow? (P2 training sequences only.)

For frame pairs (i, j) with 5-40 mm of camera motion, back-project frame i's SGBM depth, move it to frame j with the
relative pose under each hypothesis (stored pose = camera-to-world or world-to-camera; translation
in mm or in m), project into j, and compare with j's depth. The right hypothesis gives the smallest
median depth residual. Rectification: X_rect = R1 X_cam (Sequence.rectification).

    sbatch -J sanity_poses -c 8 -t 00:30:00 scripts/leonardo/cpu.sbatch python scripts/sanity/stereomis_poses.py
"""

from __future__ import annotations

import numpy as np

from ofeat.data.stereomis import Sequence
from ofeat.data.stereomis_frames import load_frame, meta
from ofeat.features.lift import backproject

MOTION_MM = (5.0, 40.0)  # camera translation of a test pair if the stored unit is metres
MAX_GAP = 600  # video frames


def moving_pairs(poses: dict, frames: list[int], rng: np.random.Generator, n: int = 15) -> list[tuple[int, int]]:
    """Pairs whose stored translation differs by MOTION_MM / 1000 (the camera is mostly static: fixed
    frame gaps give near-identical views)."""
    fr = np.array([f for f in frames if f in poses])
    t = np.stack([poses[f][:3, 3] for f in fr])
    out = []
    for a in rng.permutation(len(fr)):
        near = np.flatnonzero((fr > fr[a]) & (fr - fr[a] <= MAX_GAP))
        d = np.linalg.norm(t[near] - t[a], axis=1) * 1000
        ok = near[(d >= MOTION_MM[0]) & (d <= MOTION_MM[1])]
        if len(ok):
            out.append((int(fr[a]), int(fr[rng.choice(ok)])))
        if len(out) >= n:
            break
    return out


def residual(fi, fj, T_ji: np.ndarray) -> float:
    """Median |z_pred - z_j| (mm) of frame i's points moved to camera j by T_ji."""
    ok = fi.depth > 0
    x = backproject(fi.depth, fi.K)[ok]
    x = x[np.random.default_rng(0).choice(len(x), min(5000, len(x)), replace=False)]
    y = x @ T_ji[:3, :3].T + T_ji[:3, 3]
    uv = y @ fj.K.T
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    H, W = fj.depth.shape
    inb = (y[:, 2] > 1) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    zj = fj.depth[v[inb].astype(int), u[inb].astype(int)]
    good = zj > 0
    if good.sum() < 100:
        return float("nan")
    return float(np.median(np.abs(y[inb][good, 2] - zj[good])))


def main() -> None:
    rows = []
    for seq in ("P2_0", "P2_1", "P2_2", "P2_3"):
        s = Sequence(seq)
        R1 = np.eye(4)
        R1[:3, :3] = s.rectification["R1"]
        poses = s.poses
        for i, j in moving_pairs(poses, meta(seq)["frames"], np.random.default_rng(0)):
            fi, fj = load_frame(seq, i), load_frame(seq, j)
            res = {"seq": seq, "i": i, "j": j, "motion_mm_if_m": 1000 * float(np.linalg.norm(
                poses[j][:3, 3] - poses[i][:3, 3])), "identity": residual(fi, fj, np.eye(4))}
            for unit, sc in (("mm", 1.0), ("m", 1000.0)):
                Pi, Pj = poses[i].copy(), poses[j].copy()
                Pi[:3, 3] *= sc
                Pj[:3, 3] *= sc
                # c2w: X_w = P X_c  ->  X_cj = Pj^-1 Pi X_ci;  w2c: X_c = P X_w -> X_cj = Pj Pi^-1 X_ci
                for conv, T in (("c2w", np.linalg.inv(Pj) @ Pi), ("w2c", Pj @ np.linalg.inv(Pi))):
                    res[f"{conv}_{unit}"] = residual(fi, fj, R1 @ T @ R1.T)
            rows.append(res)
            print(res, flush=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    print(df.drop(columns=["seq", "i", "j"]).median().round(2).to_string())
    print("pairs where each hypothesis is best:",
          df[["identity", "c2w_mm", "w2c_mm", "c2w_m", "w2c_m"]].idxmin(axis=1).value_counts().to_dict())


if __name__ == "__main__":
    main()

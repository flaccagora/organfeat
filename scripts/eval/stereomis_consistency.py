"""Real RGB/geometry consistency on StereoMIS (no CT): frame-pair matching (docs/protocol.md section 8).

    sbatch -J smis_cons -t 03:00:00 scripts/leonardo/gpu_run.sbatch \
        python scripts/eval/stereomis_consistency.py --split test --desc fpfh_r<R> learned:<ckpt> learned:<ckpt>:geo

For frame pairs (i, j) of one sequence, at most 600 video frames apart, with 5-10, 10-20 or 20-40 mm
of kinematic camera translation (the camera is mostly static, so fixed frame gaps give near-identical
views), both rectified left views give partial liver clouds
(SGBM depth; liver from the frozen DINOv2 probe; dataset tool masks; specular pixels removed).
Pseudo ground truth is the relative kinematic pose T_ji (rectified frame; convention from
scripts/sanity/stereomis_poses.py). It carries breathing motion and kinematic error, so this is
consistency, not accuracy. Metrics are computed over the overlap (points of i within 3 mm of a
point of j under T_ji): point recall@10 / inlier ratio@10 against the pseudo-GT positions, and
RRE / RTE of the registered T vs T_ji.
-> outputs/eval/stereomis/<split>/<desc>.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ofeat.data.liver_probe import load_probe, p_liver, patch_mask
from ofeat.data.points import load_config as points_config, partial_cloud, random_rigid, to_input
from ofeat.data.stereomis import Sequence, sequences
from ofeat.data.stereomis_frames import load_frame, meta
from ofeat.eval.descriptors import describe, dino_tokens
from ofeat.eval.metrics import apply, correct, rre_deg, rte_mm
from ofeat.eval.pairs import Pair
from ofeat.match import icp, load_config, ransac, register

MOTION_BINS_MM = ((5.0, 10.0), (10.0, 20.0), (20.0, 40.0))  # kinematic camera translation of a pair
MAX_GAP = 600  # video frames
SUCCESS_RRE_DEG, SUCCESS_RTE_MM = 15.0, 20.0  # pseudo-GT floor (oracle, P2 smoke): median 6.7 deg / 9.6 mm
POSE_CONVENTION = "c2w_m"  # scripts/sanity/stereomis_poses.py on P2: best on 58 / 60 moving pairs


def relative_pose(seq: Sequence, i: int, j: int) -> np.ndarray:
    """Rectified camera i -> camera j."""
    conv, unit = POSE_CONVENTION.split("_")
    sc = 1.0 if unit == "mm" else 1000.0
    Pi, Pj = seq.poses[i].copy(), seq.poses[j].copy()
    Pi[:3, 3] *= sc
    Pj[:3, 3] *= sc
    T = np.linalg.inv(Pj) @ Pi if conv == "c2w" else Pj @ np.linalg.inv(Pi)
    R1 = np.eye(4)
    R1[:3, :3] = seq.rectification["R1"]
    return R1 @ T @ R1.T


def liver_cloud(fr, probe, pcfg, rng):
    tok = dino_tokens(fr.rgb).cpu().numpy()
    h, w = tok.shape[:2]
    tool_frac = cv2.resize(fr.tool.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    m = patch_mask(p_liver(probe, tok), tool_frac)
    mask = cv2.resize(m.astype(np.uint8), fr.rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    return partial_cloud(fr.rgb, fr.depth, fr.tool, fr.K, np.eye(4), pcfg, rng, mask=mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="StereoMIS split by animal (train = P2, test = P3); no descriptor trains on StereoMIS")
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--pairs-per-seq", type=int, default=60)
    ap.add_argument("--oracle", action="store_true", help="also register with pseudo-GT correspondences (sanity)")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    probe = load_probe()
    rows = {d: [] for d in a.desc}
    for name in sequences(a.split):
        seq = Sequence(name)
        frames = meta(name)["frames"]
        fset = set(frames)
        rng = np.random.default_rng(0)
        fr = np.array([f for f in frames if f in seq.poses])
        t = np.stack([seq.poses[f][:3, 3] for f in fr]) * 1000.0
        chosen = []
        for lo, hi in MOTION_BINS_MM:  # equal numbers of pairs per motion bin
            cand = [(int(fr[x]), int(fr[y])) for x in range(len(fr))
                    for y in np.flatnonzero((fr > fr[x]) & (fr - fr[x] <= MAX_GAP))
                    if lo <= np.linalg.norm(t[y] - t[x]) < hi]
            k = min(a.pairs_per_seq // len(MOTION_BINS_MM), len(cand))
            chosen += [cand[c] for c in sorted(rng.choice(len(cand), k, replace=False))] if k else []
        for i, j in chosen:
            fi, fj = load_frame(name, i), load_frame(name, j)
            ci, cj = liver_cloud(fi, probe, pcfg, rng), liver_cloud(fj, probe, pcfg, rng)
            if min(len(ci), len(cj)) < pcfg.min_partial:
                continue
            T_ji = relative_pose(seq, i, j)
            gi = apply(T_ji, ci.xyz)  # pseudo-GT positions of i's points in camera j
            ov = cKDTree(cj.xyz).query(gi, distance_upper_bound=3.0)[0] < 3.0
            if ov.mean() < 0.2:
                continue
            p_in, A = to_input(ci, random_rigid(rng))
            q_in, B = to_input(cj, random_rigid(rng))
            G = apply(B, gi)
            T_ref = B @ T_ji @ np.linalg.inv(A)
            pair = Pair(subject=name, view=i, P=p_in.xyz, nP=p_in.normals, G=G, Q=q_in.xyz, nQ=q_in.normals,
                        pix=ci.pix, rgb=fi.rgb, K=fi.K, meta={})
            if a.oracle:  # sanity of the pseudo-GT frames: register with its own correspondences
                src = np.flatnonzero(ov)
                tgt = cKDTree(pair.Q).query(G[src])[1]
                T_r, _ = ransac(pair.P, pair.Q, src, tgt, cfg)
                T_o = icp(pair.P, pair.Q, pair.nQ, T_r, cfg)
                rows.setdefault("oracle", []).append({"seq": name, "i": i, "j": j, "motion_mm": float(np.linalg.norm(T_ji[:3, 3])),
                                                      "rre_deg": rre_deg(T_o, T_ref), "rte_mm": rte_mm(T_o, T_ref),
                                                      "rre_ransac": rre_deg(T_r, T_ref)})
            for d in a.desc:
                fs = describe(d, pair, "src")
                # the target is also a real frame with an image: give it its image too
                pj = Pair(subject=name, view=j, P=q_in.xyz, nP=q_in.normals, G=G, Q=q_in.xyz, nQ=q_in.normals,
                          pix=cj.pix, rgb=fj.rgb, K=fj.K, meta={})
                ft = describe(d, pj, "src")
                reg = register(pair.P, fs, pair.Q, ft, pair.nQ, cfg)
                m = reg.matches
                in_ov = ov[m.src]
                ok = correct(G, pair.Q, m.src[in_ov], m.tgt[in_ov], 10.0)
                motion = float(np.linalg.norm(T_ji[:3, 3]))
                rows[d].append({"seq": name, "i": i, "j": j, "gap": j - i, "motion_mm": motion,
                                "motion_bin": next(f"{lo:g}-{hi:g}" for lo, hi in MOTION_BINS_MM if motion < hi + 1e-6
                                                   ) if motion < MOTION_BINS_MM[-1][1] else "40+",
                                "overlap": float(ov.mean()),
                                "point_recall@10": float(correct(G, pair.Q, np.flatnonzero(ov), m.nn[ov], 10.0).mean()),
                                "inlier_ratio@10": float(ok.mean()) if len(ok) else 0.0,
                                "rre_deg": rre_deg(reg.T, T_ref), "rte_mm": rte_mm(reg.T, T_ref),
                                "n_i": len(ci), "n_j": len(cj), "n_matches": len(m.src),
                                "n_matches_in_overlap": int(in_ov.sum()), "n_ransac_inliers": reg.n_ransac_inliers})
    out = Path("outputs/eval/stereomis") / a.split
    out.mkdir(parents=True, exist_ok=True)
    for d, r in rows.items():
        df = pd.DataFrame(r)
        df.to_parquet(out / f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', d)}.parquet")
        print(f"== {d} ({len(df)} pairs)")
        if len(df):
            cols = [c for c in ("point_recall@10", "inlier_ratio@10", "rre_deg", "rte_mm", "rre_ransac",
                                "n_matches", "n_matches_in_overlap") if c in df]
            by = "motion_bin" if "motion_bin" in df else None
            print((df.groupby(by)[cols].median() if by else df[cols].median()).round(3).to_string())
            ok = (df.rre_deg < SUCCESS_RRE_DEG) & (df.rte_mm < SUCCESS_RTE_MM)
            print(f"success (RRE < {SUCCESS_RRE_DEG:g} deg and RTE < {SUCCESS_RTE_MM:g} mm):", round(float(ok.mean()), 3))


if __name__ == "__main__":
    main()

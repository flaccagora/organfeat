"""Why does the fixed pipeline fail on tiny partials (deformed IRCAD, 5 % visible)? Diagnosis on the
dev set (livers 3-20, scripts/external/gen_deformed_ircad.py --livers 3..20 --seed-base 1000), never
on the test pairs. Post hoc (docs/ext_ircad.md); the frozen pipeline is unchanged.

    sbatch -J ircad_diag -t 01:00:00 scripts/leonardo/gpu_run.sbatch python scripts/sanity/ircad_small_patch.py \
        --desc fpfh_r20 learned:<A_GEO> learned:<B_GEO> random

Hypothesis: Open3D's correspondence RANSAC validates each hypothesis by *geometric overlap* (fraction
of source points with a target point within 7.5 mm) and stops early once that overlap is high
(confidence 0.999: ~4 valid hypotheses at fitness 0.95). A small smooth patch overlaps the liver almost
anywhere, so the descriptors only seed hypotheses and never rank them.
Variants (all followed by the frozen ICP):
  frozen     ofeat.match.register
  conf1      the same RANSAC with confidence 1.0: all 100k iterations, still ranked by overlap
  cons       RANSAC ranked by descriptor consensus (# matches within 7.5 mm), same sampling, edge
             and distance checks, 100k hypotheses, inliers refit by Kabsch
  cons_knn3  cons on each partial point's 3 nearest CT descriptors instead of mutual matches
Diagnostics of the frozen run: correct matches, consensus and overlap of the chosen pose vs the oracle.
-> outputs/eval/ircad_dev/<run>.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "scripts/eval")
from ircad_eval import inputs, src_to_tgt

from ofeat._vendor.geometry import kabsch
from ofeat.data import deformed_ircad
from ofeat.data.points import load_config as points_config
from ofeat.eval.descriptors import describe
from ofeat.eval.metrics import apply, correct, pose_metrics
from ofeat.eval.pairs import Pair
from ofeat.match import icp, load_config, match, ransac
from ofeat.match_ext import DistGrid, knn_matches
from ofeat.match_ext import ransac_torch as consensus_ransac


def overlap(T, P, tree, tau) -> tuple[float, float]:
    d = tree.query(apply(T, P))[0]
    inl = d < tau
    return float(inl.mean()), float(np.sqrt((d[inl] ** 2).mean())) if inl.any() else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--run", default="diag")
    ap.add_argument("--crops", nargs="+", type=int, default=[5, 10])
    ap.add_argument("--init", nargs="+", default=["given", "so3"])
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--variants", nargs="+", default=["frozen", "conf1", "cons", "cons_knn3"])
    ap.add_argument("--prior-deg", type=float, default=75.0, help="pose prior of the *_prior variants (given init)")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    cfg1 = dataclasses.replace(cfg, confidence=1.0)
    rows = []
    for crop in a.crops:
        for p in deformed_ircad.load(crop, "dev")[: a.max_pairs]:
            for init in a.init:
                rng = np.random.default_rng([crop, p.idx, init == "so3", 1])
                tgt, src, G, A, B, s = inputs(p, init, pcfg.normal_k, rng)
                pair = Pair(subject=f"ircad_dev{crop}/{init}/liver{p.liver}", view=p.idx, P=tgt.xyz, nP=tgt.normals,
                            G=G, Q=src.xyz, nQ=src.normals, pix=np.zeros((len(tgt), 2), np.int32),
                            rgb=np.zeros((1, 1, 3), np.uint8), K=np.eye(3), meta={})
                P, Q = pair.P.astype(np.float64), pair.Q.astype(np.float64)
                tree = cKDTree(Q)
                T_or = kabsch(P, pair.G.astype(np.float64))
                for name in a.desc:
                    fs, ft = describe(name, pair, "src"), describe(name, pair, "tgt")
                    m = match(fs, ft, cfg.mutual)
                    ok = correct(pair.G, pair.Q, m.src, m.tgt, 10.0)
                    base = {"crop": crop, "idx": p.idx, "liver": p.liver, "init": init, "desc": name,
                            "n_partial": len(P), "n_matches": len(m.src), "n_correct": int(ok.sum())}
                    cons_or = int((np.linalg.norm(apply(T_or, P[m.src]) - Q[m.tgt], axis=1) < cfg.ransac_dist_mm).sum())
                    fit_or, rmse_or = overlap(T_or, P, tree, cfg.ransac_dist_mm)
                    ks, kt = knn_matches(fs, ft)
                    grid = DistGrid(Q)
                    for var in a.variants:
                        t0 = time.perf_counter()
                        ang = a.prior_deg if var.endswith("_prior") else None
                        mi = (ks, kt) if "knn3" in var else (m.src, m.tgt)
                        if var in ("frozen", "conf1"):
                            Tr, _ = ransac(P, Q, m.src, m.tgt, cfg if var == "frozen" else cfg1)
                        elif var.startswith("geo"):  # overlap-ranked (as Open3D), optional prior
                            Tr, _ = consensus_ransac(P, Q, *mi, cfg, max_angle=ang, grid=grid)
                        else:  # cons*: consensus-ranked
                            Tr, _ = consensus_ransac(P, Q, *mi, cfg, max_angle=ang)
                        t1 = time.perf_counter()
                        T = icp(P, Q, pair.nQ, Tr, cfg)
                        pm = pose_metrics(P, pair.G.astype(np.float64), Tr, T)
                        R, t = src_to_tgt(T, A, B, s)
                        fit, rmse = overlap(Tr, P, tree, cfg.ransac_dist_mm)
                        rows.append({**base, "variant": var, "success": pm["success"],
                                     "rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))),
                                     "rot_oracle_deg": float(np.degrees(np.arccos(np.clip((np.trace(T_or[:3, :3]) - 1) / 2, -1, 1)))),
                                     "pose_rmse_ransac": pm["pose_rmse_ransac"], "pose_rmse_icp": pm["pose_rmse_icp"],
                                     "tre_mm": deformed_ircad.paper_metrics(R, t, p)["tre_mm"], "sec_ransac": t1 - t0,
                                     "cons_chosen": int((np.linalg.norm(apply(Tr, P[m.src]) - Q[m.tgt], axis=1)
                                                         < cfg.ransac_dist_mm).sum()),
                                     "cons_oracle": cons_or, "overlap_chosen": fit, "overlap_oracle": fit_or,
                                     "rmse_chosen": rmse, "rmse_oracle": rmse_or})
                torch.cuda.empty_cache()
        print("crop", crop, "done", flush=True)
    out = Path("outputs/eval/ircad_dev")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{a.run}.parquet")
    g = df.groupby(["crop", "init", "desc", "variant"])
    print(g[["success", "tre_mm", "sec_ransac"]].agg(["mean"]).round(3).to_string())
    f = df[df["variant"] == "frozen"]
    fail = f[f["success"] == 0]
    print("\nfrozen failures: n correct matches, consensus chosen vs oracle, overlap chosen vs oracle (medians)")
    print(fail.groupby(["crop", "desc"])[["n_matches", "n_correct", "cons_chosen", "cons_oracle", "overlap_chosen",
                                          "overlap_oracle"]].median().round(3).to_string())


if __name__ == "__main__":
    main()

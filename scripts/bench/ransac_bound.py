"""S0 RANSAC bound: registration success of the fixed pipeline vs the true-inlier ratio of its input
matches, on val livers. Descriptor-free: K matches are drawn per pair, a fraction rho of them correct
(source point -> CT point nearest its true position), the rest to uniformly random CT points.
Tells how good a descriptor's mutual matches must be for the frozen RANSAC + ICP to succeed.

    sbatch -J ransac_bound -c 16 -t 01:00:00 scripts/leonardo/cpu.sbatch python scripts/bench/ransac_bound.py
-> outputs/bench/ransac_bound.parquet, summary printed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ofeat.data import liver_meshes
from ofeat.data.points import load_config as points_config
from ofeat.eval.metrics import pose_metrics
from ofeat.eval.pairs import pairs
from ofeat.match import icp, load_config, ransac

RHOS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2)
KS = (250, 500, 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=4, help="views per val liver")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    rows = []
    for p in pairs("val", liver_meshes.subjects("val"), pcfg, views=slice(0, a.views)):
        rng = np.random.default_rng(p.view)
        true_t = cKDTree(p.Q).query(p.G)[1]
        for K in KS:
            for rho in RHOS:
                src = rng.choice(len(p.P), min(K, len(p.P)), replace=False)
                tgt = true_t[src].copy()
                bad = rng.random(len(src)) >= rho
                tgt[bad] = rng.integers(0, len(p.Q), bad.sum())
                T_r, _ = ransac(p.P, p.Q, src, tgt, cfg)
                T = icp(p.P, p.Q, p.nQ, T_r, cfg)
                rows.append({"subject": p.subject, "view": p.view, "K": K, "rho": rho,
                             "realised_rho": float((~bad).mean()), **pose_metrics(p.P, p.G, T_r, T)})
        print(p.subject, p.view, flush=True)
    df = pd.DataFrame(rows)
    out = Path("outputs/bench/ransac_bound.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(df.groupby(["K", "rho"])["success"].mean().unstack().round(2).to_string())


if __name__ == "__main__":
    main()

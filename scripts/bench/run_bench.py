"""Score descriptors on synthetic (partial view, CT) pairs with the fixed pipeline (docs/protocol.md).

    sbatch -J bench_val -t 02:00:00 scripts/leonardo/gpu_run.sbatch \
        python scripts/bench/run_bench.py --set val --desc fpfh_r10 fpfh_r20 fpfh_r30 sonata
-> outputs/bench/<set>[_noise<s>_holes<h>]/<desc>.parquet (one row per pair), per-liver summary on stdout.
"""

from __future__ import annotations

import argparse
import re
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ofeat.data import liver_meshes
from ofeat.data.points import load_config as points_config
from ofeat.data.points import noisy_depth
from ofeat.eval.descriptors import describe
from ofeat.eval.metrics import pair_metrics
from ofeat.eval.pairs import pairs
from ofeat.match import load_config, register

SUMMARY = ["success", "inlier_ratio@10", "fmr", "point_recall@10", "coverage_auc", "pose_rmse_icp"]


def summarise(df: pd.DataFrame) -> pd.Series:
    per_liver = df.groupby("subject")[SUMMARY].mean()
    return per_liver.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="val, test or a stress set test_* (data/processed/views)")
    ap.add_argument("--sigma100", type=float, default=0.0, help="stress: depth noise sigma at 100 mm (mm)")
    ap.add_argument("--holes", type=float, default=0.0, help="stress: fraction of depth removed in holes")
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N livers (smoke)")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    subjects = liver_meshes.subjects(a.set.split("_")[0])[: a.limit]
    noise = None
    tag = a.set
    if a.sigma100 > 0 or a.holes > 0:
        def noise(depth, rng):
            return noisy_depth(depth, rng, a.sigma100, a.holes)
        tag = f"{a.set}_noise{a.sigma100:g}_holes{a.holes:g}"
    out_dir = Path("outputs/bench") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.desc:
        rows = []
        for p in pairs(a.set, subjects, pcfg, depth_noise=noise):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            fs = describe(name, p, "src")
            t1 = time.perf_counter()
            ft = describe(name, p, "tgt")
            t2 = time.perf_counter()
            reg = register(p.P, fs, p.Q, ft, p.nQ, cfg)
            m = pair_metrics(p.P, p.G, p.Q, reg)
            m.update(subject=p.subject, view=p.view, desc=name, dim=fs.shape[1], sec_desc_src=t1 - t0,
                     sec_desc_tgt=t2 - t1, **p.meta,
                     gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0,
                     cpu_peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            rows.append(m)
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', name)}.parquet")
        print(f"== {name}  ({len(df)} pairs, {df.subject.nunique()} livers)")
        print(summarise(df).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()

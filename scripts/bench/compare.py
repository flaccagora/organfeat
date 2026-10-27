"""Compare benchmark results (docs/protocol.md section 5): per-liver means, 95 % bootstrap CIs over
livers (10 000 resamples), paired Wilcoxon signed-rank vs a reference descriptor.

    python scripts/bench/compare.py --set val --ref fpfh_r20 [--desc a b ...] [--by vis_bin]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["success", "inlier_ratio@10", "fmr", "point_recall@10", "coverage_auc", "pose_rmse_icp",
           "sec_desc_src", "sec_desc_tgt", "sec_ransac", "gpu_peak_mb"]
VIS_BINS = [0, 0.10, 0.20, 0.30, 1.01]


def load(view_set: str, names: list[str] | None) -> pd.DataFrame:
    files = sorted((Path("outputs/bench") / view_set).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if names:
        df = df[df["desc"].isin(names)]
    df["vis_bin"] = pd.cut(df["vis_frac"], VIS_BINS, right=False,
                           labels=["<10%", "10-20%", "20-30%", ">=30%"])
    return df


def boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def table(df: pd.DataFrame, ref: str | None, metric: str = "success") -> pd.DataFrame:
    per = df.groupby(["desc", "subject"])[[c for c in METRICS if c in df]].mean()
    rows = []
    for d, g in per.groupby(level=0):
        g = g.droplevel(0)
        lo, hi = boot_ci(g[metric].to_numpy())
        row = {"desc": d, "livers": len(g), **g.mean().round(4).to_dict(), f"{metric}_ci": f"[{lo:.3f}, {hi:.3f}]"}
        if ref is not None and d != ref and ref in per.index.get_level_values(0):
            r = per.loc[ref][metric]
            common = g.index.intersection(r.index)
            diff = g.loc[common, metric] - r.loc[common]
            row[f"d_{metric}_vs_ref"] = round(float(diff.mean()), 4)
            row["wilcoxon_p"] = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("desc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--desc", nargs="*", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--metric", default="success")
    ap.add_argument("--by", default=None, help="extra grouping column, e.g. vis_bin or family")
    a = ap.parse_args()
    df = load(a.set, a.desc)
    pd.set_option("display.width", 250)
    if a.by:
        for k, g in df.groupby(a.by, observed=True):
            print(f"== {a.by} = {k}  ({len(g)} pairs)")
            print(table(g, a.ref, a.metric).to_string())
    else:
        print(table(df, a.ref, a.metric).to_string())


if __name__ == "__main__":
    main()

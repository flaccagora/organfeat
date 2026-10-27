"""S3 test report (docs/protocol.md section 5): per-liver means, 95 % bootstrap CIs, paired Wilcoxon for
the pre-registered comparisons, in a given bench directory.

    python scripts/bench/test_report.py --set test_noise1_holes0.1
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts/bench")
from compare import boot_ci, load  # noqa: E402

M = ["success", "inlier_ratio@10", "point_recall@10", "coverage_auc", "fmr", "pose_rmse_icp",
     "sec_desc_src", "sec_desc_tgt", "sec_ransac", "gpu_peak_mb"]


def short(d: str) -> str:
    names = {"s1_geo_lr1e-4/step_008000.pt": "A_geo", "s2_mm_lam0_lr1e-4/step_010000.pt": "A_mm",
             "s2_mm_lam0_lr1e-4/step_010000.pt:geo": "A_mm:geo", "s1_ppf/step_008000.pt": "B_geo",
             "s2_ppf_lam0/step_008000.pt": "B_mm", "s2_ppf_lam0/step_008000.pt:geo": "B_mm:geo"}
    return names.get(d.replace("learned:outputs/train/", ""), d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    a = ap.parse_args()
    df = load(a.set, None)
    df["desc"] = df["desc"].map(short)
    per = df.groupby(["desc", "subject"])[[c for c in M if c in df]].mean()
    rows = []
    for d, g in per.groupby(level=0):
        lo, hi = boot_ci(g["success"].to_numpy())
        ilo, ihi = boot_ci(g["inlier_ratio@10"].to_numpy())
        rows.append({"desc": d, "livers": len(g), "pairs": int((df.desc == d).sum()),
                     "success": g["success"].mean(), "success_ci": f"[{lo:.3f}, {hi:.3f}]",
                     "inlier_ratio@10": g["inlier_ratio@10"].mean(), "ir_ci": f"[{ilo:.3f}, {ihi:.3f}]",
                     "point_recall@10": g["point_recall@10"].mean(), "coverage_auc": g["coverage_auc"].mean(),
                     "pose_rmse_icp": g["pose_rmse_icp"].mean(),
                     "ms_desc": 1000 * (g["sec_desc_src"].mean() + g["sec_desc_tgt"].mean()),
                     "ms_ransac": 1000 * g["sec_ransac"].mean(), "gpu_mb": g["gpu_peak_mb"].max()})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index("desc").round(4).to_string())
    print()
    for x, y in (("A_mm", "A_geo"), ("A_mm", "A_mm:geo"), ("B_mm", "B_geo"), ("B_mm", "B_mm:geo"),
                 ("A_geo", "fpfh_r20"), ("A_mm", "fpfh_r20"), ("B_geo", "fpfh_r20"), ("B_mm", "fpfh_r20"),
                 ("A_mm", "B_mm")):
        if x not in per.index.get_level_values(0) or y not in per.index.get_level_values(0):
            continue
        out = [f"{x} - {y}:"]
        for m in ("success", "inlier_ratio@10", "point_recall@10"):
            d = (per.loc[x][m] - per.loc[y][m]).dropna()
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            out.append(f"{m} {d.mean():+.4f} (+{int((d > 0).sum())}/-{int((d < 0).sum())}, p {p:.2g})")
        print("  ".join(out))


if __name__ == "__main__":
    main()

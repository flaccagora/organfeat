"""S3 stress report: per-liver means of success and inlier ratio across the test stress sets, plus the
paired image comparisons (A_mm - A_geo, B_mm - B_geo) per set (Wilcoxon over the 46 test livers).

    sbatch -J stress_report -t 00:30:00 scripts/leonardo/cpu.sbatch python scripts/bench/stress_report.py
-> outputs/bench/stress_report.md
"""

from __future__ import annotations

import glob
import sys

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts/bench")
from test_report import short  # noqa: E402

SETS = [("test_noise1_holes0.1", "base (σ 1 mm)"), ("test", "σ 0"), ("test_noise0.5_holes0.1", "σ 0.5 mm"),
        ("test_noise2_holes0.1", "σ 2 mm"), ("test_tex_heldout_noise1_holes0.1", "held-out textures"),
        ("test_tex_flat_noise1_holes0.1", "flat albedo"), ("test_tools0_noise1_holes0.1", "0 tools"),
        ("test_tools1_noise1_holes0.1", "1 tool"), ("test_tools2_noise1_holes0.1", "2 tools"),
        ("test_tools3_noise1_holes0.1", "3 tools"), ("test_deform_noise1_holes0.1", "ARAP 2–8 mm")]
ORDER = ["fpfh_r20", "A_geo", "A_mm", "A_mm:geo", "B_geo", "B_mm", "B_mm:geo"]


def md(t: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join([t.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in t.iterrows():
        lines.append("| " + " | ".join([str(i)] + [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r]) + " |")
    return "\n".join(lines)


def main() -> None:
    per = {}
    for s, label in SETS:
        fs = glob.glob(f"outputs/bench/{s}/*.parquet")
        df = pd.concat([pd.read_parquet(f) for f in fs])
        df["desc"] = df["desc"].map(short)
        per[label] = df[df.desc.isin(ORDER)].groupby(["desc", "subject"])[["success", "inlier_ratio@10", "vis_frac"]].mean()
    out = []
    for m in ("success", "inlier_ratio@10"):
        t = pd.DataFrame({k: v[m].groupby(level=0).mean() for k, v in per.items()}).reindex(ORDER)
        out += [f"### {m} (per-liver mean, 46 test livers)", "", md(t), ""]
    rows = []
    for label, p in per.items():
        r = {"set": label}
        for x, y in (("A_mm", "A_geo"), ("B_mm", "B_geo")):
            for m, short_m in (("success", "succ"), ("inlier_ratio@10", "IR")):
                d = (p.loc[x][m] - p.loc[y][m]).dropna()
                pv = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                r[f"{x}−{y} {short_m}"] = f"{d.mean():+.3f} (p {pv:.1g})"
        rows.append(r)
    out += ["### images: paired differences per set", "", md(pd.DataFrame(rows).set_index("set")), ""]
    # visible fraction bins on the base set
    fs = glob.glob("outputs/bench/test_noise1_holes0.1/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in fs])
    df["desc"] = df["desc"].map(short)
    df["vis_bin"] = pd.cut(df["vis_frac"], [0, 0.10, 0.20, 0.30, 1.01], right=False,
                           labels=["<10%", "10-20%", "20-30%", ">=30%"])
    t = df[df.desc.isin(ORDER)].groupby(["desc", "vis_bin"], observed=True)["inlier_ratio@10"].mean().unstack().reindex(ORDER)
    s = df[df.desc.isin(ORDER)].groupby(["desc", "vis_bin"], observed=True)["success"].mean().unstack().reindex(ORDER)
    n = df[df.desc == "fpfh_r20"].groupby("vis_bin", observed=True).size()
    out += ["### visible fraction bins (base setting; pairs per bin: " + ", ".join(f"{k} {v}" for k, v in n.items()) + ")", "",
            "success:", "", md(s), "", "inlier ratio@10:", "", md(t), ""]
    open("outputs/bench/stress_report.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()

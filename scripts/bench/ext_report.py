"""Post-hoc external benchmark tables (docs/ext_ircad.md, docs/ext_p2i_syn.md), next to the papers' numbers.

    python scripts/bench/ext_report.py --ircad outputs/eval/ircad/final.parquet \
        --p2i-syn outputs/eval/p2i_syn/final_g1.parquet outputs/eval/p2i_syn/final_g2.parquet ...
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/bench")
from test_report import short

# Neri et al. 2025, Table 2 (deformed IRCAD, mean +- sd): TRE mm, MAE(R) deg, MAE(t) mm
NERI = {
    5: {"OMNet": ("23.49±14.14", "9.00±4.33", "7.20±3.64"), "Lepard": ("27.03±39.44", "10.94±18.04", "10.58±15.79"),
        "LiverMatch": ("18.69±28.58", "7.47±6.09", "8.41±10.67"), "Neri et al.": ("22.36±12.04", "6.60±4.14", "7.72±4.77"),
        "GT (residual)": ("6.45±5.62", "0", "0")},
    10: {"OMNet": ("15.23±8.36", "5.77±2.58", "5.29±2.17"), "Lepard": ("9.82±8.59", "3.91±2.30", "4.86±3.04"),
         "LiverMatch": ("7.91±5.90", "3.64±2.03", "4.68±2.17"), "Neri et al.": ("13.01±6.07", "3.79±2.25", "4.25±2.00"),
         "GT (residual)": ("6.17±5.18", "0", "0")},
    15: {"OMNet": ("11.72±6.45", "4.37±1.95", "3.82±1.91"), "Lepard": ("7.45±5.49", "3.21±1.95", "3.64±2.08"),
         "LiverMatch": ("6.39±4.54", "3.08±1.91", "3.73±2.00"), "Neri et al.": ("8.49±2.62", "2.85±1.32", "2.60±1.13"),
         "GT (residual)": ("5.37±4.82", "0", "0")},
    25: {"OMNet": ("10.45±4.81", "3.86±1.76", "3.73±1.82"), "Lepard": ("7.41±6.69", "3.27±1.68", "3.04±2.26"),
         "LiverMatch": ("4.81±3.21", "2.78±1.52", "2.78±1.47"), "Neri et al.": ("6.78±2.08", "2.33±1.06", "2.86±0.87"),
         "GT (residual)": ("5.01±4.40", "0", "0")},
}
# Self-P2IR Table I, RANSAC-50k with 1000 / 250 samples: FMR, IR, RR (rho 10 mm, Table IV), RRE deg, RTE mm
SELF_P2IR = {"Predator (1000)": (95.3, 73.9, 89.0, 0.288, 2.16), "GeoTrans (1000)": (96.9, 94.7, 93.5, 0.224, 1.44),
             "Self-P2I (1000)": (96.8, 95.0, 99.0, 0.190, 1.21), "Self-P2I (250)": (96.8, 98.1, 98.8, 0.194, 1.24)}
ORDER = ["fpfh_r20", "sonata", "A_geo", "A_mm:geo", "B_geo", "B_mm:geo", "random"]


def md(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join(lines)


def pm(x: pd.Series, sd: float | None = None, f: str = "{:.2f}") -> str:
    return (f + "±" + f).format(x.mean(), x.std() if sd is None else sd)


def ircad(path: str) -> None:
    df = pd.read_parquet(path)
    df["desc"] = df["desc"].map(short)
    for init in ("given", "so3"):
        print(f"\n### Deformed IRCAD, init = {init}\n")
        for crop in sorted(df["crop"].unique()):
            rows = {}
            if init == "given":
                rows.update({f"*{k}*": {"TRE (mm)": v[0], "MAE(R) (°)": v[1], "MAE(t) (mm)": v[2]}
                             for k, v in NERI[crop].items()})
            g = df[(df["crop"] == crop) & (df["init"].isin([init, "-"]))]
            for d in ["gt"] + ORDER:
                x = g[g["desc"] == d]
                if x.empty:
                    continue
                sd = float(np.sqrt(x["tre_sq_mm2"].mean() - x["tre_mm"].mean() ** 2))  # pooled over landmarks
                r = {"TRE (mm)": pm(x["tre_mm"], sd), "MAE(R) (°)": pm(x["mae_r_deg"]), "MAE(t) (mm)": pm(x["mae_t_mm"]),
                     "median TRE": f"{x['tre_mm'].median():.2f}"}
                if d != "gt":
                    r.update({"TRE > 20 mm": f"{(x['tre_mm'] > 20).mean():.2f}", "success": f"{x['success'].mean():.2f}",
                              "IR@10": f"{x['inlier_ratio@10'].mean():.3f}"})
                rows["ours: GT (residual)" if d == "gt" else d] = r
            t = pd.DataFrame(rows).T.fillna("")
            t.index.name = f"{crop} %"
            print(md(t) + "\n")


def ircad_compact(path: str) -> None:
    """One row per method, crops as columns: TRE (mean +- pooled sd) and MAE(R) / MAE(t) means."""
    df = pd.read_parquet(path)
    df["desc"] = df["desc"].map(short)
    crops = sorted(df["crop"].unique())
    for init in ("given", "so3"):
        rows = {}
        if init == "given":
            for k in NERI[crops[0]]:
                rows[f"*{k}*"] = {f"{c} %": f"{NERI[c][k][0]} / {NERI[c][k][1].split('±')[0]} / {NERI[c][k][2].split('±')[0]}"
                                  for c in crops}
        for d in ["gt"] + ORDER:
            r = {}
            for c in crops:
                x = df[(df["crop"] == c) & (df["init"].isin([init, "-"])) & (df["desc"] == d)]
                if x.empty:
                    continue
                sd = float(np.sqrt(x["tre_sq_mm2"].mean() - x["tre_mm"].mean() ** 2))
                r[f"{c} %"] = (f"{x['tre_mm'].mean():.2f}±{sd:.2f} / {x['mae_r_deg'].mean():.2f} / {x['mae_t_mm'].mean():.2f}"
                               + ("" if d == "gt" else f" ({x['success'].mean():.2f})"))
            if r:
                rows["ours: GT (residual)" if d == "gt" else d] = r
        t = pd.DataFrame(rows).T.fillna("")
        t.index.name = f"init = {init}"
        print(md(t) + "\n")


def ircad_prior(frozen: str, prior: str) -> None:
    """Frozen pipeline vs the post-hoc extension (ofeat/match_ext.py), `given` init: TRE mean (median),
    MAE(R), success; the paper's best rows for reference."""
    df = pd.concat([pd.read_parquet(frozen).assign(pipeline="frozen"), pd.read_parquet(prior)], ignore_index=True)
    df = df[(df["init"] == "given") & (df["desc"] != "gt")]
    df["desc"] = df["desc"].map(short)
    for crop in sorted(df["crop"].unique()):
        rows = {f"*{k}*": {"frozen": "", "knn3 + prior (≤ 75°)": f"TRE {v[0]} / MAE(R) {v[1]}"}
                for k, v in NERI[crop].items() if k in ("LiverMatch", "Neri et al.", "GT (residual)")}
        for d in ORDER:
            r = {}
            for pipe, col in (("frozen", "frozen"), ("knn3_prior", "knn3 + prior (≤ 75°)")):
                x = df[(df["crop"] == crop) & (df["desc"] == d) & (df["pipeline"] == pipe)]
                if not x.empty:
                    r[col] = (f"TRE {x['tre_mm'].mean():.2f} (median {x['tre_mm'].median():.2f}) / MAE(R) "
                              f"{x['mae_r_deg'].mean():.2f} / success {x['success'].mean():.2f}")
            if r:
                rows[d] = r
        t = pd.DataFrame(rows).T.fillna("")
        t.index.name = f"{crop} %"
        print(md(t) + "\n")


def p2i_syn(paths: list[str]) -> None:
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df[~df["skipped"]].copy()
    df["desc"] = df["desc"].map(short)
    print(f"\n### P2I-LReg synthetic test split: {df[['case', 'frame']].drop_duplicates().shape[0]} frames, "
          f"{df['case'].nunique()} cases\n")
    rows = {f"*{k}*": {"FMR": f"{v[0]:.1f}", "IR (top-k)": f"{v[1]:.1f}", "RR@10": f"{v[2]:.1f}",
                       "RRE (°)": f"{v[3]:.3f}", "RTE (mm)": f"{v[4]:.2f}"} for k, v in SELF_P2IR.items()}
    for d in ORDER:
        x = df[df["desc"] == d]
        if x.empty:
            continue
        ok = x["pose_rmse_icp"] < 10
        okr = x["pose_rmse_ransac"] < 10
        rows[d] = {"FMR": f"{100 * x['fmr'].mean():.1f}", "IR (top-k)": f"{100 * x['ir_top1000'].mean():.1f}",
                   "RR@10": f"{100 * ok.mean():.1f}", "RRE (°)": f"{x.loc[ok, 'rre_icp'].mean():.3f}",
                   "RTE (mm)": f"{x.loc[ok, 'rte_icp'].mean():.2f}", "IR top-250": f"{100 * x['ir_top250'].mean():.1f}",
                   "IR all": f"{100 * x['inlier_ratio@10'].mean():.1f}",
                   "RR@5": f"{100 * (x['pose_rmse_icp'] < 5).mean():.1f}", "RR@2": f"{100 * (x['pose_rmse_icp'] < 2).mean():.1f}",
                   "RR@10 RANSAC only": f"{100 * (x['pose_rmse_ransac'] < 10).mean():.1f}",
                   "RRE / RTE RANSAC only": f"{x.loc[okr, 'rre_ransac'].mean():.3f} / {x.loc[okr, 'rte_ransac'].mean():.2f}",
                   "worst case RR@10": f"{100 * ok.groupby(x['case']).mean().min():.1f}"}
    t = pd.DataFrame(rows).T.fillna("")
    t.index.name = "method"
    print(md(t) + "\n")
    print("Our IR (top-k) is the top-1000 by ratio-test confidence; RRE / RTE are means over RR@10 successes "
          "(RANSAC only: over RANSAC-only successes).\n")
    for dev in (True, False):
        x = df[df["dev"] == dev]
        s = x.assign(ok=x["pose_rmse_icp"] < 10).groupby("desc")[["ok", "ir_top1000"]].mean().reindex(ORDER).dropna()
        print(f"{'dev' if dev else 'locked'} cases ({x['case'].nunique()}): " +
              ", ".join(f"{d} RR {100 * r.ok:.1f} / IR {100 * r.ir_top1000:.1f}" for d, r in s.iterrows()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ircad", default=None)
    ap.add_argument("--p2i-syn", nargs="+", default=None)
    ap.add_argument("--compact", action="store_true", help="IRCAD: one cross-crop table per init")
    ap.add_argument("--ircad-prior", default=None, help="IRCAD run of --pipeline knn3_prior, vs --ircad")
    a = ap.parse_args()
    if a.ircad and a.ircad_prior:
        ircad_prior(a.ircad, a.ircad_prior)
    elif a.ircad:
        (ircad_compact if a.compact else ircad)(a.ircad)
    if a.p2i_syn:
        p2i_syn(a.p2i_syn)


if __name__ == "__main__":
    main()

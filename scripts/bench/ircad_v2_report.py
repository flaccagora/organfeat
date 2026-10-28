"""IRCAD v2 report (R2; statistics pre-registered in docs/protocol.md, 2026-10-28).

    python scripts/bench/ircad_v2_report.py outputs/eval/ircad_v2/*.parquet

- unit = liver: per-liver means, then the mean over the 20 livers with a 95 % bootstrap CI over livers
  (10,000 resamples)
- primary: B_geo vs FPFH and A_geo vs FPFH on per-liver mean TRE, paired Wilcoxon over livers,
  Holm-corrected across the 4 crops (given init, frozen pipeline)
- training seeds: mean +- SD over seeds 0-2 of the across-liver mean TRE
- match_ext: livers 1-2 (clean) and 3-20 (selected on dev pairs of these livers, flagged) separately
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts/bench")
from ext_report import md  # noqa: E402
from test_report import short  # noqa: E402


def name(d: str) -> str:
    """Short descriptor name; seed replicates get '#<seed>' (seed 0 = the original run)."""
    m = re.match(r"learned:outputs/train/(.+)_seed(\d)/(step_\d+\.pt)(:geo)?$", d)
    if m:
        return short(f"learned:outputs/train/{m[1]}/{m[3]}{m[4] or ''}") + f"#{m[2]}"
    return short(d)


def boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    out, run = np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out.tolist()


def main() -> None:
    df = pd.concat([pd.read_parquet(f) for f in sys.argv[1:]], ignore_index=True)
    df = df[df["desc"] != "gt"].copy()
    df["d"] = df["desc"].map(name)
    df["base"] = df["d"].str.replace(r"#\d$", "", regex=True)
    df["seed"] = df["d"].str.extract(r"#(\d)$")[0].fillna("0").astype(int)
    per = df.groupby(["pipeline", "init", "crop", "d", "base", "seed", "liver"])[
        ["tre_mm", "success", "mae_r_deg", "mae_t_mm"]].mean().reset_index()

    print("## Across-liver means (seed-0 models; 95 % bootstrap CI over livers)\n")
    for (pipe, init), g in per[per["seed"] == 0].groupby(["pipeline", "init"]):
        rows = {}
        for (crop, d), x in g.groupby(["crop", "d"]):
            lo, hi = boot_ci(x["tre_mm"].to_numpy())
            rows.setdefault(d, {})[f"{crop} %"] = (f"{x['tre_mm'].mean():.2f} [{lo:.2f}, {hi:.2f}] / "
                                                  f"{x['success'].mean():.2f}")
        t = pd.DataFrame(rows).T
        t.index.name = f"{pipe}, {init}: TRE [CI] / success"
        print(md(t) + "\n")

    print("## Primary comparisons (frozen pipeline, given init; per-liver mean TRE; Wilcoxon, Holm over crops)\n")
    g = per[(per["pipeline"] == "frozen") & (per["init"] == "given") & (per["seed"] == 0)]
    for a_, b_ in (("B_geo", "fpfh_r20"), ("A_geo", "fpfh_r20")):
        res = []
        for crop in sorted(g["crop"].unique()):
            x = g[(g["crop"] == crop)].pivot(index="liver", columns="d", values="tre_mm")
            if a_ not in x or b_ not in x:
                continue
            dlt = (x[a_] - x[b_]).to_numpy()
            res.append((crop, float(np.median(dlt)), int((dlt < 0).sum()), int((dlt > 0).sum()),
                        float(wilcoxon(x[a_], x[b_]).pvalue)))
        for (crop, dm, nb, nw, p), ph in zip(res, holm([r[-1] for r in res])):
            print(f"- {a_} vs {b_}, {crop} %: median Δ {dm:+.2f} mm, livers better / worse {nb} / {nw}, "
                  f"p {p:.2g} (Holm {ph:.2g})")
    print()

    print("## Training seeds (frozen, given): mean over livers per seed, then mean ± SD over seeds\n")
    s = per[(per["pipeline"] == "frozen") & (per["init"] == "given") & per["base"].str.match(r"[AB]_")]
    s = s.groupby(["crop", "base", "seed"])["tre_mm"].mean().reset_index()
    rows = {}
    for (crop, b), x in s.groupby(["crop", "base"]):
        rows.setdefault(b, {})[f"{crop} %"] = f"{x['tre_mm'].mean():.2f} ± {x['tre_mm'].std(ddof=1):.2f} (n={len(x)})"
    t = pd.DataFrame(rows).T
    t.index.name = "TRE mean ± SD over seeds"
    print(md(t) + "\n")

    me = per[(per["pipeline"] == "knn3_prior") & (per["seed"] == 0)]
    if len(me):
        print("## match_ext (given): livers 1-2 (clean) vs 3-20 (flagged: selected on dev pairs of these livers)\n")
        rows = {}
        for (crop, d), x in me.groupby(["crop", "d"]):
            a_, b_ = x[x["liver"] <= 2], x[x["liver"] > 2]
            rows.setdefault(d, {})[f"{crop} %"] = (f"{a_['tre_mm'].mean():.1f} / {b_['tre_mm'].mean():.1f} "
                                                  f"(succ {a_['success'].mean():.2f} / {b_['success'].mean():.2f})")
        t = pd.DataFrame(rows).T
        t.index.name = "TRE livers 1-2 / 3-20"
        print(md(t) + "\n")


if __name__ == "__main__":
    main()

"""Deformed IRCAD (Neri et al. 2025) evaluation: post-hoc external benchmark (docs/ext_ircad.md).
Nothing is selected on it.

    sbatch -J ircad_eval -t 03:00:00 scripts/leonardo/gpu_run.sbatch python scripts/eval/ircad_eval.py \
        --run final --desc fpfh_r20 sonata learned:<A_GEO> ... random

Per pair (crop ratio 5 / 10 / 15 / 25 %, livers 1-2 x 50) and initial-pose setting:
  given  the generator's poses (each Euler angle <= 45 deg, translation <= 0.5 units): the paper's protocol
  so3    an independent random rigid transform per cloud, then centring: this project's protocol
Inputs: normalised coordinates x the liver's mean half-extent s (~mm, for the frozen mm thresholds).
CT side = src (3500 points) with the mesh's vertex normals; partial = tgt with kNN-PCA normals facing
the generator's camera. Clouds as given (no voxelisation; they are sparser than 3 mm). The fixed
pipeline registers partial -> src; its inverse (src -> tgt) is scored against gt_T with the paper's
metrics (ofeat.data.deformed_ircad.paper_metrics), and the partial with this project's pair_metrics
(G = each partial point's undeformed source position; success = pose RMSE vs the oracle rigid fit).
desc "gt" scores gt_T itself (the paper's "GT" row: the residual non-rigid TRE).
--pipeline knn3_prior: the post-hoc extension for tiny partials (ofeat/match_ext.py; k-NN matches, torch
RANSAC with a rotation prior <= 75 deg, selected on the dev livers), `given` init only.
-> outputs/eval/ircad/<run>.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ofeat.data import deformed_ircad
from ofeat.data.points import Cloud, _normals, random_rigid, to_input
from ofeat.data.points import load_config as points_config
from ofeat.eval.descriptors import describe
from ofeat.eval.metrics import pair_metrics
from ofeat.eval.pairs import Pair
from ofeat.match import load_config, register
from ofeat.match_ext import register_prior


def inputs(p: deformed_ircad.IrcadPair, init: str, k: int, rng: np.random.Generator):
    """(partial, CT, G, A, B, s): input clouds, scaled by s; A, B map the scaled given frames to the inputs."""
    s = float(p.half.mean())
    tgt = Cloud(xyz=(p.tgt * s).astype(np.float32),
                normals=_normals(p.tgt * s, k, viewpoint=p.pov * s).astype(np.float32))
    src = Cloud(xyz=(p.src * s).astype(np.float32), normals=p.src_normals.astype(np.float32))
    if init == "given":
        A = B = np.eye(4)
    else:
        tgt, A = to_input(tgt, random_rigid(rng))
        src, B = to_input(src, random_rigid(rng))
    G = (p.src[p.corr] * s) @ B[:3, :3].T + B[:3, 3]
    return tgt, src, G.astype(np.float32), A, B, s


def src_to_tgt(T: np.ndarray, A: np.ndarray, B: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline output (partial input -> CT input) -> (R, t) src -> tgt in normalised units."""
    T0 = np.linalg.inv(B) @ T @ A  # scaled tgt -> scaled src
    R = T0[:3, :3].T
    return R, -R @ T0[:3, 3] / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--run", default="final")
    ap.add_argument("--crops", nargs="+", type=int, default=list(deformed_ircad.CROPS))
    ap.add_argument("--split", default="test", choices=["test", "v2", "dev"],
                    help="v2: the locked 20-liver test (OFEAT_UNLOCK_EVAL=ircad_v2; docs/protocol.md 2026-10-28)")
    ap.add_argument("--init", nargs="+", default=["given", "so3"], choices=["given", "so3"])
    ap.add_argument("--max-pairs", type=int, default=None, help="per crop (smoke)")
    ap.add_argument("--pipeline", default="frozen", choices=["frozen", "knn3_prior"],
                    help="knn3_prior: ofeat.match_ext.register_prior (pose prior: given init only)")
    ap.add_argument("--prior-deg", type=float, default=75.0)
    a = ap.parse_args()
    if a.pipeline == "knn3_prior" and a.init != ["given"]:
        ap.error("the pose prior is centred on the generator's poses: use --init given")
    cfg, pcfg = load_config(), points_config()
    rows = []
    for crop in a.crops:
        for p in deformed_ircad.load(crop, a.split)[: a.max_pairs]:
            base = {"crop": crop, "idx": p.idx, "liver": p.liver, "n_partial": len(p.tgt), "n_ct": len(p.src),
                    "pipeline": a.pipeline, "split": a.split}
            rows.append({**base, "init": "-", "desc": "gt",
                         **deformed_ircad.paper_metrics(p.gt_T[:, :3], p.gt_T[:, 3], p)})
            for init in a.init:
                rng = np.random.default_rng([crop, p.idx, init == "so3"])
                tgt, src, G, A, B, s = inputs(p, init, pcfg.normal_k, rng)
                pair = Pair(subject=f"ircad{crop}/{init}/liver{p.liver}", view=p.idx, P=tgt.xyz, nP=tgt.normals,
                            G=G, Q=src.xyz, nQ=src.normals, pix=np.zeros((len(tgt), 2), np.int32),
                            rgb=np.zeros((1, 1, 3), np.uint8), K=np.eye(3), meta={})
                for name in a.desc:
                    fs, ft = describe(name, pair, "src"), describe(name, pair, "tgt")
                    if a.pipeline == "frozen":
                        reg = register(pair.P, fs, pair.Q, ft, pair.nQ, cfg)
                    else:
                        reg = register_prior(pair.P, fs, pair.Q, ft, pair.nQ, cfg, max_angle=a.prior_deg)
                    m = pair_metrics(pair.P, pair.G, pair.Q, reg)
                    m.pop("coverage_curve")
                    R, t = src_to_tgt(reg.T, A, B, s)
                    Rr, tr = src_to_tgt(reg.T_ransac, A, B, s)
                    ransac = deformed_ircad.paper_metrics(Rr, tr, p)
                    rows.append({**base, "init": init, "desc": name, "scale_mm": s, **m,
                                 **deformed_ircad.paper_metrics(R, t, p),
                                 "tre_ransac_mm": ransac["tre_mm"], "mae_r_ransac_deg": ransac["mae_r_deg"]})
                torch.cuda.empty_cache()
        print("crop", crop, "done", flush=True)
    out = Path("outputs/eval/ircad" if a.split == "test" else f"outputs/eval/ircad_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', a.run)}.parquet")
    g = df.groupby(["crop", "init", "desc"])
    summ = g[["tre_mm", "mae_r_deg", "mae_t_mm"]].mean()
    summ["tre_sd_pooled"] = np.sqrt(g["tre_sq_mm2"].mean() - summ["tre_mm"] ** 2)  # over all landmarks
    summ["success"] = g["success"].mean() if "success" in df else np.nan
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()

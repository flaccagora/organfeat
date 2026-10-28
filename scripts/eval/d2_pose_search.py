"""D2: a deformation-tolerant rigid stage (docs/protocol.md, 2026-10-28; post hoc).

    sbatch -J d2_depoll -t 06:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=depoll scripts/leonardo/gpu_run.sbatch \
        python scripts/eval/d2_pose_search.py --data depoll --run d2 --desc fpfh_r20 learned:<A_GEO> learned:<B_GEO>
    ... --data ircad_v2 (OFEAT_UNLOCK_EVAL=ircad_v2)

Arms (bounded / given initial pose only; the search is local):
  frozen, match_ext     the descriptor pipelines (as D1 / R2)
  search                ofeat.match_ext.closest_pose_search from the initial pose (descriptor-free;
                        defreg OT-20), then the frozen ICP
  search+<desc>         the same grid plus the descriptor's frozen and match_ext poses as seeds
DePoLL: D1's conditions (C1 shape, C2 + deformation, C3 + reconstruction) and D1's bounded poses
(scripts/eval/depoll_decompose.py), marker metrics. IRCAD v2: init "given", the paper's metrics.
-> outputs/eval/d2/<data>_<run>/<unit>.parquet (resumable), merged into outputs/eval/d2/<data>_<run>.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts/eval")
from depoll_decompose import conditions
from depoll_whole_eval import SUCCESS_MARGIN_MM, bounded_pose, recon_points
from ircad_eval import inputs, src_to_tgt

from ofeat._vendor.geometry import kabsch
from ofeat.data import deformed_ircad, depoll
from ofeat.data.points import Cloud, ct_cloud
from ofeat.data.points import load_config as points_config
from ofeat.eval.descriptors import describe
from ofeat.eval.metrics import apply, rre_deg, rte_mm
from ofeat.eval.pairs import Pair
from ofeat.match import icp, load_config, register
from ofeat.match_ext import closest_pose_search, register_prior


def arms(pair: Pair, descs: list[str], cfg) -> dict:
    """{arm: (T partial -> CT, extras)} for one pair whose initial pose is the identity."""
    out, seeds = {}, {}
    for name in descs:
        fs, ft = describe(name, pair, "src"), describe(name, pair, "tgt")
        r1 = register(pair.P, fs, pair.Q, ft, pair.nQ, cfg)
        r2 = register_prior(pair.P, fs, pair.Q, ft, pair.nQ, cfg, max_angle=75.0)
        out[f"frozen/{name}"], out[f"match_ext/{name}"] = (r1.T, {}), (r2.T, {})
        seeds[name] = [r1.T_ransac, r2.T_ransac]
    T, ex = closest_pose_search(pair.P, pair.Q, [np.eye(4)])
    out["search"] = (icp(pair.P, pair.Q, pair.nQ, T, cfg), ex)
    for name in descs:
        T, ex = closest_pose_search(pair.P, pair.Q, [np.eye(4), *seeds[name]])
        out[f"search+{name}"] = (icp(pair.P, pair.Q, pair.nQ, T, cfg), ex)
    return out


def depoll_unit(state, a, cfg, pcfg, ref):
    rows = []
    pairs_m = depoll.state_markers(state)
    ref_mk = np.vstack([pairs_m[k].ref for k in pairs_m])
    rec_mk = np.vstack([pairs_m[k].deformed for k in pairs_m])
    n_balls = len(pairs_m["balls"].ref)
    ct = ct_cloud(ref, pcfg, np.random.default_rng(1000 + int(state)))
    T_or0 = kabsch(rec_mk, ref_mk)
    rec_x, rec_n = recon_points(state, pcfg, np.random.default_rng([int(state), 0]))
    for cond, (pc, T0) in conditions(state, ref, rec_x, rec_n, T_or0, pcfg).items():
        mk_part = ref_mk if cond == "C1_shape" else rec_mk
        for k in range(2 if a.smoke else a.poses):
            prng = np.random.default_rng([int(state), False, k])  # D1's bounded poses
            A = bounded_pose(prng, apply(T0, pc.xyz)) @ T0
            p_in = Cloud(xyz=apply(A, pc.xyz).astype(np.float32), normals=(pc.normals @ A[:3, :3].T).astype(np.float32))
            mk_src, mk_tgt = apply(A, mk_part), ref_mk
            T_or = kabsch(mk_src, mk_tgt)
            tre_or = float(np.linalg.norm(apply(T_or, mk_src) - mk_tgt, axis=1).mean())
            pair = Pair(subject=f"d2/{state}/{cond}", view=k, P=p_in.xyz, nP=p_in.normals, G=np.zeros_like(p_in.xyz),
                        Q=ct.xyz, nQ=ct.normals, pix=np.zeros((len(pc), 2), np.int32), rgb=np.zeros((1, 1, 3), np.uint8),
                        K=np.eye(3), meta={})
            for arm, (T, ex) in arms(pair, a.desc, cfg).items():
                e = np.linalg.norm(apply(T, mk_src) - mk_tgt, axis=1)
                rows.append({"state": state, "cond": cond, "pose": k, "arm": arm, "n_partial": len(pc),
                             "tre_oracle_mm": tre_or, "tre_mm": float(e.mean()), "tre_balls_mm": float(e[:n_balls].mean()),
                             "tre_clips_mm": float(e[n_balls:].mean()),
                             "success": float(e.mean() <= tre_or + SUCCESS_MARGIN_MM),
                             "rre_deg": rre_deg(T, T_or), "rte_mm": rte_mm(T, T_or), **ex})
        torch.cuda.empty_cache()
    return rows


def ircad_unit(crop, a, cfg, pcfg):
    rows = []
    for p in deformed_ircad.load(crop, "v2")[: 2 if a.smoke else None]:
        tgt, src, G, A, B, s = inputs(p, "given", pcfg.normal_k, np.random.default_rng(0))
        pair = Pair(subject=f"d2_ircad{crop}/liver{p.liver}", view=p.idx, P=tgt.xyz, nP=tgt.normals, G=G, Q=src.xyz,
                    nQ=src.normals, pix=np.zeros((len(tgt), 2), np.int32), rgb=np.zeros((1, 1, 3), np.uint8),
                    K=np.eye(3), meta={})
        T_or = kabsch(pair.P.astype(np.float64), G.astype(np.float64))
        for arm, (T, ex) in arms(pair, a.desc, cfg).items():
            pose_rmse = float(np.sqrt(((apply(T, pair.P) - apply(T_or, pair.P)) ** 2).sum(1).mean()))
            R, t = src_to_tgt(T, A, B, s)
            rows.append({"crop": crop, "idx": p.idx, "liver": p.liver, "arm": arm, "success": float(pose_rmse < 10.0),
                         **deformed_ircad.paper_metrics(R, t, p), **ex})
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["depoll", "ircad_v2"])
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--run", default="d2")
    ap.add_argument("--poses", type=int, default=20)
    ap.add_argument("--units", nargs="+", default=None, help="states (depoll) or crops (ircad_v2)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    out = Path("outputs/eval/d2") / f"{a.data}_{re.sub(r'[^A-Za-z0-9_.+-]', '_', a.run)}"
    out.mkdir(parents=True, exist_ok=True)
    if a.data == "depoll":
        ref = depoll.reference_mesh()
        units = depoll.states()[:1] if a.smoke else [s for s in depoll.states() if a.units is None or s in a.units]
    else:
        units = [5] if a.smoke else [int(u) for u in (a.units or deformed_ircad.CROPS)]
    for u in units:
        f = out / f"{u}.parquet"
        if f.exists() and not a.smoke:
            print(u, "already done", flush=True)
            continue
        rows = depoll_unit(u, a, cfg, pcfg, ref) if a.data == "depoll" else ircad_unit(u, a, cfg, pcfg)
        if a.smoke:
            df = pd.DataFrame(rows)
            print(df.groupby("arm")[["n_hyp", "from_seed", "score_mm"]].first().to_string() if "n_hyp" in df else df.head())
            print("smoke ok: no metrics reported")
            return
        pd.DataFrame(rows).to_parquet(f)
        print(u, "done", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in sorted(out.glob("*.parquet"))], ignore_index=True)
    df.to_parquet(out.parent / f"{out.name}.parquet")
    key = ["cond", "arm"] if a.data == "depoll" else ["crop", "arm"]
    print(df.groupby(key)[["tre_mm", "success"]].agg({"tre_mm": "median", "success": "mean"}).round(2).to_string())


if __name__ == "__main__":
    main()

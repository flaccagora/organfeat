"""D: what breaks DePoLL? (post hoc diagnostic; docs/protocol.md, 2026-10-28).

    sbatch -J depoll_dec -t 06:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=depoll scripts/leonardo/gpu_run.sbatch \
        python scripts/eval/depoll_decompose.py --run d1 --desc fpfh_r20 learned:<A_GEO> ... random

The same visible region (the footprint of each state's video reconstruction) and the same target
(the porcine reference CT, sampled as configs/points.yaml), with three partial surfaces:
  C1  shape only: the reference surface itself, resampled independently, cropped to the footprint
      (nearest reference points of the oracle-aligned reconstruction, dilated 6 mm); no deformation,
      ground truth = identity, oracle TRE 0
  C2  + real deformation: the state's intraoperative CT surface (surface_CT.ply, reconstruction
      frame), points within 5 mm of the reconstruction
  C3  + reconstruction quality: the video reconstruction (as depoll_whole_eval.py, variant "full")
Poses, pipelines and metrics as scripts/eval/depoll_whole_eval.py (bounded / so3; frozen + match_ext).
-> outputs/eval/depoll_decompose/<run>/state<NN>.parquet, merged into <run>.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "scripts/eval")
from depoll_whole_eval import SUCCESS_MARGIN_MM, bounded_pose, recon_points

from ofeat._vendor.geometry import kabsch
from ofeat._vendor.io import read_ply
from ofeat.data import depoll
from ofeat.data.points import Cloud, ct_cloud, random_rigid, to_input
from ofeat.data.points import load_config as points_config
from ofeat.eval.descriptors import describe
from ofeat.eval.metrics import apply, rre_deg, rte_mm
from ofeat.eval.pairs import Pair
from ofeat.match import load_config, register
from ofeat.match_ext import register_prior
from ofeat.mesh import Mesh


def cap(c: Cloud, n: int, rng) -> Cloud:
    if len(c) <= n:
        return c
    i = np.sort(rng.choice(len(c), n, replace=False))
    return Cloud(xyz=c.xyz[i], normals=c.normals[i])


def conditions(state: str, ref: Mesh, rec_x: np.ndarray, rec_n: np.ndarray, T_or0: np.ndarray, pcfg) -> dict:
    """{name: (partial Cloud, T0 = the partial's frame -> reference CT frame for the bounded init)}."""
    rng = np.random.default_rng([int(state), 7])
    # C1: independent resampling of the reference surface, cropped to the reconstruction's footprint
    dense = ct_cloud(ref, pcfg, rng)
    tree = cKDTree(dense.xyz)
    foot = np.unique(tree.query(apply(T_or0, rec_x))[1])
    near = tree.query_ball_point(dense.xyz[foot], r=6.0)
    keep = np.unique(np.concatenate([np.asarray(k, int) for k in near])) if len(near) else foot
    c1 = Cloud(xyz=dense.xyz[keep], normals=dense.normals[keep])
    # C2: the intraoperative CT surface (reconstruction frame), within 5 mm of the reconstruction
    v, _, f = read_ply(depoll.root() / state / "surfaces" / "surface_CT.ply")
    sct = ct_cloud(Mesh(v, f, mesh_id=f"depoll/{state}/surface_CT"), pcfg, rng)
    d = cKDTree(rec_x).query(sct.xyz)[0]
    c2 = Cloud(xyz=sct.xyz[d < 5.0], normals=sct.normals[d < 5.0])
    c3 = Cloud(xyz=rec_x.astype(np.float32), normals=rec_n.astype(np.float32))
    return {"C1_shape": (cap(c1, pcfg.max_partial, rng), np.eye(4)),
            "C2_deform": (cap(c2, pcfg.max_partial, rng), T_or0),
            "C3_recon": (cap(c3, pcfg.max_partial, rng), T_or0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--run", default="d1")
    ap.add_argument("--inits", nargs="+", default=["bounded", "so3"])
    ap.add_argument("--poses", type=int, default=20)
    ap.add_argument("--states", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    out = Path("outputs/eval/depoll_decompose") / re.sub(r"[^A-Za-z0-9_.+-]", "_", a.run)
    out.mkdir(parents=True, exist_ok=True)
    ref = depoll.reference_mesh()
    todo = depoll.states()[:1] if a.smoke else [s for s in depoll.states() if a.states is None or s in a.states]
    for state in todo:
        if not a.smoke and (out / f"state{state}.parquet").exists():
            print("state", state, "already done", flush=True)
            continue
        rows = []
        pairs_m = depoll.state_markers(state)
        ref_mk = np.vstack([pairs_m[k].ref for k in pairs_m])
        rec_mk = np.vstack([pairs_m[k].deformed for k in pairs_m])
        n_balls = len(pairs_m["balls"].ref)
        ct = ct_cloud(ref, pcfg, np.random.default_rng(1000 + int(state)))  # the R3a / S4 target
        T_or0 = kabsch(rec_mk, ref_mk)
        rec_x, rec_n = recon_points(state, pcfg, np.random.default_rng([int(state), 0]))  # as R3a "full"
        for cond, (pc, T0) in conditions(state, ref, rec_x, rec_n, T_or0, pcfg).items():
            mk_part = ref_mk if cond == "C1_shape" else rec_mk  # markers in the partial's own frame
            for init in a.inits:
                for k in range(2 if a.smoke else a.poses):
                    prng = np.random.default_rng([int(state), init == "so3", k])
                    if init == "so3":
                        p_in, A = to_input(pc, random_rigid(prng))
                        q_in, B = to_input(ct, random_rigid(prng))
                    else:
                        A = bounded_pose(prng, apply(T0, pc.xyz)) @ T0
                        p_in = Cloud(xyz=apply(A, pc.xyz).astype(np.float32),
                                     normals=(pc.normals @ A[:3, :3].T).astype(np.float32))
                        q_in, B = ct, np.eye(4)
                    mk_src, mk_tgt = apply(A, mk_part), apply(B, ref_mk)
                    T_or = kabsch(mk_src, mk_tgt)
                    tre_or = float(np.linalg.norm(apply(T_or, mk_src) - mk_tgt, axis=1).mean())
                    pair = Pair(subject=f"depoll_dec/{state}/{cond}/{init}", view=k, P=p_in.xyz, nP=p_in.normals,
                                G=np.zeros_like(p_in.xyz), Q=q_in.xyz, nQ=q_in.normals,
                                pix=np.zeros((len(pc), 2), np.int32), rgb=np.zeros((1, 1, 3), np.uint8), K=np.eye(3),
                                meta={})
                    base = {"state": state, "cond": cond, "init": init, "pose": k, "n_partial": len(pc),
                            "tre_oracle_mm": tre_or}
                    for name in a.desc:
                        fs, ft = describe(name, pair, "src"), describe(name, pair, "tgt")
                        for pipe in ("frozen", "match_ext") if init == "bounded" else ("frozen",):
                            reg = (register(pair.P, fs, pair.Q, ft, pair.nQ, cfg) if pipe == "frozen" else
                                   register_prior(pair.P, fs, pair.Q, ft, pair.nQ, cfg, max_angle=75.0))
                            e = np.linalg.norm(apply(reg.T, mk_src) - mk_tgt, axis=1)
                            rows.append({**base, "desc": name, "pipeline": pipe, "tre_mm": float(e.mean()),
                                         "tre_balls_mm": float(e[:n_balls].mean()), "tre_clips_mm": float(e[n_balls:].mean()),
                                         "success": float(e.mean() <= tre_or + SUCCESS_MARGIN_MM),
                                         "rre_deg": rre_deg(reg.T, T_or), "rte_mm": rte_mm(reg.T, T_or)})
                    if a.smoke:
                        print({"state": state, "cond": cond, "init": init, "pose": k, "n_partial": len(pc),
                               "tre_oracle_mm": round(tre_or, 1)}, flush=True)
            torch.cuda.empty_cache()
        if not a.smoke:
            pd.DataFrame(rows).to_parquet(out / f"state{state}.parquet")
        print("state", state, "done", flush=True)
    if a.smoke:
        print("smoke ok: no metrics reported")
        return
    df = pd.concat([pd.read_parquet(f) for f in sorted(out.glob("state*.parquet"))], ignore_index=True)
    df.to_parquet(out.parent / f"{out.name}.parquet")
    print(df.groupby(["cond", "init", "pipeline", "desc"])[["tre_mm", "tre_oracle_mm", "success"]].median()
          .round(2).to_string())


if __name__ == "__main__":
    main()

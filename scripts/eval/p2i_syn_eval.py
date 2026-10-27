"""P2I-LReg synthetic test split vs Self-P2IR (arXiv 2504.15152, Table I): post-hoc external benchmark
(docs/ext_p2i_syn.md). Nothing is selected on it.

    sbatch -J p2i_syn -t 04:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=p2i_lreg scripts/leonardo/gpu_run.sbatch \
        python scripts/eval/p2i_syn_eval.py --cases 01 02 03 04 05 06 07 --run g1 --desc fpfh_r20 ...

The authors' synthetic frames (<case>/test_syn.txt: 2,800 frames over all 21 cases) are renders of each
case's preoperative model: noise-free visible liver points in the camera frame with the exact camera pose.
No RGB is shipped, so only geometry-only descriptors run. The locked cases' synthetic frames need
OFEAT_UNLOCK_EVAL; they are not the locked real frames (docs/protocol.md deviations, 2026-10-27).
Per frame:
  - partial: the rendered points at 3 mm voxels (cap 8192), kNN-PCA normals facing the camera (origin)
  - CT: the preoperative model (mm), sampled as configs/points.yaml (same seed as p2i_eval.py)
  - each side its own random rigid transform, then the fixed pipeline and pair_metrics; ground truth
    G = camPose(P), exact. Self-P2IR's IR (tau 1 cm) and FMR (IR > 5 %) are inlier_ratio@10 and fmr.
    Self-P2IR computes IR on its top-k scored correspondences (k = 250 ... 2000): ir_top250 / ir_top1000 rank
    the mutual matches by ratio-test confidence.
-> outputs/eval/p2i_syn/<run>.parquet
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ofeat.data import p2i_lreg
from ofeat.data.points import Cloud, _normals, ct_cloud, random_rigid, to_input, voxel_pick
from ofeat.data.points import load_config as points_config
from ofeat.eval.descriptors import describe
from ofeat.eval.metrics import apply, correct, pair_metrics
from ofeat.eval.pairs import Pair
from ofeat.match import load_config, register


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=None, help="default: all 21")
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--run", default="final")
    ap.add_argument("--max-frames", type=int, default=None, help="per case (smoke)")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    cases = a.cases or sorted(p2i_lreg.cases("dev") + p2i_lreg.cases("locked"))
    rows = []
    for case in cases:
        mesh = p2i_lreg.model(case)
        ct = ct_cloud(mesh, pcfg, np.random.default_rng(1000 + int(case)))
        for k, frame in enumerate(p2i_lreg.syn_frames(case)[: a.max_frames]):
            rng = np.random.default_rng([int(case), int(frame)])
            x, T_c2m = p2i_lreg.load_syn(case, frame)
            idx = voxel_pick(x, pcfg.voxel_mm, rng)
            if len(idx) > pcfg.max_partial:
                idx = np.sort(rng.choice(idx, pcfg.max_partial, replace=False))
            x = x[idx]
            base = {"case": case, "frame": frame, "dev": case in p2i_lreg.cases("dev"),
                    "p2ilf_derived": case in p2i_lreg.P2ILF_DERIVED, "n_partial": len(x), "n_ct": len(ct)}
            if len(x) < pcfg.min_partial:
                rows.append({**base, "skipped": True})
                continue
            pc = Cloud(xyz=x.astype(np.float32), normals=_normals(x, pcfg.normal_k, viewpoint=np.zeros(3)).astype(np.float32))
            p_in, _ = to_input(pc, random_rigid(rng))
            q_in, B_in = to_input(ct, random_rigid(rng))
            G = apply(B_in @ T_c2m, x).astype(np.float32)
            pair = Pair(subject=f"p2i_syn/{case}", view=int(frame), P=p_in.xyz, nP=p_in.normals, G=G,
                        Q=q_in.xyz, nQ=q_in.normals, pix=np.zeros((len(x), 2), np.int32),
                        rgb=np.zeros((1, 1, 3), np.uint8), K=np.eye(3), meta={})
            for name in a.desc:
                t0 = time.perf_counter()
                fs = describe(name, pair, "src")
                t1 = time.perf_counter()
                ft = describe(name, pair, "tgt")
                t2 = time.perf_counter()
                reg = register(pair.P, fs, pair.Q, ft, pair.nQ, cfg)
                m = pair_metrics(pair.P, pair.G, pair.Q, reg)
                m.pop("coverage_curve")
                ok = correct(pair.G, pair.Q, reg.matches.src, reg.matches.tgt, 10.0)[np.argsort(-reg.matches.conf, kind="stable")]
                m.update({f"ir_top{k}": float(ok[:k].mean()) if len(ok) else 0.0 for k in (250, 1000)})  # Self-P2IR "samples"
                rows.append({**base, "desc": name, "skipped": False, "sec_desc_src": t1 - t0, "sec_desc_tgt": t2 - t1, **m})
            torch.cuda.empty_cache()
        print(case, "done", flush=True)
    out = Path("outputs/eval/p2i_syn")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', a.run)}.parquet")
    ok = df[~df["skipped"]]
    print(ok.groupby("desc")[["inlier_ratio@10", "fmr", "success", "rre_icp", "rte_icp"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

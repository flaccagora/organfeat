"""R1: LiverMatch retrained on deformed-IRCAD generator pairs, on the test pairs (docs/protocol.md, 2026-10-28).

    sbatch -J lm_eval -t 03:00:00 scripts/leonardo/gpu_run.sbatch python scripts/eval/livermatch_ircad_eval.py \
        --snapshot external/livermatch/snapshot/ircad_neri --split test
    (--split v2 needs OFEAT_UNLOCK_EVAL=ircad_v2; only livers 1-2 are scored: LiverMatch trains on 3-20)

Per pair:
  input   as external/livermatch/datasets/ircad_neri.py without augmentation: normalised coordinates x
          the liver's mean half-extent / 100 (LiverMatch units, mm / 100), each cloud centred
  native  LiverMatch's matches (match_pred) -> Open3D correspondence RANSAC (5 mm, n = 4, the
          upstream eva_regist_RANASC call) -> point-to-point ICP (5 mm): "LiverMatch + RANSAC-ICP", as
          Neri et al. evaluate it
  frozen  LiverMatch's per-point features through ofeat's fixed pipeline (partial -> CT, as
          scripts/eval/ircad_eval.py with init "given")
  metrics the paper's (ofeat.data.deformed_ircad.paper_metrics); for "frozen" also pair_metrics
-> outputs/eval/ircad_lm/<split>_<run>.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
LM = REPO / "external" / "livermatch"
sys.path[:0] = [str(LM / "_shims"), str(LM), str(REPO / "scripts" / "eval")]

from ircad_eval import inputs, src_to_tgt

from ofeat.data import deformed_ircad
from ofeat.data.points import load_config as points_config
from ofeat.eval.metrics import pair_metrics
from ofeat.match import load_config, register


def load_model(snapshot: Path):
    from configs.models import architectures
    from datasets.dataloader import calibrate_neighbors, collate_fn_descriptor
    from datasets.ircad_neri import IrcadNeri
    from easydict import EasyDict
    from models.framework import KPFCNN

    config = EasyDict(json.loads((snapshot / "config.json").read_text()))
    config.device = torch.device("cuda")
    config.architecture = architectures[config.model_name]
    model = KPFCNN(config).to(config.device)
    ck = min((snapshot / "checkpoints").glob("model_best_loss*.pth"))
    state = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    limits = calibrate_neighbors(IrcadNeri(config, "train", data_augmentation=False), config,
                                 collate_fn=collate_fn_descriptor)
    return model, config, limits, collate_fn_descriptor, ck.name


@torch.no_grad()
def run_livermatch(model, config, limits, collate, p):
    s = float(p.half.mean()) / 100.0
    src, tgt = (p.src * s).astype(np.float32), (p.tgt * s).astype(np.float32)
    cs, ct = src.mean(0), tgt.mean(0)
    corr = torch.from_numpy(np.stack([p.corr, np.arange(len(p.tgt))], 1).astype(np.int64))
    one_s, one_t = np.ones((len(src), 1), np.float32), np.ones((len(tgt), 1), np.float32)
    item = (src - cs, tgt - ct, one_s, one_t, np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32), corr,
            src - cs, tgt - ct, torch.ones(1))
    batch = collate([item], config, limits)
    for k, v in batch.items():
        batch[k] = [x.to(config.device) for x in v] if isinstance(v, list) else v.to(config.device)
    out = model(batch)
    feats = out["feats"].float().cpu().numpy()
    m = out["match_pred"].detach().cpu().numpy()[:, 1:].astype(np.int64)  # (K, 2) [src, tgt]
    return src - cs, tgt - ct, cs, ct, s, feats[: len(src)], feats[len(src):], m


def ransac_icp(src_c, tgt_c, m):
    import open3d as o3d

    reg = o3d.pipelines.registration
    o3d.utility.random.seed(0)
    a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_c.astype(np.float64)))
    b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_c.astype(np.float64)))
    if len(m) < 4:
        return np.eye(4)
    res = reg.registration_ransac_based_on_correspondence(
        source=a, target=b, corres=o3d.utility.Vector2iVector(m.astype(np.int32)), max_correspondence_distance=0.05,
        estimation_method=reg.TransformationEstimationPointToPoint(False), ransac_n=4)
    icp = reg.registration_icp(a, b, 0.05, res.transformation, reg.TransformationEstimationPointToPoint(),
                               reg.ICPConvergenceCriteria(max_iteration=50))
    return np.asarray(icp.transformation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(LM / "snapshot" / "ircad_neri"))
    ap.add_argument("--split", default="test", choices=["test", "v2"])
    ap.add_argument("--crops", nargs="+", type=int, default=list(deformed_ircad.CROPS))
    ap.add_argument("--run", default="final")
    ap.add_argument("--max-pairs", type=int, default=None)
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    model, config, limits, collate, ckname = load_model(Path(a.snapshot))
    print("checkpoint", ckname, "neighborhood limits", limits, flush=True)
    rows = []
    for crop in a.crops:
        pairs = [p for p in deformed_ircad.load(crop, a.split) if p.liver <= 2][: a.max_pairs]
        for p in pairs:
            src_c, tgt_c, cs, ct, s, f_src, f_tgt, m = run_livermatch(model, config, limits, collate, p)
            base = {"crop": crop, "idx": p.idx, "liver": p.liver, "n_lm_matches": len(m), "ckpt": ckname}
            if len(m):  # LiverMatch match correctness: matched source point within 10 mm of the true one
                d = np.linalg.norm((p.src[m[:, 0]] - p.src[p.corr[m[:, 1]]]) * p.half, axis=1)
                base["lm_inlier_ratio@10"] = float((d < 10).mean())
            T = ransac_icp(src_c, tgt_c, m)  # centred src (mm / 100) -> centred tgt
            R = T[:3, :3]
            t = (T[:3, 3] + ct - R @ cs) / s
            rows.append({**base, "pipeline": "livermatch_ransac_icp", **deformed_ircad.paper_metrics(R, t, p)})
            tgt, src, G, A, B, s_mm = inputs(p, "given", pcfg.normal_k, np.random.default_rng(0))
            reg = register(tgt.xyz, f_tgt, src.xyz, f_src, src.normals, cfg)
            pm = pair_metrics(tgt.xyz, G, src.xyz, reg)
            pm.pop("coverage_curve")
            R2, t2 = src_to_tgt(reg.T, A, B, s_mm)
            rows.append({**base, "pipeline": "livermatch_feats_frozen", **pm, **deformed_ircad.paper_metrics(R2, t2, p)})
            torch.cuda.empty_cache()
        print("crop", crop, "done", flush=True)
    out = REPO / "outputs" / "eval" / "ircad_lm"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{a.split}_{re.sub(r'[^A-Za-z0-9_.+-]', '_', a.run)}.parquet")
    g = df.groupby(["crop", "pipeline"])
    summ = g[["tre_mm", "mae_r_deg", "mae_t_mm"]].mean()
    summ["tre_sd_pooled"] = np.sqrt(g["tre_sq_mm2"].mean() - summ["tre_mm"] ** 2)
    summ["lm_ir10"] = g["lm_inlier_ratio@10"].mean()
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()

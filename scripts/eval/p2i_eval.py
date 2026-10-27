"""P2I-LReg evaluation (docs/protocol.md section 8, docs/p2i_audit.md).

    dev (format / scale / camera debugging only; never selection):
        sbatch -J p2i_dev -t 02:00:00 scripts/leonardo/gpu_run.sbatch python scripts/eval/p2i_eval.py --split dev --desc fpfh_r20
    LOCKED, once at S4:
        sbatch -J p2i_eval -t 08:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=p2i_lreg scripts/leonardo/gpu_run.sbatch \
            python scripts/eval/p2i_eval.py --split locked --desc fpfh_r20 learned:<A> learned:<A>:geo learned:<B> ...

Per real frame:
  - partial cloud: the provided reconstruction (as given) at the liver-mask pixels, minus specular
    pixels, one point per 3 mm voxel. Normals face the viewing side (-z).
  - CT: the preoperative model (mm), sampled as configs/points.yaml.
  - registration: the fixed pipeline (partial -> CT).
  - projection: the CT model, moved into the partial frame by T^-1 and projected with the frame's
    fitted affine camera (ofeat.data.p2i_lreg.fit_camera), filled as the union of projected triangles.
  - metrics: Dice and mean symmetric boundary distance (px) vs the annotated mask; the ceiling (the
    cloud itself projected the same way); Chamfer from the partial points to the posed model surface
    (consistency only).
-> outputs/eval/p2i/<split>_<run>.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from ofeat.data import p2i_lreg
from ofeat.data.points import Cloud, _normals, ct_cloud, load_config as points_config, random_rigid, to_input
from ofeat.data.points import voxel_pick
from ofeat.eval.descriptors import describe
from ofeat.eval.metrics import apply
from ofeat.eval.pairs import Pair
from ofeat.features.lift import specular_mask
from ofeat.match import load_config, register


def partial_from_frame(fr: p2i_lreg.P2IFrame, cfg, rng) -> Cloud:
    keep = fr.mask & ~specular_mask(fr.rgb) & np.isfinite(fr.cloud).all(-1)
    v, u = np.nonzero(keep)
    x = fr.cloud[v, u]
    idx = voxel_pick(x, cfg.voxel_mm, rng)
    if len(idx) > cfg.max_partial:
        idx = np.sort(rng.choice(idx, cfg.max_partial, replace=False))
    x, u, v = x[idx], u[idx], v[idx]
    view = x.mean(0) - np.array([0.0, 0.0, 1e4])  # viewing side: -z
    n = _normals(x, cfg.normal_k, viewpoint=view)
    return Cloud(xyz=x.astype(np.float32), normals=n.astype(np.float32), pix=np.stack([u, v], 1).astype(np.int32))


def project_mask(verts: np.ndarray, faces: np.ndarray, A: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    uv = np.c_[verts, np.ones(len(verts))] @ A.T
    m = np.zeros(shape, np.uint8)
    tri = np.round(uv[faces]).astype(np.int32)
    tri = tri[(np.abs(tri) < 1e5).all((1, 2))]
    cv2.fillPoly(m, list(tri), 1)
    return m.astype(bool)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def boundary_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean symmetric distance (px) between the two mask boundaries."""
    def edge(m):
        return m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    ea, eb = edge(a), edge(b)
    if not ea.any() or not eb.any():
        return float("nan")
    da = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)
    return float(0.5 * (da[ea].mean() + db[eb].mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "locked"])
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--run", default="final")
    ap.add_argument("--max-frames", type=int, default=None, help="per case (smoke)")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    rows = []
    for case in p2i_lreg.cases(a.split):
        mesh = p2i_lreg.model(case)
        ct = ct_cloud(mesh, pcfg, np.random.default_rng(1000 + int(case)))
        for k, frame in enumerate(p2i_lreg.frames(case)[: a.max_frames]):
            rng = np.random.default_rng([int(case), k])
            fr = p2i_lreg.load_frame(case, frame)
            pc = partial_from_frame(fr, pcfg, rng)
            A_cam = p2i_lreg.fit_camera(fr)
            vv, uu = np.nonzero(fr.mask)
            # ceiling: the frame's own cloud projected with the same fitted camera, splatted and closed
            uvc = np.round(np.c_[fr.cloud[vv, uu], np.ones(len(vv))] @ A_cam.T).astype(int)
            okc = (uvc[:, 0] >= 0) & (uvc[:, 0] < fr.mask.shape[1]) & (uvc[:, 1] >= 0) & (uvc[:, 1] < fr.mask.shape[0])
            ceil = np.zeros(fr.mask.shape, np.uint8)
            ceil[uvc[okc, 1], uvc[okc, 0]] = 1
            ceil = cv2.morphologyEx(ceil, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
            base = {"case": case, "frame": frame, "p2ilf_derived": case in p2i_lreg.P2ILF_DERIVED,
                    "n_partial": len(pc), "n_ct": len(ct),
                    "dice_ceiling": dice(ceil, fr.mask), "bd_ceiling_px": boundary_dist(ceil, fr.mask)}
            if len(pc) < pcfg.min_partial:
                rows.append({**base, "skipped": True})
                continue
            p_in, A_in = to_input(pc, random_rigid(rng))
            q_in, B_in = to_input(ct, random_rigid(rng))
            pair = Pair(subject=f"p2i/{case}", view=k, P=p_in.xyz, nP=p_in.normals, G=np.zeros_like(p_in.xyz),
                        Q=q_in.xyz, nQ=q_in.normals, pix=pc.pix, rgb=fr.rgb, K=fr.K, meta={})
            for name in a.desc:
                reg = register(pair.P, describe(name, pair, "src"), pair.Q, describe(name, pair, "tgt"), pair.nQ, cfg)
                # CT model (world mm) -> CT input (B_in) -> partial input (T^-1) -> partial frame (A_in^-1)
                M = np.linalg.inv(A_in) @ np.linalg.inv(reg.T) @ B_in
                posed = apply(M, mesh.verts)
                pred = project_mask(posed, mesh.faces, A_cam, fr.mask.shape)
                d_ch = cKDTree(posed).query(pc.xyz)[0]
                rows.append({**base, "desc": name, "dice": dice(pred, fr.mask), "bd_px": boundary_dist(pred, fr.mask),
                             "chamfer_partial_to_model_mm": float(d_ch.mean()), "n_matches": len(reg.matches.src),
                             "skipped": False})
            torch.cuda.empty_cache()
        print(case, "done", flush=True)
    out = Path("outputs/eval/p2i")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{a.split}_{re.sub(r'[^A-Za-z0-9_.+-]', '_', a.run)}.parquet")
    ok = df[~df["skipped"]]
    print(ok.groupby(["desc", "case"])[["dice", "bd_px", "dice_ceiling", "chamfer_partial_to_model_mm"]].mean()
          .groupby(level=0).mean().round(3).to_string())


if __name__ == "__main__":
    main()

"""LOCKED DePoLL evaluation (docs/protocol.md section 8): run once, at S4, for the final descriptors.

    sbatch -J depoll_eval -t 04:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=depoll scripts/leonardo/gpu_run.sbatch \
        python scripts/eval/depoll_eval.py --desc fpfh_r<R> sonata learned:<ckpt> learned:<ckpt>:geo
    ... --smoke : state 01, first 2 views, pipeline sanity only (no metrics printed or saved)

Per state: the coloured stereo reconstruction is Poisson-meshed and rendered from 24 virtual views
(vertex colours). Liver pixels come from the frozen DINOv2 probe (patch mask, upsampled) and must
have rendered depth; tools are absent. Partial cloud: ofeat.data.points.partial_cloud (camera frame).
CT: the reference surface (surfaceFull.ply), sampled as in configs/points.yaml.
Ground truth: the 60 marker pairs (reference CT frame <-> reconstruction frame). Per view the
markers are expressed in the partial and CT input frames:
  TRE        marker error after our T
  oracle TRE marker error after the best rigid fit of the markers (deformation floor)
  success    TRE <= oracle TRE + 10 mm
  RRE / RTE  rotation / translation error of T vs the oracle rigid transform
-> outputs/eval/depoll/<run>.parquet (per view and descriptor) + summary json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ofeat._vendor.geometry import kabsch
from ofeat.data import depoll
from ofeat.data.depoll_views import coloured_mesh, render_colours, views
from ofeat.data.liver_probe import load_probe, p_liver, patch_mask
from ofeat.data.points import ct_cloud, load_config as points_config, partial_cloud, random_rigid, to_input
from ofeat.eval.descriptors import describe, dino_tokens
from ofeat.eval.metrics import apply, rre_deg, rte_mm
from ofeat.eval.pairs import Pair
from ofeat.match import load_config, register

SUCCESS_MARGIN_MM = 10.0


def liver_pixels(rgb: np.ndarray, depth: np.ndarray, probe) -> np.ndarray:
    tok = dino_tokens(rgb).cpu().numpy()
    h, w = tok.shape[:2]
    cover = cv2.resize((depth > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.9
    m = patch_mask(p_liver(probe, tok)) & cover
    return cv2.resize(m.astype(np.uint8), rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--run", default="final")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg, pcfg = load_config(), points_config()
    probe = load_probe()
    ref = depoll.reference_mesh()
    rows = []
    for state in depoll.states()[:1] if a.smoke else depoll.states():
        mesh, colours = coloured_mesh(state)
        pairs_m = depoll.state_markers(state)
        ref_mk = np.vstack([pairs_m[k].ref for k in pairs_m])  # reference CT frame
        rec_mk = np.vstack([pairs_m[k].deformed for k in pairs_m])  # reconstruction frame
        rng = np.random.default_rng(int(state))
        ct = ct_cloud(ref, pcfg, np.random.default_rng(1000 + int(state)))
        for vi, cam in enumerate(views(mesh, rng)[: 2 if a.smoke else None]):
            rgb, depth = render_colours(mesh, colours, cam)
            mask = liver_pixels(rgb, depth, probe)
            pc = partial_cloud(rgb, depth, np.zeros(depth.shape, bool), cam.K, cam.E, pcfg, rng, mask=mask)
            if len(pc) < pcfg.min_partial:
                rows.append({"state": state, "view": vi, "n_partial": len(pc), "skipped": True})
                continue
            p_in, A = to_input(pc, random_rigid(rng))
            q_in, B = to_input(ct, random_rigid(rng))
            # markers: reconstruction frame -> camera (E) -> partial input (A); reference -> CT input (B)
            mk_src = apply(A @ cam.E, rec_mk)
            mk_tgt = apply(B, ref_mk)
            T_or = kabsch(mk_src, mk_tgt)
            tre_or = float(np.linalg.norm(apply(T_or, mk_src) - mk_tgt, axis=1).mean())
            pair = Pair(subject=f"depoll/{state}", view=vi, P=p_in.xyz, nP=p_in.normals, G=np.zeros_like(p_in.xyz),
                        Q=q_in.xyz, nQ=q_in.normals, pix=pc.pix, rgb=rgb, K=cam.K, meta={})
            for name in a.desc:
                reg = register(pair.P, describe(name, pair, "src"), pair.Q, describe(name, pair, "tgt"), pair.nQ, cfg)
                tre = float(np.linalg.norm(apply(reg.T, mk_src) - mk_tgt, axis=1).mean())
                rows.append({"state": state, "view": vi, "desc": name, "n_partial": len(pc), "n_ct": len(ct),
                             "tre_mm": tre, "tre_oracle_mm": tre_or,
                             "success": float(tre <= tre_or + SUCCESS_MARGIN_MM),
                             "rre_deg": rre_deg(reg.T, T_or), "rte_mm": rte_mm(reg.T, T_or),
                             "n_matches": len(reg.matches.src), "skipped": False})
            if a.smoke:
                print({"state": state, "view": vi, "n_partial": len(pc), "liver_px": int(mask.sum()),
                       "tre_oracle_mm": round(tre_or, 1)}, flush=True)
        torch.cuda.empty_cache()
    if a.smoke:
        print("smoke ok: no metrics reported (locked set)")
        return
    out = Path("outputs/eval/depoll")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', a.run)}.parquet")
    ok = df[~df["skipped"]]
    summ = ok.groupby(["desc", "state"])[["success", "tre_mm", "tre_oracle_mm", "rre_deg"]].mean() \
        .groupby(level=0).mean().round(3)
    (out / f"{a.run}_summary.json").write_text(json.dumps(summ.to_dict(orient="index"), indent=1))
    print(summ.to_string())


if __name__ == "__main__":
    main()

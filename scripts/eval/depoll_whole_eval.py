"""R3a / R5a: DePoLL with whole per-state reconstructions (post hoc; DePoLL was spent at S4).
Protocol pre-registered in docs/protocol.md (2026-10-28); the protocol of Neri et al. 2025 and
Huang et al. 2025, which register the whole reconstruction rather than our S4 views.

    sbatch -J depoll_whole -t 04:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=depoll scripts/leonardo/gpu_run.sbatch \
        python scripts/eval/depoll_whole_eval.py --run r3a --desc fpfh_r20 learned:<A_GEO> ... random
    ... --smoke : state 01, 2 poses, mechanics only (no metrics)

Per state (13):
  partial  variant "full": video_reconstruction.ply as provided (reconstruction frame), 3 mm voxels,
           cap 8,192; variant "liver": the union of the probe-liver points of the 24 S4 views
           (same views as scripts/eval/depoll_eval.py). Normals: kNN PCA, oriented like the file's
           normals (estimated and turned to a far point where the file has none, as S4).
  CT       the reference surface, sampled as configs/points.yaml (same seed as S4).
  images   (R5a; learned:<multimodal ckpt> without :geo) DINOv2 PCA-64 lifted from the 24 S4 renders
           of the coloured reconstruction, averaged per point over the views that see it.
  poses    20 per state and setting: "bounded" = the oracle-aligned partial perturbed by Euler angles
           U(0, 45) deg about its centroid and a translation U(-50, 50) mm per axis (the generator
           range of Neri et al.); "so3" = independent random rigid transforms of both clouds.
  pipelines  the frozen one (both settings); match_ext (k-NN + <= 75 deg prior; "bounded" only).
  metrics  marker TRE (all 60; balls; clips), oracle TRE (best rigid fit of the markers), success
           (TRE <= oracle + 10 mm), RRE / RTE vs the oracle.
-> outputs/eval/depoll_whole/<run>/state<NN>.parquet (resumable), merged into outputs/eval/depoll_whole/<run>.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from ofeat._vendor.geometry import kabsch
from ofeat._vendor.io import read_ply
from ofeat.data import depoll
from ofeat.data.depoll_views import coloured_mesh, render_colours, views
from ofeat.data.liver_probe import load_probe, p_liver, patch_mask
from ofeat.data.points import Cloud, _normals, ct_cloud, partial_cloud, random_rigid, to_input, voxel_pick
from ofeat.data.points import load_config as points_config
from ofeat.eval.descriptors import describe, dino_tokens, extractor
from ofeat.eval.metrics import apply, rre_deg, rte_mm
from ofeat.eval.pairs import Pair
from ofeat.match import load_config, register
from ofeat.match_ext import register_prior

SUCCESS_MARGIN_MM = 10.0
DEPTH_TOL_MM = 3.0


def recon_points(state: str, pcfg, rng) -> tuple[np.ndarray, np.ndarray]:
    """Whole reconstruction at 3 mm voxels (cap) with oriented kNN-PCA normals, reconstruction frame."""
    import open3d as o3d

    pts, nrm, _ = read_ply(depoll.root() / state / "surfaces" / "video_reconstruction.ply")
    pts = pts.astype(np.float64)
    if nrm is None or np.linalg.norm(nrm, axis=1).min() <= 1e-6:  # states 09-13: as S4 (depoll_views)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        c = pts.mean(0)
        pc.orient_normals_towards_camera_location(c + 1000.0 * np.linalg.svd(pts - c, full_matrices=False)[2][-1])
        nrm = np.asarray(pc.normals)
    idx = voxel_pick(pts, pcfg.voxel_mm, rng)
    if len(idx) > pcfg.max_partial:
        idx = np.sort(rng.choice(idx, pcfg.max_partial, replace=False))
    x = pts[idx]
    return x, _normals(x, pcfg.normal_k, reference=np.asarray(nrm)[idx])


def lifted_vis(x: np.ndarray, renders: list, vc) -> tuple[np.ndarray, np.ndarray]:
    """(N, 64) DINO-PCA-64 averaged over the renders where each point is visible, (N,) seen."""
    acc, cnt = np.zeros((len(x), 64)), np.zeros(len(x))
    for cam, rgb, depth in renders:
        xc = apply(cam.E, x)
        z = xc[:, 2]
        uv = xc @ cam.K.T
        u = np.round(uv[:, 0] / np.maximum(z, 1e-6)).astype(int)
        v = np.round(uv[:, 1] / np.maximum(z, 1e-6)).astype(int)
        H, W = depth.shape
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ok[ok] &= np.abs(depth[v[ok], u[ok]] - z[ok]) < DEPTH_TOL_MM
        if not ok.any():
            continue
        img = torch.as_tensor(rgb, device="cuda").permute(2, 0, 1)[None].float() / 255
        tok = vc.tokens(img)[0]
        pix = torch.as_tensor(np.stack([u[ok], v[ok]], 1), device="cuda")
        acc[ok] += vc.project(tok, pix, (W, H)).cpu().numpy()
        cnt[ok] += 1
    seen = cnt > 0
    acc[seen] /= cnt[seen, None]
    return acc.astype(np.float32), seen


def liver_pixels(rgb: np.ndarray, depth: np.ndarray, probe) -> np.ndarray:
    """Copied from scripts/eval/depoll_eval.py: probe-liver patches with rendered depth, at pixel size."""
    tok = dino_tokens(rgb).cpu().numpy()
    h, w = tok.shape[:2]
    cover = cv2.resize((depth > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.9
    m = patch_mask(p_liver(probe, tok)) & cover
    return cv2.resize(m.astype(np.uint8), rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)


def liver_union(renders: list, probe, pcfg, rng) -> tuple[np.ndarray, np.ndarray]:
    """Union of the probe-liver points of the S4 views, reconstruction frame, 3 mm voxels (cap); normals
    from each view (facing its camera), re-estimated on the union and oriented like them."""
    pts, nrm = [], []
    for cam, rgb, depth in renders:
        pc = partial_cloud(rgb, depth, np.zeros(depth.shape, bool), cam.K, cam.E, pcfg, rng,
                           mask=liver_pixels(rgb, depth, probe))
        pts.append(pc.gt.astype(np.float64))
        nrm.append(pc.normals @ np.linalg.inv(cam.E)[:3, :3].T)
    allp, alln = np.concatenate(pts), np.concatenate(nrm)
    idx = voxel_pick(allp, pcfg.voxel_mm, rng)
    if len(idx) > pcfg.max_partial:
        idx = np.sort(rng.choice(idx, pcfg.max_partial, replace=False))
    x = allp[idx]
    if len(x) < pcfg.normal_k:
        return x, np.zeros_like(x)
    return x, _normals(x, pcfg.normal_k, reference=alln[idx])


def bounded_pose(rng, x_aligned: np.ndarray) -> np.ndarray:
    """Perturbation of the oracle-aligned partial: Euler U(0, 45) deg about its centroid + U(-50, 50) mm."""
    R = Rotation.from_euler("xyz", rng.uniform(0, 45, 3), degrees=True).as_matrix()
    c = x_aligned.mean(0)
    P = np.eye(4)
    P[:3, :3], P[:3, 3] = R, c - R @ c + rng.uniform(-50, 50, 3)
    return P


def descriptors(name, pair, vis, seen):
    if name.startswith("learned:") and not name.endswith(":geo"):
        ex = extractor(name[len("learned:"):])
        if ex.uses_images:
            return ex(pair.P, pair.nP, vis=vis, vis_ok=seen), describe(name, pair, "tgt")
    return describe(name, pair, "src"), describe(name, pair, "tgt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--run", default="r3a")
    ap.add_argument("--variants", nargs="+", default=["full", "liver"])
    ap.add_argument("--inits", nargs="+", default=["bounded", "so3"])
    ap.add_argument("--poses", type=int, default=20)
    ap.add_argument("--states", nargs="+", default=None, help="subset (default: all 13)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path("outputs/eval/depoll_whole") / re.sub(r"[^A-Za-z0-9_.+-]", "_", a.run)
    out.mkdir(parents=True, exist_ok=True)
    cfg, pcfg = load_config(), points_config()
    probe = load_probe()
    ref = depoll.reference_mesh()
    vc = None
    rows = []
    todo = depoll.states()[:1] if a.smoke else [s for s in depoll.states() if a.states is None or s in a.states]
    for state in todo:
        if not a.smoke and (out / f"state{state}.parquet").exists():  # resume: Open3D can segfault (S4, exit 11)
            print("state", state, "already done", flush=True)
            continue
        n_before = len(rows)
        mesh, colours = coloured_mesh(state)
        cams = views(mesh, np.random.default_rng(int(state)))  # the S4 views
        renders = [(cam, *render_colours(mesh, colours, cam)) for cam in cams]
        pairs_m = depoll.state_markers(state)
        ref_mk = np.vstack([pairs_m[k].ref for k in pairs_m])
        rec_mk = np.vstack([pairs_m[k].deformed for k in pairs_m])
        n_balls = len(pairs_m["balls"].ref)
        ct = ct_cloud(ref, pcfg, np.random.default_rng(1000 + int(state)))
        T_or0 = kabsch(rec_mk, ref_mk)  # reconstruction -> reference CT
        for variant in a.variants:
            rng = np.random.default_rng([int(state), variant == "liver"])
            if variant == "full":
                x, n = recon_points(state, pcfg, rng)
            else:
                x, n = liver_union(renders, probe, pcfg, rng)
            if len(x) < pcfg.min_partial:
                rows.append({"state": state, "variant": variant, "n_partial": len(x), "skipped": True})
                continue
            vis, seen = (None, None)
            if any(d.startswith("learned:") and not d.endswith(":geo") for d in a.desc):
                if vc is None:
                    from ofeat.models.infer import VisualChannel
                    vc = VisualChannel()
                vis, seen = lifted_vis(x, renders, vc)
            base_pc = Cloud(xyz=x.astype(np.float32), normals=n.astype(np.float32))
            for init in a.inits:
                for k in range(2 if a.smoke else a.poses):
                    prng = np.random.default_rng([int(state), variant == "liver", init == "so3", k])
                    if init == "so3":
                        p_in, A = to_input(base_pc, random_rigid(prng))
                        q_in, B = to_input(ct, random_rigid(prng))
                    else:
                        A = bounded_pose(prng, apply(T_or0, x)) @ T_or0
                        p_in = Cloud(xyz=apply(A, x).astype(np.float32), normals=(n @ A[:3, :3].T).astype(np.float32))
                        q_in, B = ct, np.eye(4)
                    mk_src, mk_tgt = apply(A, rec_mk), apply(B, ref_mk)
                    T_or = kabsch(mk_src, mk_tgt)
                    tre_or = float(np.linalg.norm(apply(T_or, mk_src) - mk_tgt, axis=1).mean())
                    pair = Pair(subject=f"depoll_whole/{state}/{variant}/{init}", view=k, P=p_in.xyz, nP=p_in.normals,
                                G=np.zeros_like(p_in.xyz), Q=q_in.xyz, nQ=q_in.normals,
                                pix=np.zeros((len(x), 2), np.int32), rgb=np.zeros((1, 1, 3), np.uint8), K=np.eye(3), meta={})
                    base = {"state": state, "variant": variant, "init": init, "pose": k, "n_partial": len(x),
                            "n_ct": len(ct), "tre_oracle_mm": tre_or, "skipped": False,
                            "vis_seen_frac": float(seen.mean()) if seen is not None else np.nan}
                    for name in a.desc:
                        fs, ft = descriptors(name, pair, vis, seen)
                        for pipe in ("frozen", "match_ext") if init == "bounded" else ("frozen",):
                            if pipe == "frozen":
                                reg = register(pair.P, fs, pair.Q, ft, pair.nQ, cfg)
                            else:
                                reg = register_prior(pair.P, fs, pair.Q, ft, pair.nQ, cfg, max_angle=75.0)
                            e = np.linalg.norm(apply(reg.T, mk_src) - mk_tgt, axis=1)
                            rows.append({**base, "desc": name, "pipeline": pipe, "tre_mm": float(e.mean()),
                                         "tre_balls_mm": float(e[:n_balls].mean()), "tre_clips_mm": float(e[n_balls:].mean()),
                                         "success": float(e.mean() <= tre_or + SUCCESS_MARGIN_MM),
                                         "rre_deg": rre_deg(reg.T, T_or), "rte_mm": rte_mm(reg.T, T_or),
                                         "n_matches": len(reg.matches.src)})
                    if a.smoke:
                        print({"state": state, "variant": variant, "init": init, "pose": k, "n_partial": len(x),
                               "tre_oracle_mm": round(tre_or, 1)}, flush=True)
            torch.cuda.empty_cache()
        if not a.smoke:
            pd.DataFrame(rows[n_before:]).to_parquet(out / f"state{state}.parquet")
        print("state", state, "done", flush=True)
    if a.smoke:
        print("smoke ok: no metrics reported")
        return
    df = pd.concat([pd.read_parquet(f) for f in sorted(out.glob("state*.parquet"))], ignore_index=True)
    df.to_parquet(out.parent / f"{out.name}.parquet")
    ok = df[~df["skipped"]]
    print(ok.groupby(["variant", "init", "pipeline", "desc"])[["tre_mm", "tre_oracle_mm", "success", "rre_deg"]]
          .median().round(2).to_string())


if __name__ == "__main__":
    main()

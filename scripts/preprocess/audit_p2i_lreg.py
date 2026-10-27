"""P2I-LReg audit (docs/protocol.md section 8), run before any descriptor sees the data.

Layout (Kaggle junezhou001/p2i-lreg, data/raw/p2i_lreg/Liver_regis/<case>/):
    readme.txt                   case tag + real image size. Only the "patientN" tag is used
                                 (P2ILF naming); other tags are never copied into outputs.
    model/*.obj                  preoperative liver model (CT)
    real/CAM_K.yml               intrinsics
    real/labels/<frame>/         labelme export: img.png (RGB), label.png (liver mask), ...
    real/liverPcds/<frame>.ply   dense reconstructed cloud (one point per pixel?)
    real/scale/<frame>.txt       per-frame scale factor
    syn/...                      Blender synthetic frames with camPose.yml (not used here)
    {train,test}_{real,syn}.txt  the authors' split

Checks per real frame: RGB and mask sizes vs readme, mask fraction, cloud size vs H x W (pixel
mapping), cloud extent inside the mask, the scale factor, whether cloud points project back to their
own pixel with the intrinsics, model extent and units.

    sbatch -J audit_p2i -c 16 --mem 64G -t 01:00:00 scripts/leonardo/cpu.sbatch python scripts/preprocess/audit_p2i_lreg.py
-> outputs/audit/p2i_lreg_frames.csv, p2i_lreg_cases.csv, overlay sheets outputs/audit/p2i_<case>.jpg
"""

from __future__ import annotations

import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ofeat.mesh import data_root

ROOT = data_root() / "raw" / "p2i_lreg" / "Liver_regis"
OUT = Path("outputs/audit")


def read_K(path: Path) -> np.ndarray:
    import yaml

    d = yaml.safe_load(path.read_text())
    first = d[sorted(d)[0]]
    k = first[0]["cam_K"] if isinstance(first, list) else first["cam_K"]
    return np.array(k, np.float64).reshape(3, 3)


def read_cloud(path: Path) -> np.ndarray:
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def read_obj_verts(path: Path) -> np.ndarray:
    vs = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    return np.array(vs, np.float64)


def case_tag(case: Path) -> str:
    first = (case / "readme.txt").read_text(errors="replace").strip().splitlines()[0].strip()
    return first if re.fullmatch(r"patient\d+", first) else "partner"  # never propagate other tags


def frame_row(args) -> dict:
    case, frame, K = args
    d = ROOT / case / "real"
    row = {"case": case, "frame": frame}
    lab = d / "labels" / frame
    img = cv2.imread(str(lab / "img.png"))
    msk = cv2.imread(str(lab / "label.png"), cv2.IMREAD_UNCHANGED)
    row["img_hw"] = None if img is None else f"{img.shape[0]}x{img.shape[1]}"
    if msk is not None:
        m = (msk.max(-1) if msk.ndim == 3 else msk) > 0
        row["mask_hw"] = f"{m.shape[0]}x{m.shape[1]}"
        row["mask_frac"] = float(m.mean())
    sc = d / "scale" / f"{frame}.txt"
    row["scale"] = float(sc.read_text().split()[0]) if sc.exists() else np.nan
    pc = d / "liverPcds" / f"{frame}.ply"
    if pc.exists() and img is not None:
        x = read_cloud(pc)
        H, W = img.shape[:2]
        row["cloud_n"] = len(x)
        row["cloud_is_dense"] = bool(len(x) == H * W)
        if len(x) == H * W:
            X = x.reshape(H, W, 3)
            v, u = np.nonzero(m) if msk is not None else np.nonzero(np.ones((H, W), bool))
            P = X[v, u]
            ok = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)
            P, u, v = P[ok], u[ok], v[ok]
            row["cloud_in_mask_valid"] = float(ok.mean())
            row["cloud_extent_in_mask"] = float(np.linalg.norm(np.ptp(P, 0))) if len(P) else np.nan
            row["z_median"] = float(np.median(P[:, 2])) if len(P) else np.nan
            if len(P):  # does each point project to its own pixel? (camera-frame cloud, intrinsics K)
                uv = P @ K.T
                pu, pv = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
                err = np.hypot(pu - u, pv - v)
                row["reproj_px_median"] = float(np.median(err))
                row["reproj_px_p90"] = float(np.quantile(err, 0.9))
        else:
            row["cloud_extent"] = float(np.linalg.norm(np.ptp(x, 0)))
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    jobs, case_rows = [], []
    for c in cases:
        d = ROOT / c
        K = read_K(d / "real" / "CAM_K.yml")
        frames = sorted(p.name for p in (d / "real" / "labels").iterdir() if p.is_dir())
        objs = sorted((d / "model").glob("*.obj"))
        mv = read_obj_verts(objs[0]) if objs else np.zeros((0, 3))
        lists = {n: (d / f"{n}.txt").read_text().split() if (d / f"{n}.txt").exists() else []
                 for n in ("train_real", "test_real")}
        case_rows.append({"case": c, "tag": case_tag(d), "n_real": len(frames),
                          "n_train_real": len(lists["train_real"]), "n_test_real": len(lists["test_real"]),
                          "n_syn": len(list((d / "syn" / "labels").glob("*.png"))) if (d / "syn").exists() else 0,
                          "fx": K[0, 0], "cx": K[0, 2], "cy": K[1, 2],
                          "model_file": objs[0].name if objs else None, "model_n_verts": len(mv),
                          "model_extent": float(np.linalg.norm(np.ptp(mv, 0))) if len(mv) else np.nan})
        jobs += [(c, f, K) for f in frames]
    with Pool(16) as pool:
        rows = pool.map(frame_row, jobs)
    fr = pd.DataFrame(rows)
    cs = pd.DataFrame(case_rows)
    fr.to_csv(OUT / "p2i_lreg_frames.csv", index=False)
    cs.to_csv(OUT / "p2i_lreg_cases.csv", index=False)
    pd.set_option("display.width", 250)
    print(cs.to_string())
    agg = fr.groupby("case").agg(frames=("frame", "size"), img=("img_hw", "first"), mask=("mask_hw", "first"),
                                 mask_frac=("mask_frac", "median"), dense=("cloud_is_dense", "mean"),
                                 valid=("cloud_in_mask_valid", "median"), extent=("cloud_extent_in_mask", "median"),
                                 z=("z_median", "median"), scale=("scale", "median"),
                                 reproj=("reproj_px_median", "median"))
    print(agg.round(3).to_string())
    for c in cases:  # overlay sheet: up to 6 frames, mask outline
        tiles = []
        for f in sorted(fr[fr.case == c].frame)[:6]:
            lab = ROOT / c / "real" / "labels" / f
            img = cv2.imread(str(lab / "img.png"))
            msk = cv2.imread(str(lab / "label.png"), cv2.IMREAD_UNCHANGED)
            if img is None or msk is None:
                continue
            m = ((msk.max(-1) if msk.ndim == 3 else msk) > 0).astype(np.uint8)
            cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnt, -1, (0, 255, 0), max(2, img.shape[1] // 400))
            tiles.append(cv2.resize(img, (480, 270)))
        if tiles:
            cv2.imwrite(str(OUT / f"p2i_{c}.jpg"), np.concatenate(tiles, 1))


if __name__ == "__main__":
    main()

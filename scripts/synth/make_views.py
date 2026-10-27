"""Render and cache synthetic views per TotalSeg split (ofeat.synth.views). CPU raycast, one process
per liver; resumable (existing files are skipped). Splits are fixed before rendering: each set
reads only its own split's subjects.

    sbatch -J views_train -c 32 --mem 96G -t 02:00:00 scripts/leonardo/cpu.sbatch \
        python scripts/synth/make_views.py --set train --workers 32
    ... --set val / --set test (8 groups x 2 views); stress sets test_* (see SET_CFG)
    python scripts/synth/make_views.py --set val --preview 4   (contact sheet of 4 livers, no cache)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ofeat.data import liver_meshes
from ofeat.mesh import data_root
from ofeat.synth.texture import HELD_OUT_FAMILIES
from ofeat.synth.views import ViewConfig, render_liver

_EVAL = ViewConfig(n_groups=8, group_size=2)
SET_CFG = {"train": ViewConfig(), "val": _EVAL, "test": _EVAL,
           # stress sets (test livers only; docs/protocol.md section 7). Same seeds as test, separate
           # texture / tool random streams: the texture and tool sets are paired view for view.
           "test_tex_heldout": replace(_EVAL, families=HELD_OUT_FAMILIES),
           "test_tex_flat": replace(_EVAL, families=("flat",)),
           **{f"test_tools{k}": replace(_EVAL, n_tools=k) for k in range(4)},
           "test_deform": replace(_EVAL, deform_mm=(2.0, 8.0))}
SET_SALT = {"train": 1, "val": 2, **{k: 3 for k in SET_CFG if k.startswith("test")}}
SPLIT_OF = {k: k.split("_")[0] for k in SET_CFG}


def out_dir(view_set: str) -> Path:
    return data_root() / "processed" / "views" / view_set


def seed_of(subject: str, view_set: str) -> int:
    return int(subject.lstrip("s")) * 10 + SET_SALT[view_set]


def one(args: tuple[str, str]) -> str:
    subject, view_set = args
    out = out_dir(view_set) / f"{subject}.npz"
    if out.exists():
        return f"{subject} exists"
    mesh = liver_meshes.load(subject)
    d = render_liver(mesh, SET_CFG[view_set], seed_of(subject, view_set), backend="raycast")
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, subject=subject, split=SPLIT_OF[view_set], view_set=view_set, **d)
    os.replace(tmp, out)
    return f"{subject} {len(d['E'])} views, vis_frac {np.median(d['vis_frac']):.2f}"


def preview(view_set: str, n: int) -> None:
    import cv2

    rows = []
    for s in liver_meshes.subjects(SPLIT_OF[view_set])[:n]:
        d = render_liver(liver_meshes.load(s), replace(SET_CFG[view_set], n_groups=3, group_size=2),
                         seed_of(s, view_set), backend="raycast")
        tiles = []
        for rgb, depth in zip(d["rgb"], d["depth"]):
            dz = depth.astype(np.float32)
            dv = np.zeros_like(dz)
            ok = dz > 0
            if ok.any():
                dv[ok] = 255 * (dz[ok] - dz[ok].min()) / max(np.ptp(dz[ok]), 1e-6)
            tiles.append(np.concatenate([rgb, np.repeat(dv[..., None], 3, -1).astype(np.uint8)], 0))
        rows.append(cv2.resize(np.concatenate(tiles, 1), None, fx=0.35, fy=0.35))
    out = Path("outputs/synth") / f"preview_{view_set}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, 0)[..., ::-1])
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SET_CFG))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--preview", type=int, default=0)
    a = ap.parse_args()
    if a.preview:
        return preview(a.set, a.preview)
    subjects = liver_meshes.subjects(SPLIT_OF[a.set])[: a.limit]
    out_dir(a.set).mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(one, [(s, a.set) for s in subjects]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

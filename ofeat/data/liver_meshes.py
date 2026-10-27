"""TotalSegmentator liver surfaces (world mm), from scripts/preprocess/masks_to_meshes.py.

Layout: data/processed/totalseg_meshes/{index.csv, liver/<subject>.ply}. Subjects follow the
official TotalSegmentator split (configs/splits/totalseg_liver.yaml, built from meta.csv by
scripts/preprocess/make_splits.py). Only meshes with status "ok" are used.

Copied from correspondences/corr/data/liver_meshes.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ofeat._vendor.io import read_ply
from ofeat.data.splits import load_split
from ofeat.mesh import Mesh, data_root


def mesh_dir() -> Path:
    return data_root() / "processed" / "totalseg_meshes"


MIN_VOLUME_ML = 300.0  # organfeat: three "ok" meshes are 50-60 ml fragments cut by the CT field of view


def ok_subjects() -> list[str]:
    with open(mesh_dir() / "index.csv") as f:
        return sorted(r["subject"] for r in csv.DictReader(f)
                      if r["organ"] == "liver" and r["status"] == "ok"
                      and float(r["volume_ml"]) >= MIN_VOLUME_ML)


def subjects(split: str) -> list[str]:
    ok = set(ok_subjects())
    return [s for s in load_split("totalseg_liver")[split] if s in ok]


def load(subject: str) -> Mesh:
    v, _, f = read_ply(mesh_dir() / "liver" / f"{subject}.ply")
    return Mesh(v, f, mesh_id=f"totalseg/{subject}", meta={"subject": subject})

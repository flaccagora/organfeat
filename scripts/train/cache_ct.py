"""Cache `n` independent CT samplings (ofeat.data.points.ct_cloud) per liver of a split, for training
(resampling augmentation without recomputing normals in the data loader).

    sbatch -J cache_ct -c 32 --mem 64G -t 01:00:00 scripts/leonardo/cpu.sbatch \
        python scripts/train/cache_ct.py --split train --workers 32
-> data/processed/ct_cache/<subject>.npz: xyz (n, N, 3) world mm, normals (n, N, 3)
Samplings of one liver differ slightly in size; each is randomly subsampled to the smallest.
"""

from __future__ import annotations

import argparse
import os
import zlib
from multiprocessing import Pool

import numpy as np

from ofeat.data import liver_meshes
from ofeat.data.points import ct_cloud, load_config
from ofeat.train.data import ct_cache_path

N_SAMPLINGS = 4


def one(s: str) -> str:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    out = ct_cache_path(s)
    if out.exists():
        return f"{s} exists"
    cfg = load_config()
    mesh = liver_meshes.load(s)
    cl = [ct_cloud(mesh, cfg, np.random.default_rng(zlib.crc32(f"ctcache/{s}/{k}".encode())))
          for k in range(N_SAMPLINGS)]
    n = min(len(c) for c in cl)  # equal sizes: subsample the larger ones
    rng = np.random.default_rng(0)
    idx = [np.sort(rng.choice(len(c), n, replace=False)) for c in cl]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, xyz=np.stack([c.xyz[i] for c, i in zip(cl, idx)]),
             normals=np.stack([c.normals[i] for c, i in zip(cl, idx)]))
    return f"{s} {n} points"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(1)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(one, liver_meshes.subjects(a.split)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

"""Fit the PCA projections used by baselines and the visual input channel, on TRAIN livers only.

  dino64     DINOv2 tokens lifted to partial points of train renders (2 views x 100 livers, 500 points each)
  sonata32   frozen Sonata features of train CT clouds (50 livers, 2000 points each)

    sbatch -J fit_pca -t 01:00:00 scripts/leonardo/gpu_run.sbatch python scripts/bench/fit_pca.py
-> outputs/pca/{dino64,sonata32}.npz (mean, components, explained)
"""

from __future__ import annotations

import numpy as np

from ofeat.data import liver_meshes
from ofeat.data.points import load_config
from ofeat.eval.descriptors import PCA_DIR, dino_lifted, sonata_features
from ofeat.eval.pairs import ct_for, pairs
from ofeat.features.patches import fit_pca


def save(name: str, x: np.ndarray, dim: int) -> None:
    mu, comp, expl = fit_pca(x.astype(np.float64), dim)
    PCA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PCA_DIR / f"{name}.npz", mean=mu.astype(np.float32), components=comp.astype(np.float32),
             explained=expl, n=len(x))
    print(name, x.shape, f"explained {expl:.3f}", flush=True)


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(0)
    train = liver_meshes.subjects("train")
    subj = sorted(rng.choice(train, 100, replace=False).tolist())
    feats = []
    for p in pairs("train", subj, cfg, views=slice(0, 6, 3)):  # one view from each of 2 groups
        f = dino_lifted(p)
        feats.append(f[rng.choice(len(f), min(500, len(f)), replace=False)])
    save("dino64", np.concatenate(feats), 64)

    feats = []
    for s in subj[:50]:
        ct = ct_for(s, cfg)
        f = sonata_features(ct.xyz - ct.xyz.mean(0), ct.normals)
        feats.append(f[rng.choice(len(f), 2000, replace=False)])
    save("sonata32", np.concatenate(feats), 32)


if __name__ == "__main__":
    main()

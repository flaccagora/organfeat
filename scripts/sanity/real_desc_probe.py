"""Do the learned descriptors behave differently on real StereoMIS clouds than on synthetic views?
For a few P2 frame pairs and a few val synthetic pairs: number of points, mutual matches, descriptor
spread (mean pairwise cosine; ~1 = collapsed), and the effective rank of the descriptor matrix.

    sbatch -J real_probe -t 01:00:00 scripts/leonardo/gpu_run.sbatch python scripts/sanity/real_desc_probe.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts/eval")
from stereomis_consistency import liver_cloud  # noqa: E402

from ofeat.data import liver_meshes  # noqa: E402
from ofeat.data.liver_probe import load_probe  # noqa: E402
from ofeat.data.points import load_config, random_rigid, to_input  # noqa: E402
from ofeat.data.stereomis_frames import load_frame, meta  # noqa: E402
from ofeat.eval.descriptors import extractor  # noqa: E402
from ofeat.eval.pairs import pairs  # noqa: E402
from ofeat.match import match  # noqa: E402

CKPTS = {"A_mm": "outputs/train/s2_mm_lam0_lr1e-4/step_010000.pt", "B_mm": "outputs/train/s2_ppf_lam0/step_008000.pt"}


def stats(f: np.ndarray) -> str:
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    sub = f[np.random.default_rng(0).choice(len(f), min(2000, len(f)), replace=False)]
    cos = (sub @ sub.T)[np.triu_indices(len(sub), 1)].mean()
    s = np.linalg.svd(sub - sub.mean(0), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    return f"mean cos {cos:.3f}, eff. rank {np.exp(-(p * np.log(p + 1e-12)).sum()):.1f}"


def main() -> None:
    cfg = load_config()
    probe = load_probe()
    rng = np.random.default_rng(0)
    for name, ck in CKPTS.items():
        ex = extractor(ck)
        print(f"===== {name}")
        for p in list(pairs("val", liver_meshes.subjects("val")[:2], cfg, views=slice(0, 2))):
            fs = ex(p.P, p.nP, rgb=p.rgb, pix=p.pix)
            ft = ex(p.Q, p.nQ)
            m = match(fs, ft)
            print(f"synthetic {p.subject}/{p.view}: n {len(p.P)} / {len(p.Q)}, mutual {len(m.src)}; src {stats(fs)}; ct {stats(ft)}")
        frames = meta("P2_1")["frames"]
        for i in frames[100:500:100]:
            fr = load_frame("P2_1", i)
            c = liver_cloud(fr, probe, cfg, rng)
            if len(c) < cfg.min_partial:
                print(f"real P2_1/{i}: only {len(c)} liver points")
                continue
            ci, _ = to_input(c, random_rigid(rng))
            f = ex(ci.xyz, ci.normals, rgb=fr.rgb, pix=c.pix)
            g = ex(ci.xyz, ci.normals)
            nn_self = match(f, g)
            depth = fr.depth[c.pix[:, 1], c.pix[:, 0]]
            print(f"real P2_1/{i}: n {len(c)}, depth median {np.median(depth):.0f} mm; img {stats(f)}; geo {stats(g)}; "
                  f"img<->geo mutual {len(nn_self.src)} (same points)")


if __name__ == "__main__":
    main()

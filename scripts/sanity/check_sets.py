"""Sanity of the cached test / stress sets: points per view, visible fraction, and the oracle-rigid
residual (0 for rigid sets, a few mm for test_deform) on the first livers of each set.

    sbatch -J sanity_sets -c 8 -t 00:30:00 scripts/leonardo/cpu.sbatch python scripts/sanity/check_sets.py
"""

from __future__ import annotations

import numpy as np

from ofeat._vendor.geometry import kabsch
from ofeat.data import liver_meshes
from ofeat.data.points import load_config
from ofeat.eval.metrics import rmse
from ofeat.eval.pairs import pairs

SETS = ("val", "test", "test_tex_heldout", "test_tex_flat", "test_tools0", "test_tools3", "test_deform")


def main() -> None:
    cfg = load_config()
    for s in SETS:
        split = "val" if s == "val" else "test"
        r, n, vf, nct = [], [], [], []
        for p in pairs(s, liver_meshes.subjects(split)[:3], cfg):
            r.append(rmse(kabsch(p.P, p.G), p.P, p.G))
            n.append(len(p.P))
            vf.append(p.meta["vis_frac"])
            nct.append(len(p.Q))
        print(f"{s:18s} pairs {len(r):3d}  partial pts median {np.median(n):6.0f}  CT pts {np.median(nct):6.0f}  "
              f"vis_frac {np.median(vf):.2f}  oracle-rigid rmse median {np.median(r):.2f} max {np.max(r):.2f} mm",
              flush=True)


if __name__ == "__main__":
    main()

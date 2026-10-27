"""Evaluation-validity check for the locked DePoLL run (no descriptor involved, no method change):
are the partial clouds and the marker frames consistent? Each scored view's partial cloud is moved
by the marker oracle-rigid transform into the CT input frame. Its distance to the reference CT
surface should be deformation-sized (~5-20 mm), not ~100 mm.

    sbatch -J depoll_check -t 01:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=depoll scripts/leonardo/gpu_run.sbatch \
        python scripts/sanity/depoll_frame_check.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ofeat._vendor.geometry import kabsch
from ofeat.data import depoll
from ofeat.data.depoll_views import coloured_mesh, render_colours, views
from ofeat.data.liver_probe import load_probe
from ofeat.data.points import ct_cloud, load_config as points_config, partial_cloud, random_rigid, to_input
from ofeat.eval.metrics import apply

import sys

sys.path.insert(0, "scripts/eval")
from depoll_eval import liver_pixels  # noqa: E402


def main() -> None:
    pcfg = points_config()
    probe = load_probe()
    ref = depoll.reference_mesh()
    for state in ("01", "02"):
        mesh, colours = coloured_mesh(state)
        pm = depoll.state_markers(state)
        ref_mk = np.vstack([pm[k].ref for k in pm])
        rec_mk = np.vstack([pm[k].deformed for k in pm])
        rng = np.random.default_rng(int(state))
        ct = ct_cloud(ref, pcfg, np.random.default_rng(1000 + int(state)))
        res = []
        for vi, cam in enumerate(views(mesh, rng)):
            rgb, depth = render_colours(mesh, colours, cam)
            mask = liver_pixels(rgb, depth, probe)
            pc = partial_cloud(rgb, depth, np.zeros(depth.shape, bool), cam.K, cam.E, pcfg, rng, mask=mask)
            if len(pc) < pcfg.min_partial:
                continue
            p_in, A = to_input(pc, random_rigid(rng))
            q_in, B = to_input(ct, random_rigid(rng))
            mk_src, mk_tgt = apply(A @ cam.E, rec_mk), apply(B, ref_mk)
            T = kabsch(mk_src, mk_tgt)
            d = cKDTree(q_in.xyz).query(apply(T, p_in.xyz))[0]
            # also: the whole reconstruction (no probe mask), same transform chain
            dall = cKDTree(apply(B, ref.verts)).query(apply(T @ A @ cam.E, mesh.verts))[0]
            res.append((len(pc), np.median(d), np.quantile(d, 0.9), np.median(dall)))
        r = np.array(res)
        print(f"state {state}: {len(r)} views; partial points median {np.median(r[:, 0]):.0f}; "
              f"partial-to-CT distance after oracle rigid: median {np.median(r[:, 1]):.1f} mm, p90 {np.median(r[:, 2]):.1f} mm; "
              f"whole reconstruction median {np.median(r[:, 3]):.1f} mm", flush=True)


if __name__ == "__main__":
    main()

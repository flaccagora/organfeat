"""Post-hoc chance reference for the locked P2I-LReg Dice (no descriptor, no change to the evaluation).
Same frames, same partial clouds and fitted affine cameras as scripts/eval/p2i_eval.py. The CT model is
centred on the partial cloud's centroid with a uniformly random rotation (10 draws per frame) and
projected the same way. A matching pipeline that registers should beat this reference.

    sbatch -J p2i_ref -t 04:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=p2i_lreg scripts/leonardo/cpu.sbatch \
        python scripts/sanity/p2i_reference.py
-> outputs/eval/p2i/locked_random_pose_reference.parquet
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, "scripts/eval")
from p2i_eval import boundary_dist, dice, partial_from_frame, project_mask  # noqa: E402

from ofeat.data import p2i_lreg  # noqa: E402
from ofeat.data.points import load_config as points_config  # noqa: E402

N_DRAWS = 10


def main() -> None:
    pcfg = points_config()
    rows = []
    for case in p2i_lreg.cases("locked"):
        mesh = p2i_lreg.model(case)
        c_model = mesh.verts.mean(0)
        for k, frame in enumerate(p2i_lreg.frames(case)):
            rng = np.random.default_rng([int(case), k])
            fr = p2i_lreg.load_frame(case, frame)
            pc = partial_from_frame(fr, pcfg, rng)
            if len(pc) < pcfg.min_partial:
                continue
            A_cam = p2i_lreg.fit_camera(fr)
            d, b = [], []
            for r in range(N_DRAWS):
                R = Rotation.random(random_state=1000 * k + r).as_matrix()
                posed = (mesh.verts - c_model) @ R.T + pc.xyz.mean(0)
                pred = project_mask(posed, mesh.faces, A_cam, fr.mask.shape)
                d.append(dice(pred, fr.mask))
                b.append(boundary_dist(pred, fr.mask))
            rows.append({"case": case, "frame": frame, "p2ilf_derived": case in p2i_lreg.P2ILF_DERIVED,
                         "dice_random_pose": float(np.mean(d)), "bd_random_pose_px": float(np.nanmean(b))})
        print(case, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet("outputs/eval/p2i/locked_random_pose_reference.parquet")
    print(df.groupby("case")[["dice_random_pose", "bd_random_pose_px"]].mean().mean().round(3).to_string())


if __name__ == "__main__":
    main()

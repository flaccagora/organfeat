"""What frame / units is a P2I-LReg real cloud in? (audit follow-up, one frame per case; dev only)

The clouds have one point per pixel (row-major?) but do not project back to their pixel with the
intrinsics. For a few frames: check the colour order against img.png (is it row-major?), fit
x ~ a (u - cx) + b and y ~ c (v - cy) + d inside the mask, look at z, and test the pinhole hypotheses
x = (u - cx) z / fx and x = (u - cx) / fx (unit-depth rays) up to the stored scale factor.

    sbatch -J p2i_probe -c 8 --mem 32G -t 00:30:00 scripts/leonardo/cpu.sbatch python scripts/sanity/p2i_cloud_probe.py
"""

from __future__ import annotations

import cv2
import numpy as np
from plyfile import PlyData

from ofeat.mesh import data_root

ROOT = data_root() / "raw" / "p2i_lreg" / "Liver_regis"
PROBES = [(f"{c:02d}", 0) for c in range(1, 22)]


def main() -> None:
    import yaml

    for case, k in PROBES:
        d = ROOT / case / "real"
        frame = sorted(p.name for p in (d / "labels").iterdir())[k]
        img = cv2.imread(str(d / "labels" / frame / "img.png"))[..., ::-1]
        msk = cv2.imread(str(d / "labels" / frame / "label.png"), cv2.IMREAD_UNCHANGED)
        m = (msk.max(-1) if msk.ndim == 3 else msk) > 0
        H, W = m.shape
        Kd = yaml.safe_load((d / "CAM_K.yml").read_text())
        K = np.array(Kd[sorted(Kd)[0]][0]["cam_K"]).reshape(3, 3)
        v = PlyData.read(str(d / "liverPcds" / f"{frame}.ply"))["vertex"].data
        if len(v) != H * W:
            f = int(round(np.sqrt(H * W / len(v))))
            print(f"== case {case} {frame}: cloud has {len(v)} points for a {H}x{W} image (1/{f} per side?)")
            if len(v) != (H // f) * (W // f):
                continue
            img = cv2.resize(img, (W // f, H // f), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m.astype(np.uint8), (W // f, H // f), interpolation=cv2.INTER_NEAREST) > 0
            H, W = m.shape
            K = np.diag([1 / f, 1 / f, 1.0]) @ K
        X = np.stack([v["x"], v["y"], v["z"]], 1).reshape(H, W, 3)
        C = np.stack([v["red"], v["green"], v["blue"]], 1).reshape(H, W, 3).astype(float)
        scale = float((d / "scale" / f"{frame}.txt").read_text().split()[0])
        print(f"== case {case} {frame}  K fx {K[0, 0]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}  scale {scale:.2f}")
        print("colour vs image, row-major: mean |diff|", np.abs(C - img).mean().round(2),
              " transposed-order check:", np.abs(C.reshape(-1, 3) - img.transpose(1, 0, 2).reshape(-1, 3)).mean().round(2))
        vv, uu = np.nonzero(m)
        P = X[vv, uu]
        print("x range", np.percentile(P[:, 0], [1, 50, 99]).round(2), " y range", np.percentile(P[:, 1], [1, 50, 99]).round(2),
              " z range", np.percentile(P[:, 2], [1, 50, 99]).round(2))
        for name, coord, c0 in (("x~u", uu, K[0, 2]), ("y~v", vv, K[1, 2])):
            A = np.c_[coord - c0, np.ones(len(coord))]
            col = 0 if name == "x~u" else 1
            coef, res, *_ = np.linalg.lstsq(A, P[:, col], rcond=None)
            r = np.corrcoef(coord, P[:, col])[0, 1]
            print(f"{name}: slope {coef[0]:.5f} offset {coef[1]:.3f} corr {r:.4f}")
        # pinhole with depth z: x / z vs (u - cx) / fx
        rx = P[:, 0] / np.where(np.abs(P[:, 2]) > 1e-9, P[:, 2], np.nan)
        print("corr(x/z, (u-cx)/fx)", np.round(np.corrcoef(np.nan_to_num(rx), (uu - K[0, 2]) / K[0, 0])[0, 1], 4),
              " median ratio (x/z)/((u-cx)/fx)", np.nanmedian(rx / ((uu - K[0, 2]) / K[0, 0] + 1e-12)).round(3))
        # camera that maps the provided cloud to its pixels: affine (8 params) vs pinhole with free f, cx, cy
        sub = np.random.default_rng(0).choice(len(P), min(50000, len(P)), replace=False)
        Ps, us, vs = P[sub], uu[sub], vv[sub]
        A = np.c_[Ps, np.ones(len(Ps))]
        au = np.linalg.lstsq(A, us, rcond=None)[0]
        av = np.linalg.lstsq(A, vs, rcond=None)[0]
        r_aff = np.hypot(A @ au - us, A @ av - vs)
        good = Ps[:, 2] > 1e-6
        Bu = np.c_[Ps[good, 0] / Ps[good, 2], np.ones(good.sum())]
        Bv = np.c_[Ps[good, 1] / Ps[good, 2], np.ones(good.sum())]
        fu = np.linalg.lstsq(Bu, us[good], rcond=None)[0]
        fv = np.linalg.lstsq(Bv, vs[good], rcond=None)[0]
        r_per = np.hypot(Bu @ fu - us[good], Bv @ fv - vs[good])
        print(f"affine camera residual px: median {np.median(r_aff):.1f} p90 {np.quantile(r_aff, .9):.1f}   "
              f"pinhole(free f) residual px: median {np.median(r_per):.1f} p90 {np.quantile(r_per, .9):.1f} "
              f"(fu {fu[0]:.0f}, fv {fv[0]:.0f}, cx {fu[1]:.0f}, cy {fv[1]:.0f}; K fx {K[0, 0]:.0f})")
        ok = np.abs(uu - K[0, 2]) > 50
        print("median x/(u-cx)", np.median(P[ok, 0] / (uu[ok] - K[0, 2])).round(5),
              " median z", np.median(P[:, 2]).round(3), " z*scale median", (np.median(P[:, 2]) * scale).round(2))


if __name__ == "__main__":
    main()

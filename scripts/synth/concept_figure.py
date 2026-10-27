"""Concept figure for CONCEPT.md: one cached synthetic view, the partial cloud the methods see, the
complete CT surface (shape only) and the view's patch painted onto it (the ground-truth answer).
Everything is drawn in the view's camera frame; no random rigid transform is applied.

    scripts/leonardo/run.sh python scripts/synth/concept_figure.py --set val --subject s0607 --view 3 \
        --out docs/img/concept_partial_vs_ct.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from ofeat.data import liver_meshes
from ofeat.data.points import load_config, partial_cloud
from ofeat.features.lift import backproject

PAINT_MM = 1.5  # a face is "seen" if a visible pixel's surface point lies this close to its centroid


def to_plot(x: np.ndarray) -> np.ndarray:
    """Camera frame (x right, y down, z forward) -> matplotlib axes (x right, y forward, z up)."""
    return np.stack([x[..., 0], x[..., 2], -x[..., 1]], -1)


def towards_viewer(elev: float, azim: float) -> np.ndarray:
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def frame(ax, pts: np.ndarray, elev: float, azim: float) -> None:
    c, r = 0.5 * (pts.min(0) + pts.max(0)), 0.5 * np.ptp(pts, 0).max()
    for set_lim, ci in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
        set_lim(ci - r, ci + r)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val")
    ap.add_argument("--subject", default="s0607")
    ap.add_argument("--view", type=int, default=3)
    ap.add_argument("--out", default="docs/img/concept_partial_vs_ct.png")
    a = ap.parse_args()

    d = np.load(Path("data/processed/views") / a.set / f"{a.subject}.npz")
    rgb, depth, tool, K, E = d["rgb"][a.view], d["depth"][a.view], d["tool"][a.view], d["K"], d["E"][a.view]
    part = partial_cloud(rgb, depth, tool, K, E, load_config(), np.random.default_rng(0))
    mesh = liver_meshes.load(a.subject)

    # complete surface in the camera frame (E maps world -> camera)
    verts = to_plot(mesh.verts @ E[:3, :3].T + E[:3, 3])
    tris = verts[mesh.faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12

    # every visible, tool-free pixel's surface point, to paint the faces the camera saw
    keep = (depth > 0) & ~tool
    seen = cKDTree(to_plot(backproject(depth, K)[keep]))
    dist, nn = seen.query(tris.mean(1), distance_upper_bound=PAINT_MM)
    hit = np.isfinite(dist)

    elev, azim = 22.0, -60.0  # a three-quarter view: the patch faces the viewer, the rest is in context
    grey = np.array([0.80, 0.78, 0.76])
    shade = 0.30 + 0.70 * np.abs(fn @ towards_viewer(elev, azim))
    plain = shade[:, None] * grey
    painted = plain.copy()
    painted[hit] = rgb[keep][nn[hit]] / 255.0

    fig = plt.figure(figsize=(16, 4.4), dpi=110)
    titles = []
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(rgb)
    ax.set_axis_off()
    titles.append((ax, "1. What the camera sees\n(tool pixels are discarded)"))

    ax = fig.add_subplot(1, 4, 2, projection="3d")
    xp = to_plot(part.xyz)
    ax.scatter(*xp.T, c=rgb[part.pix[:, 1], part.pix[:, 0]] / 255.0, s=3, depthshade=False, linewidths=0)
    frame(ax, xp, elev=15.0, azim=-75.0)
    titles.append((ax, f"2. Partial 3D surface\n({len(xp):,} points, colour from the image)"))

    for i, (colours, title) in enumerate([(plain, "3. Complete CT surface\n(shape only, no colour)"),
                                          (painted, "4. The answer: where the view\nsits on the CT surface")]):
        ax = fig.add_subplot(1, 4, 3 + i, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=colours, edgecolors="none"))
        frame(ax, verts, elev, azim)
        titles.append((ax, title))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.01, wspace=0.02)
    for ax, title in titles:  # one baseline for all titles: the image panel is shorter than the 3D ones
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1), 0.97, title, ha="center", va="top", fontsize=11)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="white")
    print(f"wrote {a.out}: {len(xp)} partial points, {hit.sum()} / {len(hit)} CT faces seen")


if __name__ == "__main__":
    main()

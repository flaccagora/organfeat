"""File readers. Copied from defreg/data/io.py (read_ply, read_opencv_yml, as_points_3xn_or_nx3).

Copied into organfeat from correspondences/corr/_vendor/io.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return (points, normals, faces). Tolerates extra elements (e.g. PCL 'camera')."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = v.dtype.names
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    normals = None
    if {"nx", "ny", "nz"} <= set(names):
        normals = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    faces = None
    if "face" in [e.name for e in ply.elements]:
        f = ply["face"].data
        key = "vertex_indices" if "vertex_indices" in f.dtype.names else f.dtype.names[0]
        rows = f[key]
        if len(rows) and all(len(r) == 3 for r in rows[: min(len(rows), 1000)]):
            faces = np.stack(rows).astype(np.int64)
    return points, normals, faces


def read_opencv_yml(path: str | Path) -> dict[str, np.ndarray]:
    """Read all top-level matrix nodes of an OpenCV FileStorage YAML file."""
    import cv2

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    try:
        out = {}
        for k in fs.root().keys():  # noqa: SIM118  (cv2.FileNode is not a mapping)
            node = fs.getNode(k)
            m = node.mat()
            if m is not None:
                out[k] = np.asarray(m)
            elif node.isReal() or node.isInt():
                out[k] = np.asarray(node.real())
        return out
    finally:
        fs.release()


def as_points_3xn_or_nx3(m: np.ndarray) -> np.ndarray:
    """OpenCV files store point sets as 3xK or Kx3; normalise to Kx3."""
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2 or 3 not in m.shape:
        raise ValueError(f"expected 3xK or Kx3, got {m.shape}")
    return m.T if m.shape[0] == 3 and m.shape[1] != 3 else m

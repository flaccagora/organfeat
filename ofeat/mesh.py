"""Core geometry type: a triangle surface in world millimetres.

Copied from correspondences/corr/mesh.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """$OFEAT_DATA when set (jobs), else the repo's data/ symlinks."""
    return Path(os.environ.get("OFEAT_DATA", REPO / "data"))


@dataclass
class Mesh:
    verts: np.ndarray  # (N, 3) float64, mm
    faces: np.ndarray  # (F, 3) int64
    normals: np.ndarray | None = None  # (N, 3) outward unit vertex normals
    mesh_id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.verts = np.asarray(self.verts, np.float64)
        self.faces = np.asarray(self.faces, np.int64)
        if self.normals is None:
            self.normals = vertex_normals(self.verts, self.faces)

    def to_trimesh(self):
        import trimesh

        return trimesh.Trimesh(self.verts, self.faces, process=False)


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, globally oriented outward (organs are closed surfaces)."""
    n = np.zeros_like(v)
    tri = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for j in range(3):
        np.add.at(n, f[:, j], tri)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if ((v - v.mean(0)) * n).sum() < 0:
        n = -n
    return n

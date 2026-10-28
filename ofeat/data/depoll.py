"""DePoLL: Deformable Porcine Laparoscopic Liver dataset v1.0 (Modrzejewski et al., IJCARS 2019).
Marker/transform reader copied from defreg/data/sources/depoll.py, adapted to plain arrays.

LOCKED evaluation set (ofeat.data.splits): every entry point calls require_unlocked("depoll").
License CC BY-NC-ND 4.0: never redistribute derived data.

Layout under data/raw/depoll:
    reference/referenceModel/nonsegmentedLobes/surfaceFull.ply     preoperative surface, CT frame
    reference/referenceModel/markers/markers.yml                   centroidsBalls (3x45), centroidsClips (3x15)
    <NN>/surfaces/{video_reconstruction.ply,surface_CT.ply}        targets, reconstruction frame
    <NN>/markers/{markers.yml,associations.yml}                    deformed markers (intraop CT frame)
    coordinate_transforms/<NN>/M.yml                               rigid intraop CT -> reconstruction

Copied from correspondences/corr/data/depoll.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ofeat._vendor.io import as_points_3xn_or_nx3, read_opencv_yml, read_ply
from ofeat.data.splits import require_unlocked
from ofeat.mesh import Mesh, data_root

STATES = [f"{i:02d}" for i in range(1, 14)]


@dataclass
class MarkerPairs:
    ref: np.ndarray  # (K, 3) reference position, preoperative CT frame
    deformed: np.ndarray  # (K, 3) deformed position, reconstruction frame
    kind: str  # "internal" (balls) or "surface" (clips)


def root() -> Path:
    return data_root() / "raw" / "depoll"


def phase2_states(split: str) -> list[str]:
    """DePoLL states of the phase-2 split (configs/splits/depoll_phase2.yaml): "dev" 01-04, "test" 05-13."""
    from ofeat.data.splits import load_split

    return list(load_split("depoll_phase2")[split])


def states() -> list[str]:
    require_unlocked("depoll")
    return [s for s in STATES if (root() / s / "markers" / "markers.yml").exists()]


def reference_markers() -> dict[str, np.ndarray]:
    require_unlocked("depoll")
    m = read_opencv_yml(root() / "reference/referenceModel/markers/markers.yml")
    return {"balls": as_points_3xn_or_nx3(m["centroidsBalls"]),
            "clips": as_points_3xn_or_nx3(m["centroidsClips"])}


def reference_mesh() -> Mesh:
    require_unlocked("depoll")
    v, _, f = read_ply(root() / "reference/referenceModel/nonsegmentedLobes/surfaceFull.ply")
    return Mesh(v, f, mesh_id="depoll/reference")


def rigid_ct_to_recon(state: str) -> np.ndarray:
    require_unlocked("depoll")
    return read_opencv_yml(root() / "coordinate_transforms" / state / "M.yml")["M"]


def state_markers(state: str) -> dict[str, MarkerPairs]:
    """Reference markers (CT frame) paired with deformed markers (reconstruction frame)."""
    require_unlocked("depoll")
    d = root() / state / "markers"
    return _pair_markers(reference_markers(), read_opencv_yml(d / "markers.yml"),
                         read_opencv_yml(d / "associations.yml"), rigid_ct_to_recon(state))


def _pair_markers(ref: dict[str, np.ndarray], deformed: dict[str, np.ndarray],
                  assoc: dict[str, np.ndarray], M: np.ndarray) -> dict[str, MarkerPairs]:
    """Two deviations from the dataset's README/PDF, verified on state 01 in defreg:
    - <NN>/markers/markers.yml is in the *intraoperative CT* frame, not the reconstruction
      frame: M (coordinate_transforms/<NN>/M.yml) puts the clips on surface_CT.ply (max ~3 mm).
    - Association is 1-based over [balls(45); clips(15)] and Association[k] is the *deformed*
      marker index of *reference* marker k (the README states the inverse).
    """
    ref_all = np.vstack([ref["balls"], ref["clips"]])
    st_all = np.vstack([as_points_3xn_or_nx3(deformed["centroidsBalls"]),
                        as_points_3xn_or_nx3(deformed["centroidsClips"])])
    n_balls = len(ref["balls"])
    a = np.asarray(assoc["Association"]).ravel().astype(np.int64) - 1
    if sorted(a.tolist()) != list(range(len(ref_all))):
        raise ValueError("Association is not a permutation of the reference markers")
    if sorted(a[:n_balls].tolist()) != list(range(n_balls)):
        raise ValueError("Association mixes balls and clips")
    st_recon = st_all @ M[:3, :3].T + M[:3, 3]
    deformed_for_ref = st_recon[a]  # row k: deformed position of reference marker k
    return {
        "balls": MarkerPairs(ref_all[:n_balls], deformed_for_ref[:n_balls], kind="internal"),
        "clips": MarkerPairs(ref_all[n_balls:], deformed_for_ref[n_balls:], kind="surface"),
    }

"""P2P datasets (Yang et al., IEEE JBHI 2025, doi:10.1109/JBHI.2025.3583875; github.com/zixinyang9109/P2P,
repo commit 7f42cc7, data from the authors' Google Drive via scripts/download/p2p.sh). The LiverMatch
README states the P2P data is released under CC BY-SA 4.0; the P2P repo has no licence file.
Phase-2 external benchmarks (docs/miccai_plan.md, docs/protocol.md 2026-10-28):
  E3  in silico test: 6,315 pairs. Complete source `src_pcd` (MSD liver surface, mm), cropped deformed
      target `tgt_pcd` (FEM, neo-Hookean), volumetric fiducials `src_vol` / `tgt_vol` (RMS-TRE over all
      of them), rigid part `R_gt`, `t_gt`. The authors' protocol (rot=True, sigma None) rotates the
      target by `rot_tgt`; visibility bins use Deform_mesh_npz_test/stat_svd.npz.
  E4  in vitro: 800 pairs from 4 phantoms (1, 2, 4, 5), CT of undeformed / deformed phantoms; `src_vs`,
      partial `tgt_vs`, 53 embedded fiducials; visibility in in_vitro/stat.npz.
Both test sets are LOCKED (OFEAT_UNLOCK_EVAL=p2p_test): test-only, scored once. The in silico *train*
livers (Dataset/Deform_mesh_npz/<id>) may build validation subsets, never training data.
Published per-sample results (`published`) reproduce the paper's tables exactly (checked 2026-10-28).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from ofeat.data.splits import require_unlocked
from ofeat.mesh import data_root

VIS_BINS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # the paper's bins (inclusive bounds, as Table_I&II.py)


def root() -> Path:
    return data_root() / "raw" / "p2p"


def _insilico_test_dir() -> Path:
    return root() / "data" / "in_silico_x" / "Dataset" / "Deform_mesh_npz_test"


@cache
def insilico_test_names() -> list[str]:
    require_unlocked("p2p_test")
    return [str(s) for s in np.load(_insilico_test_dir() / "list.npz")["test"]]


@cache
def insilico_stats() -> dict[str, np.ndarray]:
    require_unlocked("p2p_test")
    z = np.load(_insilico_test_dir() / "stat_svd.npz")
    return {"vis": z["vis"], "deform": z["deform"]}


def insilico_test(i: int, sigma: int | None = None) -> dict:
    """Sample i in the authors' protocol: target rotated by rot_tgt (rot=True); optional noise key.
    (R_gt, t_gt) is the rigid part source -> rotated target: rot_tgt @ R, rot_tgt @ t. Its fiducial RMS
    equals stat_svd.npz `deform` exactly (checked on samples 0 / 100 / 3000). The authors' eval script
    composes R_gt @ rot_tgt, used there only for inlier-ratio diagnostics, not for the published RMS-TRE."""
    require_unlocked("p2p_test")
    with np.load(_insilico_test_dir() / "Test" / insilico_test_names()[i], allow_pickle=True) as e:
        src, tgt = e["src_pcd"].astype(np.float64), e["tgt_pcd"].astype(np.float64)
        sm, tm = e["src_vol"].astype(np.float64), e["tgt_vol"].astype(np.float64)
        R, t, rt = e["R_gt"], e["t_gt"], e["rot_tgt"]
        if sigma is not None:
            tgt = tgt + e[str(sigma)]
    tgt, tm = tgt @ rt.T, tm @ rt.T
    st = insilico_stats()
    return {"name": insilico_test_names()[i], "src": src, "tgt": tgt, "src_markers": sm, "tgt_markers": tm,
            "R_gt": rt @ R, "t_gt": (rt @ t).ravel(), "vis": float(st["vis"][i]), "deform": float(st["deform"][i])}


def _invitro_dir() -> Path:
    return root() / "data" / "in_vitro_x" / "in_vitro"


@cache
def invitro_names() -> list[str]:
    require_unlocked("p2p_test")
    return [str(s) for s in np.load(_invitro_dir() / "rigid_list.npy", allow_pickle=True)]


def invitro(i: int) -> dict:
    require_unlocked("p2p_test")
    name = invitro_names()[i]
    with np.load(_invitro_dir() / "Rigid_test_data" / name, allow_pickle=True) as e:
        return {"name": name, "phantom": int(name.split("_")[0]), "src": e["src_vs"].astype(np.float64),
                "tgt": e["tgt_vs"].astype(np.float64), "src_markers": e["src_marker"].astype(np.float64),
                "tgt_markers": e["tgt_marker"].astype(np.float64), "vis": float(e["vis"])}


def published(dataset: str, rel: str) -> np.ndarray:
    """Per-sample RMS-TRE (mm) of a published method, e.g. ("in_silico", "LiverMatch/None004002.npy") or
    ("in_vitro", "Result/LiverMatch/0.npy"); the order follows insilico_test_names / invitro_names."""
    sub = {"in_silico": "Eva_in_silico/Results_in_silico", "in_vitro": "Eva_in_vitro/Results_in_vitro"}[dataset]
    return np.load(root() / "repo" / sub / rel, allow_pickle=True).astype(np.float64)


def rms_tre(T: np.ndarray, src_markers: np.ndarray, tgt_markers: np.ndarray) -> float:
    """The paper's RMS-TRE: source fiducials moved by the estimated source -> target transform vs target."""
    d = src_markers @ T[:3, :3].T + T[:3, 3] - tgt_markers
    return float(np.sqrt((d ** 2).sum(1).mean()))


def train_livers() -> list[str]:
    """In silico training livers (validation subsets only; never training)."""
    return sorted(p.name for p in (root() / "data" / "in_silico_x" / "Dataset" / "Deform_mesh_npz").iterdir() if p.is_dir())

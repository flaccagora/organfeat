"""Frozen subject splits and the locked-evaluation guard.

Splits are fixed in configs/splits/*.yaml before anything is rendered or cropped. Paired datasets
(DePoLL, the locked part of P2I-LReg) are locked: no training, normalisation (e.g. DINO PCA) or
model selection may see them. Their loaders call `require_unlocked(name)`, which raises unless the
final evaluation explicitly sets OFEAT_UNLOCK_EVAL=<name>[,<name>...] in the environment.

Copied from correspondences/corr/data/splits.py.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from ofeat.mesh import REPO

LOCKED = ("depoll", "p2i_lreg", "ircad_v2", "p2p_test")  # ircad_v2, p2p_test: docs/protocol.md, 2026-10-28
SPLITS_DIR = REPO / "configs" / "splits"


class LockedDatasetError(RuntimeError):
    pass


def require_unlocked(name: str) -> None:
    unlocked = {s.strip() for s in os.environ.get("OFEAT_UNLOCK_EVAL", "").split(",") if s.strip()}
    if name in LOCKED and name not in unlocked:
        raise LockedDatasetError(
            f"{name} is a locked evaluation set: set OFEAT_UNLOCK_EVAL={name} only in the final "
            "evaluation run (never while training, fitting PCA or selecting models)")


@cache
def load_split(name: str, splits_dir: Path = SPLITS_DIR) -> dict[str, list[str]]:
    """{split: [ids]} from configs/splits/<name>.yaml."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(splits_dir / f"{name}.yaml"))
    return {k: list(v) for k, v in cfg["splits"].items()}


def split_of(name: str, item: str) -> str | None:
    for split, ids in load_split(name).items():
        if item in ids:
            return split
    return None

"""StereoMIS frames as partial liver clouds (real RGB/geometry consistency, no CT).

From data/processed/stereomis/<seq> (extracted in correspondences, copied here): rectified left frame
(JPEG), SGBM z-depth (f16, half resolution), rectified tool mask, meta (rectified K, frame list).
Liver points: the frozen DINOv2 liver probe (patch mask) on the 644 x 518 frame, minus tools and
specular pixels. The kinematic camera poses are converted to the rectified left camera
(X_rect = R1 X_cam) with the convention checked by scripts/sanity/stereomis_poses.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

from ofeat.data.liver_probe import FRAME_SIZE
from ofeat.mesh import data_root


def proc_dir(seq: str) -> Path:
    return data_root() / "processed" / "stereomis" / seq


@cache
def meta(seq: str) -> dict:
    return json.loads((proc_dir(seq) / "meta.json").read_text())


@dataclass
class StereoFrame:
    seq: str
    frame: int
    rgb: np.ndarray  # (518, 644, 3) u8, rectified left, resized to FRAME_SIZE
    depth: np.ndarray  # (518, 644) mm, 0 invalid (SGBM, resized nearest)
    tool: np.ndarray  # (518, 644) bool
    K: np.ndarray  # intrinsics at FRAME_SIZE


def load_frame(seq: str, frame: int) -> StereoFrame:
    d = proc_dir(seq)
    m = meta(seq)
    name = f"{frame:06d}"
    W, H = FRAME_SIZE
    full_w, full_h = m["size"]
    rgb = cv2.resize(cv2.imread(str(d / "left" / f"{name}.jpg"))[..., ::-1], (W, H), interpolation=cv2.INTER_AREA)
    z = np.load(d / "depth" / f"{name}.npz")["z"].astype(np.float32)
    depth = cv2.resize(z, (W, H), interpolation=cv2.INTER_NEAREST)
    tp = d / "tool" / f"{name}.png"
    tool = (cv2.resize(cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST) > 127
            if tp.exists() else np.zeros((H, W), bool))
    K = np.array(m["K"], np.float64)
    S = np.diag([W / full_w, H / full_h, 1.0])
    return StereoFrame(seq=seq, frame=frame, rgb=np.ascontiguousarray(rgb), depth=depth, tool=tool, K=S @ K)

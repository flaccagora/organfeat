"""Patch-level liver probe on frozen DINOv2 tokens (upstream organ selection on real frames).

A logistic regression on DINOv2 ViT-L/14-reg patch tokens, trained in correspondences (Project 10) on
grid-cell labels of StereoMIS *training* frames (animal P2; leave-one-sequence-out accuracy
0.94-0.96, low recall on P3). It never saw DePoLL, P2I-LReg or any CT. Here it only selects liver
points on real frames; synthetic renders use the exact ground-truth mask.

Copied from correspondences/corr/data/liver_probe.py (inference helpers only). The weights,
correspondences/outputs/liver_probe/probe.joblib, are copied to data/weights/liver_probe/.
"""

from __future__ import annotations

import numpy as np

from ofeat.mesh import data_root

FRAME_SIZE = (644, 518)  # (W, H): 46 x 37 DINOv2 patches, ~5:4 like the 1280 x 1024 frames


def load_probe():
    import joblib

    return joblib.load(data_root() / "weights" / "liver_probe" / "probe.joblib")


def p_liver(probe, tokens: np.ndarray) -> np.ndarray:
    """tokens (h, w, C) L2-normalised DINOv2 patch tokens -> (h, w) p(liver)."""
    h, w, c = tokens.shape
    return probe.predict_proba(tokens.reshape(-1, c))[:, 1].reshape(h, w)


def patch_mask(p_liver: np.ndarray, tool_frac: np.ndarray | None = None, thr: float = 0.5,
               min_component: float = 0.03, max_tool: float = 0.2) -> np.ndarray:
    """(h, w) bool liver patches: threshold, 3x3 opening (the probe's errors on spleen / stomach are
    isolated patches, liver is one large region), connected components >= `min_component` of the
    frame, minus patches with > `max_tool` tool coverage. Fixed by eye on the training audit in
    correspondences, not tuned on any evaluation set."""
    import cv2

    m = (np.asarray(p_liver, np.float32) > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    keep = np.zeros(m.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_component * m.size:
            keep |= lab == i
    if tool_frac is not None:
        tf = np.nan_to_num(np.asarray(tool_frac, np.float32), nan=0.0)
        keep &= tf <= max_tool
    return keep

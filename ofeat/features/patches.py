"""Small shared helpers for per-patch quantities.

`fit_pca` copied from correspondences/corr/features/patches.py.
"""

from __future__ import annotations

import numpy as np


def fit_pca(sample: np.ndarray, dim: int):
    """-> (mean (C,), components (dim, C), explained variance fraction)."""
    mu = sample.mean(0)
    _, s, Vt = np.linalg.svd(sample - mu, full_matrices=False)
    return mu, Vt[:dim], float((s[:dim] ** 2).sum() / (s ** 2).sum())

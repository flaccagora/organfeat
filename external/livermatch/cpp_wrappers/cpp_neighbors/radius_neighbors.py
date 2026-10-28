"""ofeat: pure-Python replacement of the KPConv C++ wrapper (its numpy-1.7 C API does not build with
numpy 2). Copied from defreg/defreg/methods/livermatch.py, where it reproduces the C++ output exactly."""

import numpy as np
from scipy.spatial import cKDTree


def batch_query(queries, supports, q_batches, s_batches, radius=0.1):
    queries, supports = np.asarray(queries), np.asarray(supports)
    rows, kmax = [], 1
    qs = ss = 0
    for nq, ns in zip(np.asarray(q_batches, np.int64), np.asarray(s_batches, np.int64)):
        tree = cKDTree(supports[ss:ss + ns])
        q = queries[qs:qs + nq]
        k = max(1, int(tree.query_ball_point(q, radius, return_length=True).max(initial=0)))
        _, idx = tree.query(q, k=k, distance_upper_bound=radius)  # sorted, missing -> ns
        idx = np.asarray(idx).reshape(nq, k)
        idx = np.where(idx >= ns, len(supports), idx + ss)
        rows.append(idx)
        kmax = max(kmax, k)
        qs, ss = qs + nq, ss + ns
    out = np.full((len(queries), kmax), len(supports), np.int32)
    r = 0
    for idx in rows:
        out[r:r + len(idx), :idx.shape[1]] = idx
        r += len(idx)
    return out

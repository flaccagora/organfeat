"""ofeat: pure-Python replacement of the KPConv C++ wrapper (its numpy-1.7 C API does not build with
numpy 2). Copied from defreg/defreg/methods/livermatch.py, where it reproduces the C++ output exactly."""

import numpy as np


def subsample_batch(points, batches_len, features=None, classes=None, sampleDl=0.1, max_p=0,
                    verbose=0):
    out_p, out_l, out_f = [], [], []
    start = 0
    for n in np.asarray(batches_len, np.int64):
        p = np.asarray(points[start:start + n], np.float32)
        origin = np.floor(p.min(0) * (1 / sampleDl)) * sampleDl
        cell = np.floor((p - origin) / sampleDl).astype(np.int64)
        _, inv, cnt = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        acc = np.zeros((len(cnt), 3), np.float64)
        np.add.at(acc, inv, p)
        out_p.append((acc / cnt[:, None]).astype(np.float32))
        if features is not None:
            f = np.asarray(features[start:start + n], np.float64)
            facc = np.zeros((len(cnt), f.shape[1]))
            np.add.at(facc, inv, f)
            out_f.append((facc / cnt[:, None]).astype(np.float32))
        out_l.append(len(cnt))
        start += n
    res = [np.concatenate(out_p), np.asarray(out_l, np.int32)]
    if features is not None:
        res.append(np.concatenate(out_f))
    if classes is not None:
        raise NotImplementedError("label subsampling is training-only")
    return tuple(res)

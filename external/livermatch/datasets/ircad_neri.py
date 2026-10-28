"""ofeat: LiverMatch dataset over the deformed-IRCAD generator's h5 files (Neri et al. 2025 protocol;
scripts/external/gen_deformed_ircad.py). Each file holds pairs: src_complete (complete source),
tgt_partial_def (partial target: deformed, rigidly moved, jittered), corr_partial ([src, tgt] indices),
gt_T (source -> target rigid part) and min / max (mm per normalised unit per axis).
Coordinates are scaled to LiverMatch's units (mm / 100) with the liver's mean half-extent (as ofeat's
own IRCAD evaluation scales them to mm). Augmentation follows the upstream `livermatch` dataset: a
random rotation of the target with probability 0.5, then centring of both clouds.
"""
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

# ofeat: upstream datasets/liver.py switches to the 'file_system' sharing strategy, which leaks /dev/shm
# memory once DataLoader workers return tensors (upstream ran num_workers 0; with workers the first full
# run was OOM-killed at 32 GB). This module is imported after it (datasets/dataloader.get_datasets), so the
# default strategy is restored for the ircad_neri runs.
torch.multiprocessing.set_sharing_strategy("file_descriptor")


class IrcadNeri(Dataset):
    def __init__(self, config, split, data_augmentation=True):
        super().__init__()
        self.config = config
        self.files = list(config.h5_train if split == "train" else config.h5_val)
        self.data_augmentation = data_augmentation and split == "train"
        self.max_points = config.max_points
        livers = set(config.train_livers if split == "train" else config.val_livers)
        self.index = []
        for fi, f in enumerate(self.files):
            with h5py.File(f, "r") as h:
                lv = h["liver"][:]
            self.index += [(fi, int(i)) for i in np.flatnonzero(np.isin(lv, list(livers)))]
        if split != "train" and getattr(config, "val_slice", None):
            self.index = self.index[: config.val_slice]
        if split == "train" and getattr(config, "train_slice", None):  # smoke runs only
            self.index = self.index[: config.train_slice]
        self._h = {}

    def __len__(self):
        return len(self.index)

    def _file(self, fi):
        if fi not in self._h:
            self._h[fi] = h5py.File(self.files[fi], "r")
        return self._h[fi]

    def __getitem__(self, index):
        fi, i = self.index[index]
        h = self._file(fi)
        s = float(((h["max"][i] - h["min"][i]) / 2).mean()) / 100.0  # normalised unit -> mm / 100
        src = h["src_complete"][i].astype(np.float32) * s
        tgt = h["tgt_partial_def"][i].astype(np.float32) * s
        c = h["corr_partial"][i].astype(np.int64)  # rows [src index, tgt index]
        T = h["gt_T"][i]
        rot, trans = T[:, :3].astype(np.float32), (T[:, 3:] * s).astype(np.float32)
        if self.data_augmentation:  # as upstream livermatch.__getitem__
            rot = Rotation.from_euler("zyx", np.random.rand(3) * np.pi * 2).as_matrix().astype(np.float32)
            trans = (np.random.rand(3, 1) * 0.1).astype(np.float32)
            if np.random.rand(1)[0] > 0.5:
                tgt = (rot @ tgt.T + trans).T.astype(np.float32)
        if len(src) > self.max_points or len(tgt) > self.max_points:
            raise ValueError("ircad_neri: clouds exceed max_points")
        corr = torch.from_numpy(c[:, :2])
        src = src - src.mean(0)
        tgt = tgt - tgt.mean(0)
        ones_s, ones_t = np.ones_like(src[:, :1]), np.ones_like(tgt[:, :1])
        return src, tgt, ones_s, ones_t, rot, trans, corr, src, tgt, torch.ones(1)

"""StereoMIS v0.0.1 (Hayoz et al., IJCARS 2023; Zenodo 8154924, license "other-nc").

Layout under data/raw/stereomis/StereoMIS_0_0_1/<seq>/:
    *.mp4                  vertically stacked stereo, 1280 x 2048 (left on top)
    StereoCalibration.ini  [StereoLeft] / [StereoRight]: fc, cc, kc_0..4 (OpenCV k1 k2 p1 p2 k3),
                           R_0..8 / T_0..2 (right camera w.r.t. left; baseline ~4.2 mm)
    masks/<frame>l.png     automatic tool masks of left frames (every `step` frames), half
                           resolution, white = tissue / black = tool (verified on P2_0)
    groundtruth.txt        camera forward kinematics per frame: `frame tx ty tz qx qy qz qw`
                           (identity at frame 0); used only as pseudo ground truth for
                           frame-pair consistency, convention checked by scripts/sanity
    *_split.csv            annotated [start, end] frame ranges
Splits by animal: configs/splits/stereomis.yaml.

Copied from correspondences/corr/data/stereomis.py; adds the forward-kinematics camera poses.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ofeat.data.splits import load_split
from ofeat.mesh import data_root


def root() -> Path:
    return data_root() / "raw" / "stereomis" / "StereoMIS_0_0_1"


def sequences(split: str) -> list[str]:
    return load_split("stereomis")[split]


@dataclass
class StereoCalib:
    K1: np.ndarray
    d1: np.ndarray
    K2: np.ndarray
    d2: np.ndarray
    R: np.ndarray
    T: np.ndarray  # mm
    size: tuple[int, int]  # (W, H)

    @classmethod
    def read(cls, path: Path) -> StereoCalib:
        c = configparser.ConfigParser()
        c.read(path)

        def cam(sec):
            s = c[sec]
            K = np.array([[float(s["fc_x"]), 0, float(s["cc_x"])],
                          [0, float(s["fc_y"]), float(s["cc_y"])], [0, 0, 1]])
            return K, np.array([float(s[f"kc_{i}"]) for i in range(5)])

        K1, d1 = cam("StereoLeft")
        K2, d2 = cam("StereoRight")
        s = c["StereoRight"]
        R = np.array([float(s[f"R_{i}"]) for i in range(9)]).reshape(3, 3)
        T = np.array([float(s[f"T_{i}"]) for i in range(3)])
        size = (int(c["StereoLeft"]["res_x"]), int(c["StereoLeft"]["res_y"]))
        return cls(K1, d1, K2, d2, R, T, size)


class Sequence:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = root() / name

    @cached_property
    def video(self) -> Path:
        """The stereo video (not copied into organfeat: frames come from data/processed)."""
        return sorted(self.dir.glob("*.mp4"))[0]

    @cached_property
    def calib(self) -> StereoCalib:
        return StereoCalib.read(self.dir / "StereoCalibration.ini")

    @cached_property
    def ranges(self) -> list[tuple[int, int]]:
        rows = (self.dir / next(p.name for p in self.dir.glob("*_split.csv"))).read_text().split()
        return [tuple(int(x) for x in r.split(",")) for r in rows[1:]]

    @cached_property
    def rectification(self) -> dict:
        """OpenCV stereo rectification (alpha 0: valid pixels only). P1/P2 are the rectified
        projections; the rectified baseline is -P2[0, 3] / P2[0, 0] mm."""
        import cv2

        c = self.calib
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(c.K1, c.d1, c.K2, c.d2, c.size, c.R,
                                                    c.T.reshape(3, 1),
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        maps = [cv2.initUndistortRectifyMap(K, d, Rr, P, c.size, cv2.CV_32FC1)
                for K, d, Rr, P in ((c.K1, c.d1, R1, P1), (c.K2, c.d2, R2, P2))]
        return {"R1": R1, "P1": P1, "P2": P2, "Q": Q, "maps": maps}

    def frames(self, step: int = 1, rectify: bool = True, start: int = 0,
               stop: int | None = None):
        """Yield (frame index, left RGB, right RGB) uint8 over the annotated ranges, restricted
        to [start, stop]. Always decodes sequentially from frame 0: seeking in these HEVC files
        lands on frames that disagree with the dataset's tool masks
        (scripts/sanity/seek_check.py, 2026-10-22), sequential decoding agrees."""
        import cv2

        wanted = {f for a, b in self.ranges for f in range(a, b + 1, step)
                  if f >= start and (stop is None or f <= stop)}
        if not wanted:
            return
        last = max(wanted)
        cap = cv2.VideoCapture(str(self.video))
        i = 0
        try:
            while i <= last:
                ok = cap.grab()
                if not ok:
                    break
                if i in wanted:
                    _, bgr = cap.retrieve()
                    H = bgr.shape[0] // 2
                    lr = [bgr[:H, :, ::-1], bgr[H:, :, ::-1]]
                    if rectify:
                        lr = [cv2.remap(np.ascontiguousarray(im), mx, my, cv2.INTER_LINEAR)
                              for im, (mx, my) in zip(lr, self.rectification["maps"])]
                    yield i, lr[0], lr[1]
                i += 1
        finally:
            cap.release()

    @cached_property
    def poses(self) -> dict[int, np.ndarray]:
        """{frame: (4, 4)} forward-kinematics camera poses as stored (rotation from the unit
        quaternion, translation as given). Units and direction (camera-to-world or inverse) are not
        documented: scripts/sanity/stereomis_poses.py checks them by warping depth between frames."""
        from scipy.spatial.transform import Rotation

        rows = np.loadtxt(self.dir / "groundtruth.txt")
        out = {}
        for r in np.atleast_2d(rows):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(r[4:8]).as_matrix()
            T[:3, 3] = r[1:4]
            out[int(r[0])] = T
        return out

    def tool_mask(self, frame: int, rectify: bool = True) -> np.ndarray | None:
        """Tool mask of the left frame (bool), None if the dataset has none for this frame."""
        import cv2

        p = self.dir / "masks" / f"{frame:06d}l.png"
        if not p.exists():
            return None
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, self.calib.size, interpolation=cv2.INTER_NEAREST)  # stored at half size
        if rectify:
            mx, my = self.rectification["maps"][0]
            m = cv2.remap(m, mx, my, cv2.INTER_NEAREST)
        return m <= 127  # files are white on tissue

"""DePoLL reader against the data on disk (skipped if not extracted). Pins the two places where
the files disagree with the README (marker frame, association direction); copied from defreg."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from ofeat._vendor.geometry import kabsch
from ofeat._vendor.io import read_ply
from ofeat.data import depoll

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not (depoll.root() / "01" / "markers").exists(), reason="DePoLL not extracted"),
]


@pytest.fixture(autouse=True)
def _unlock(monkeypatch):
    monkeypatch.setenv("OFEAT_UNLOCK_EVAL", "depoll")


def test_deformed_clips_lie_on_ct_surface():
    clips = depoll.state_markers("01")["clips"].deformed
    surf, _, _ = read_ply(depoll.root() / "01/surfaces/surface_CT.ply")
    assert cKDTree(surf).query(clips)[0].max() < 5.0


def test_association_direction():
    t = depoll.state_markers("01")
    P = np.vstack([t["balls"].ref, t["clips"].ref])
    Q = np.vstack([t["balls"].deformed, t["clips"].deformed])
    T = kabsch(P, Q)
    assert np.linalg.norm(P @ T[:3, :3].T + T[:3, 3] - Q, axis=1).mean() < 30.0  # README: ~49 mm

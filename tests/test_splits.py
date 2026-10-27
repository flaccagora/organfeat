"""The locked-evaluation guard and the frozen splits (copied from correspondences)."""

import pytest

from ofeat.data import depoll
from ofeat.data.splits import LockedDatasetError, SPLITS_DIR, load_split, require_unlocked


def test_locked_by_default(monkeypatch):
    monkeypatch.delenv("OFEAT_UNLOCK_EVAL", raising=False)
    with pytest.raises(LockedDatasetError):
        require_unlocked("depoll")
    with pytest.raises(LockedDatasetError):
        depoll.states()


def test_unlock_is_per_dataset(monkeypatch):
    monkeypatch.setenv("OFEAT_UNLOCK_EVAL", "p2i_lreg")
    with pytest.raises(LockedDatasetError):
        require_unlocked("depoll")
    monkeypatch.setenv("OFEAT_UNLOCK_EVAL", "depoll")
    require_unlocked("depoll")


@pytest.mark.skipif(not (SPLITS_DIR / "totalseg_liver.yaml").exists(), reason="splits not frozen")
def test_totalseg_splits_disjoint():
    s = load_split("totalseg_liver")
    assert set(s) == {"train", "val", "test"}
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["train"]) & set(s["val"]))


def test_p2i_lreg_split_and_guard(monkeypatch):
    from ofeat.data import p2i_lreg

    s = load_split("p2i_lreg")
    assert len(s["dev"]) == 4 and len(s["locked"]) == 17 and not set(s["dev"]) & set(s["locked"])
    assert {"01", "02", "13", "17", "18"} <= set(s["locked"])  # the authors' test cases stay locked
    monkeypatch.delenv("OFEAT_UNLOCK_EVAL", raising=False)
    with pytest.raises(LockedDatasetError):
        p2i_lreg.frames(s["locked"][0])

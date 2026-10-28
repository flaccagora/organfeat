# LiverMatch (vendored baseline)

Upstream: Yang et al., LiverMatch (IJCARS 2023), https://github.com/zixinyang9109/LiverMatch, MIT licence
(LICENSE). Copied from defreg/data/raw/livermatch/repo on 2026-10-28, without `snapshot/` (published
weights) and `test_data/`. Used only as the R1 baseline (docs/protocol.md, 2026-10-28): retrained on
deformed-IRCAD generator pairs.

Changes here, all marked "ofeat:":
- cpp_wrappers/*/setup.py: setuptools + numpy.get_include(); distutils and numpy.distutils are gone in
  Python 3.12 / numpy 2
- datasets/ircad_neri.py: a dataset over the generator's h5 files (complete source, partial target,
  correspondences)
- configs/ircad_neri.yaml: training config for it
- lib/liver_trainer.py: next(iterator) instead of iterator.next() (removed from torch DataLoader iterators)
- datasets/ircad_neri.py restores torch's 'file_descriptor' sharing strategy (upstream's 'file_system' leaks /dev/shm with DataLoader workers; first full run OOM-killed)

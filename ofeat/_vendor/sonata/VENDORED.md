# Vendored: facebookresearch/sonata

- Source: https://github.com/facebookresearch/sonata
- Commit: 18c09ff8d713494f78a8213792262b910977a65d (2025-06-04)
- License: Apache-2.0 (LICENSE in this folder)
- Weights: Hugging Face `facebook/sonata` (`sonata.pth`, `sonata_small.pth`), not vendored

## Local changes
1. `model.py`: `import torch_scatter` replaced by `from ._scatter import segment_csr` (and the
   three `torch_scatter.segment_csr(...)` calls by `segment_csr(...)`). torch-scatter has no
   PyPI wheels and data.pyg.org fails TLS verification from our network (ADR-0011);
   `_scatter.py` implements `segment_csr` with native `torch.segment_reduce`.

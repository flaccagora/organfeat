"""Native replacement for torch_scatter.segment_csr (CSR segments along dim 0)."""

import torch


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    if reduce not in ("sum", "mean", "max", "min"):
        raise ValueError(f"unsupported reduce {reduce!r}")
    lengths = (indptr[1:] - indptr[:-1]).to(torch.long)
    return torch.segment_reduce(src, reduce, lengths=lengths, axis=0, unsafe=True)

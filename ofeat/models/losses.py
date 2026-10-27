"""Multi-positive InfoNCE with a safe radius (adapted from defreg/pretrain/p3.py::p3_loss).

Anchors are points of cloud A that have a counterpart in cloud B (a B point within `pos_radius` of
the anchor's true position). Positives are all B points within `pos_radius`. Negatives are the B
points farther than `safe_radius`. B points in between are left out of the denominator: on a smooth
organ, nearby points are nearly indistinguishable and should not be pushed apart. Positions are
true positions in one shared frame (labels, never inputs). defreg's P3 (positives 2-4 mm, no safe
radius, ~6 mm deformation) did not learn (defreg results 014). Here the core scope is rigid, so the
positions are exact.
"""

from __future__ import annotations

import torch


def infonce(za: torch.Tensor, zb: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor, tau: float = 0.07,
            pos_radius: float = 4.0, safe_radius: float = 12.0, n_anchor: int = 1024,
            generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """za (Na, D), zb (Nb, D) L2-normalised; pa (Na, 3), pb (Nb, 3) true positions in one frame (mm).
    Returns (loss, stats: top-1 accuracy within pos_radius and within 10 mm, number of anchors)."""
    with torch.no_grad():
        dmin = torch.cdist(pa, pb).min(1).values
        valid = torch.nonzero(dmin < pos_radius).squeeze(1)
        if len(valid) > n_anchor:
            valid = valid[torch.randperm(len(valid), device=valid.device, generator=generator)[:n_anchor]]
    if len(valid) == 0:
        return za.sum() * 0.0, {"acc": 0.0, "acc10": 0.0, "n": 0}
    with torch.no_grad():
        d = torch.cdist(pa[valid], pb)
        pos = d < pos_radius
        keep = pos | (d > safe_radius)
    logits = (za[valid] @ zb.T).float() / tau
    neg_inf = torch.finfo(logits.dtype).min
    lse_pos = torch.logsumexp(logits.masked_fill(~pos, neg_inf), 1)
    lse_all = torch.logsumexp(logits.masked_fill(~keep, neg_inf), 1)
    loss = (lse_all - lse_pos).mean()
    with torch.no_grad():
        top = logits.argmax(1)
        dt = d.gather(1, top[:, None])[:, 0]
        stats = {"acc": float((dt < pos_radius).float().mean()), "acc10": float((dt < 10.0).float().mean()),
                 "n": int(len(valid))}
    return loss, stats

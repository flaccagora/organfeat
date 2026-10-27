#!/usr/bin/env bash
# S2: multimodal descriptor (geometry + lifted DINOv2 on the partial side, missing on the CT side),
# S1 recipe; grid lambda (cross-modal consistency) in {0, 1}, mu (view pairs) = 0.5.
# Selection: registration recall on val in the primary setting (scripts/bench/select.sh).
# Usage (repo root): bash scripts/train/s2_mm.sh [steps]
set -euo pipefail
STEPS="${1:-10000}"
for LAM in 0 1; do  # encoder lr grid {2e-5 (default), 1e-4}: see s2_mm_lam*_lr1e-4
  sbatch -J s2_mm_lam$LAM -t 12:00:00 scripts/leonardo/gpu_run.sbatch \
    python -m ofeat.train.run --config base name=s2_mm_lam$LAM loss.use_vis=true \
      loss.lambda_cross=$LAM optim.steps=$STEPS
done

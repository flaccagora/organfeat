#!/usr/bin/env bash
# S1: geometry-only descriptor (the gate run of docs/protocol.md section 7), encoder-lr grid {2e-5, 1e-4},
# with the recipe from the S1 diagnosis (fp32, head lr 1e-4; docs/s1_findings.md).
# Selection: registration recall on val over the saved checkpoints (scripts/bench/select.sh).
# Usage (repo root): bash scripts/train/s1_geo.sh [steps]
set -euo pipefail
STEPS="${1:-10000}"
for LR in 2e-5 1e-4; do
  sbatch -J s1_geo_lr$LR -t 12:00:00 scripts/leonardo/gpu_run.sbatch \
    python -m ofeat.train.run --config base name=s1_geo_lr$LR loss.use_vis=false \
      optim.encoder_lr=$LR optim.steps=$STEPS
done

#!/usr/bin/env bash
# Checkpoint selection on val (docs/protocol.md section 7): score every saved checkpoint of a run with
# the fixed pipeline on the val livers in the primary setting (depth noise sigma 1 mm at 100 mm, 10 %
# holes): registration recall, inlier ratio as tie-break.
# Usage (repo root): bash scripts/bench/select.sh <run> [geo]   (geo: force geometry-only inference)
set -euo pipefail
RUN="$1"; MODE="${2:-}"
DESCS=()
for c in outputs/train/$RUN/step_*.pt; do
  DESCS+=("learned:$c${MODE:+:$MODE}")
done
sbatch -J sel_$RUN -t 06:00:00 scripts/leonardo/gpu_run.sbatch \
  python scripts/bench/run_bench.py --set val --sigma100 1 --holes 0.1 --desc "${DESCS[@]}"

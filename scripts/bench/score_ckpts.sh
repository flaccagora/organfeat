#!/usr/bin/env bash
# Score training checkpoints on val with the fixed pipeline, one GPU job per checkpoint, in the
# primary setting (sigma 1 mm at 100 mm, 10 % holes); NOISEFREE=1 also noise-free, GEO=1 also
# geometry-only inference of multimodal checkpoints.
# Usage (repo root): bash scripts/bench/score_ckpts.sh outputs/train/<run>/step_<k>.pt [...]
set -euo pipefail
for c in "$@"; do
  run=$(basename "$(dirname "$c")")
  sbatch -J sc_${run} -t 02:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/bench/run_bench.py --set val --sigma100 1 --holes 0.1 --desc "learned:$c"
  if [ "${GEO:-0}" = 1 ]; then  # multimodal checkpoints: also geometry-only inference
    sbatch -J scg_${run} -t 02:00:00 scripts/leonardo/gpu_run.sbatch \
      python scripts/bench/run_bench.py --set val --sigma100 1 --holes 0.1 --desc "learned:$c:geo"
  fi
  if [ "${NOISEFREE:-0}" = 1 ]; then
    sbatch -J sc0_${run} -t 02:00:00 scripts/leonardo/gpu_run.sbatch \
      python scripts/bench/run_bench.py --set val --desc "learned:$c"
  fi
done

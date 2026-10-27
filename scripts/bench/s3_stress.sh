#!/usr/bin/env bash
# S3 stress tests on TEST livers (docs/protocol.md section 7), after the main test, for the final
# descriptors. Base = the primary setting (depth noise sigma 1 mm at 100 mm, 10 % holes); each set
# varies one factor. Depth-noise stress: sigma in {0, 0.5, 1, 2} on the base test views.
# Visible-fraction bins come from the main test results (compare.py --by vis_bin).
# Usage (repo root): bash scripts/bench/s3_stress.sh [<desc> ...]   (default: FPFH r20 + the final learned set)
set -euo pipefail
source scripts/bench/final_descs.sh
if [ $# -eq 0 ]; then set -- fpfh_r20 "${LEARNED[@]}"; fi
for SET in test_tex_heldout test_tex_flat test_tools0 test_tools1 test_tools2 test_tools3 test_deform; do
  sbatch -J st_$SET -t 06:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/bench/run_bench.py --set $SET --sigma100 1 --holes 0.1 --desc "$@"
done
for SG in 0.5 2; do
  sbatch -J st_noise$SG -t 06:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/bench/run_bench.py --set test --sigma100 $SG --holes 0.1 --desc "$@"
done
sbatch -J st_noise0 -t 06:00:00 scripts/leonardo/gpu_run.sbatch \
  python scripts/bench/run_bench.py --set test --desc "$@"

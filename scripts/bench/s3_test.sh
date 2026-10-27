#!/usr/bin/env bash
# S3 main test (scored once): final descriptors + baselines on the 46 test livers, primary setting
# (sigma 1 mm, 10 % holes) and noise-free (secondary). One job per descriptor and setting.
# Usage (repo root): bash scripts/bench/s3_test.sh
set -euo pipefail
source scripts/bench/final_descs.sh
for d in "${LEARNED[@]}" "${BASELINES[@]}"; do
  tag=$(echo "$d" | sed 's#learned:outputs/train/##; s#/step_0*#_#; s#\.pt##; s#[:/+]#_#g')
  sbatch -J t1_$tag -t 04:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/bench/run_bench.py --set test --sigma100 1 --holes 0.1 --desc "$d"
  sbatch -J t0_$tag -t 04:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/bench/run_bench.py --set test --desc "$d"
done

#!/usr/bin/env bash
# R2 training-seed replicates (docs/protocol.md, 2026-10-28): evaluate seeds 1-2 of the four final models,
# each at the step selected for seed 0 (no new selection), on
#   - the synthetic test, primary setting (sigma 1 mm at 100 mm, 10 % holes)
#   - IRCAD v2 (locked; geometry-only: frozen pipeline given + so3, match_ext given)
#   - StereoMIS P2 (--split train, the reported real-video consistency set)
# Usage (repo root, after the replicate trainings finished): bash scripts/bench/seed_replicates.sh
set -euo pipefail
declare -A STEP=([s1_geo_lr1e-4]=8000 [s2_mm_lam0_lr1e-4]=10000 [s1_ppf]=8000 [s2_ppf_lam0]=8000)
D=()  # all variants (multimodal checkpoints with and without images)
G=()  # geometry-only variants (IRCAD has no images)
for r in "${!STEP[@]}"; do
  for s in 1 2; do
    c=outputs/train/${r}_seed$s/step_$(printf %06d "${STEP[$r]}").pt
    [ -f "$c" ] || { echo "missing $c"; exit 1; }
    case $r in
      s2_*) D+=("learned:$c" "learned:$c:geo"); G+=("learned:$c:geo") ;;
      *) D+=("learned:$c"); G+=("learned:$c") ;;
    esac
  done
done
for d in "${D[@]}"; do
  sbatch -J rep_test -t 03:00:00 scripts/leonardo/gpu_run.sbatch \
    python scripts/bench/run_bench.py --set test --sigma100 1 --holes 0.1 --desc "$d"
done
for c in 5 10 15 25; do
  sbatch -J rep_v2_frozen_$c -t 04:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=ircad_v2 scripts/leonardo/gpu_run.sbatch \
    python scripts/eval/ircad_eval.py --split v2 --crops $c --run rep_frozen_crop$c --desc "${G[@]}"
  sbatch -J rep_v2_prior_$c -t 04:00:00 --export=ALL,OFEAT_UNLOCK_EVAL=ircad_v2 scripts/leonardo/gpu_run.sbatch \
    python scripts/eval/ircad_eval.py --split v2 --crops $c --run rep_prior_crop$c --pipeline knn3_prior --init given \
      --desc "${G[@]}"
done
half=$(( ${#D[@]} / 2 ))
sbatch -J rep_smis_a -t 06:00:00 scripts/leonardo/gpu_run.sbatch \
  python scripts/eval/stereomis_consistency.py --split train --desc "${D[@]:0:$half}"
sbatch -J rep_smis_b -t 06:00:00 scripts/leonardo/gpu_run.sbatch \
  python scripts/eval/stereomis_consistency.py --split train --desc "${D[@]:$half}"

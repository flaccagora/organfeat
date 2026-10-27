#!/bin/bash
# Run pytest on a booster GPU node (debug QoS), e.g. for GPU-only tests.
# Usage: sbatch scripts/leonardo/gpu_test.sbatch [pytest args]   (from the repo root)
#SBATCH -A IscrC_FLAC
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpu_test
#SBATCH --output=jobs/out/%x_%j.out
#SBATCH --error=jobs/out/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
scripts/leonardo/run.sh python -m pytest -q "$@"

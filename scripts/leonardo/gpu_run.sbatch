#!/bin/bash
# Run any command in the core env on one booster GPU.
# Usage: sbatch [-t T] -J <name> scripts/leonardo/gpu_run.sbatch python scripts/...
#SBATCH -A IscrC_FLAC
#SBATCH -p boost_usr_prod
#SBATCH --time 01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpu_run
#SBATCH --output=jobs/out/%x_%j.out
#SBATCH --error=jobs/out/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
scripts/leonardo/run.sh "$@"

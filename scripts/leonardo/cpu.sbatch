#!/bin/bash
# Run any command in the core env on a DCGP CPU node (no network; the DCGP budget is on IscrC_FLAC_0).
# Usage: sbatch [-c N --mem M -t T] -J <name> scripts/leonardo/cpu.sbatch python scripts/...
#SBATCH -A IscrC_FLAC_0
#SBATCH -p dcgp_usr_prod
#SBATCH --time 01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=jobs/out/%x_%j.out
#SBATCH --error=jobs/out/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"  # multiprocess jobs: submit with OMP_NUM_THREADS=1
date; echo "== $*"
scripts/leonardo/run.sh "$@"
date

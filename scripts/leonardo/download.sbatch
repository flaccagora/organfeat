#!/bin/bash
# Run one scripts/download/<name>.sh on the serial partition (internet access, no budget, 4 h).
# Login nodes cap a process at 10 min CPU time, too little for unzip/tar of large archives.
# Usage: sbatch -J dl_<name> scripts/leonardo/download.sbatch <name> [args...]
#        (from the repo root; scripts are resumable, resubmit if the 4 h limit is hit)
#SBATCH -A IscrC_FLAC
#SBATCH -p lrd_all_serial
#SBATCH --time 04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=jobs/out/%x_%j.out
#SBATCH --error=jobs/out/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/leonardo/env.sh
NAME="$1"; shift
date; echo "== $NAME $*"
singularity exec -B /leonardo_scratch,/leonardo_work,/leonardo \
  --env "UV_CACHE_DIR=$UV_CACHE_DIR,UV_LINK_MODE=copy,UV_PYTHON_INSTALL_DIR=/opt/python,UV_PYTHON_DOWNLOADS=never,HF_HOME=$HF_HOME,OFEAT_DATA=$OFEAT_DATA,PATH=/opt/python/cpython-3.12.11-linux-x86_64-gnu/bin:/usr/local/bin:/usr/bin:/bin" \
  "$OFEAT_CORE_SIF" bash "scripts/download/$NAME.sh" "$@"
date

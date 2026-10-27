#!/bin/bash
# Compile nvdiffrast into wheels/ against the host cuda/12.6 module (bound under /leonardo/prod),
# inside core.sif (needs the toolchain image: build_core.sh, serial partition, it downloads debs).
# No network needed here (repo cloned, uv cache warm): CPU nodes of the DCGP partition.
# Usage: sbatch scripts/leonardo/build_nvdiffrast.sbatch   (from the repo root)
#SBATCH -A IscrC_FLAC_0  # DCGP budget lives on the _0 account
#SBATCH -p dcgp_usr_prod
#SBATCH --time 01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=build_nvdiffrast
#SBATCH --output=jobs/out/%x_%j.out
#SBATCH --error=jobs/out/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/leonardo/env.sh
module load cuda/12.6
CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
SRC="$OFEAT_SCRATCH/tmp/nvdiffrast"
[ -d "$SRC" ] || { echo "clone https://github.com/NVlabs/nvdiffrast.git to $SRC first (login node)"; exit 1; }
scripts/leonardo/run.sh uv sync --frozen
singularity exec -B /leonardo_scratch,/leonardo_work,/leonardo \
  --env "CC=gcc,CXX=g++,CUDA_HOME=$CUDA_HOME,PATH=$CUDA_HOME/bin:$PWD/.venv/bin:/usr/local/bin:/usr/bin:/bin,VIRTUAL_ENV=$PWD/.venv,TORCH_CUDA_ARCH_LIST=8.0,MAX_JOBS=16,UV_CACHE_DIR=$UV_CACHE_DIR,UV_PYTHON_DOWNLOADS=never" \
  "$OFEAT_CORE_SIF" bash -c "gcc --version | head -1 && nvcc --version | tail -1 && \
    uv build --wheel --no-build-isolation --python $PWD/.venv/bin/python -o wheels/ $SRC"
ls -la wheels/

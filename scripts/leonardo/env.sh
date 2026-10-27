# Source on LEONARDO (login node or job): `source scripts/leonardo/env.sh`
# Data and artefacts live on $SCRATCH (large, purged after 40 days without access); data/* in the
# repo are symlinks to it. Home is nearly full: uv and HF caches go to scratch too.
export OFEAT_SCRATCH="${OFEAT_SCRATCH:-/leonardo_scratch/large/userexternal/$USER/organfeat}"
export OFEAT_DATA="$OFEAT_SCRATCH/data"
export OFEAT_CORE_SIF="$OFEAT_SCRATCH/sif/core.sif"
export OFEAT_CACHE="$OFEAT_DATA/cache"
export WANDB_DIR="$OFEAT_SCRATCH/wandb" WANDB_MODE="${WANDB_MODE:-offline}"
export UV_CACHE_DIR="/leonardo_scratch/large/userexternal/$USER/uv-cache" UV_LINK_MODE=copy
export HF_HOME="/leonardo_scratch/large/userexternal/$USER/HF_DOWNLOADS"
export PATH="/leonardo_scratch/large/userexternal/$USER/bin:$PATH"  # ffmpeg / ffprobe
# Leonardo ships SingularityPRO (no `apptainer` binary); these are honoured by both
export SINGULARITY_TMPDIR="$OFEAT_SCRATCH/tmp" SINGULARITY_CACHEDIR="$OFEAT_SCRATCH/tmp/cache"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
mkdir -p "$SINGULARITY_CACHEDIR" "$WANDB_DIR" 2>/dev/null || true

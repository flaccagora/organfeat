#!/usr/bin/env bash
# Run a command in the core environment on LEONARDO: host glibc 2.28 is too old for open3d 0.20,
# so the uv env (.venv, built inside the image) only works inside core.sif (Ubuntu 24.04).
# Usage: scripts/leonardo/run.sh python -m pytest
#        scripts/leonardo/run.sh uv sync --frozen      (login node: (re)build .venv)
# GPU is passed through automatically when visible (--nv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/scripts/leonardo/env.sh"
NV=(); nvidia-smi -L >/dev/null 2>&1 && NV=(--nv)
ENV="UV_CACHE_DIR=$UV_CACHE_DIR,UV_LINK_MODE=copy,UV_PYTHON_INSTALL_DIR=/opt/python,UV_PYTHON_DOWNLOADS=never,HF_HOME=$HF_HOME,OFEAT_DATA=$OFEAT_DATA,OFEAT_CACHE=$OFEAT_CACHE"
[ "${1:-}" = uv ] || set -- uv run --no-sync "$@"
cd "$ROOT"
exec singularity exec "${NV[@]}" -B /leonardo_scratch,/leonardo_work,/leonardo --env "$ENV" "$OFEAT_CORE_SIF" "$@"

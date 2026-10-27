#!/bin/bash
# GPU smoke test of the core env on a booster node: torch/CUDA, open3d, nvdiffrast, Sonata and DINOv2 forwards.
# Usage: sbatch scripts/leonardo/gpu_smoke.sbatch   (from the repo root)
#SBATCH -A IscrC_FLAC
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --time 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpu_smoke
#SBATCH --output=jobs/out/%x_%j.out
#SBATCH --error=jobs/out/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
nvidia-smi -L
scripts/leonardo/run.sh python - <<'PY'
import torch, open3d, nvdiffrast
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), "open3d", open3d.__version__)
from ofeat._vendor.sonata_encoder import SonataEncoder
enc = SonataEncoder().cuda().eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = enc([torch.rand(4096, 3, device="cuda") for _ in range(2)])
print("sonata", [tuple(o.shape) for o in out])
from ofeat.features.dino import DinoPatches
dino = DinoPatches().cuda()
print("dino", tuple(dino(torch.rand(2, 3, 518, 518, device="cuda")).shape))
PY

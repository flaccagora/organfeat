#!/usr/bin/env bash
# S3 training ablations (docs/protocol.md section 6) around the arm-A multimodal recipe
# (lambda 0, mu 0.5, encoder lr 1e-4, 10 k steps): no visual-modality dropout, 25 / 50 % of the train
# livers, scratch initialisation. Each is selected on val like the main runs (recall, then inlier ratio).
# Visual features off / train-only / train + infer come from s1_geo_lr1e-4 and the :geo / img inference
# of the main multimodal run.
# Usage (repo root): bash scripts/train/s3_ablations.sh
set -euo pipefail
COMMON="loss.use_vis=true loss.lambda_cross=0 optim.encoder_lr=1e-4 optim.steps=10000"
run() { sbatch -J "$1" -t 12:00:00 scripts/leonardo/gpu_run.sbatch python -m ofeat.train.run --config base name="$1" $COMMON "${@:2}"; }
run abl_nodrop loss.vis_drop_sample=0 loss.vis_drop_point=0
run abl_frac25 data.frac_livers=0.25
run abl_frac50 data.frac_livers=0.5
run abl_scratch model.init=scratch

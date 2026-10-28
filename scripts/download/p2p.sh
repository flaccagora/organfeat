#!/usr/bin/env bash
# P2P (Yang et al., JBHI 2025; github.com/zixinyang9109/P2P): in silico (MSD livers, FEM) and in vitro
# (4 phantoms, CT fiducials) partial-to-complete liver datasets, plus the authors' baseline configs and
# weights (LiverMatch, Lepard, RegTR, RoITr, Go-ICP). Phase-2 external benchmarks E3 / E4
# (docs/miccai_plan.md, docs/protocol.md 2026-10-28). Licence: the LiverMatch README states the P2P
# dataset is released under CC BY-SA 4.0 (the P2P README has no licence statement; checked again below).
# Run on the serial partition (internet): sbatch -J dl_p2p scripts/leonardo/download.sbatch p2p
set -euo pipefail
DST="${OFEAT_DATA:?source scripts/leonardo/env.sh}/raw/p2p"
mkdir -p "$DST/data" "$DST/baselines"
CA=/etc/ssl/certs/ca-certificates.crt
for v in REQUESTS_CA_BUNDLE SSL_CERT_FILE CURL_CA_BUNDLE; do
  if [ -n "${!v:-}" ] && [ ! -f "${!v}" ]; then export "$v=$CA"; fi
done
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CA}" SSL_CERT_FILE="${SSL_CERT_FILE:-$CA}"
[ -d "$DST/repo/.git" ] || git clone --depth 1 https://github.com/zixinyang9109/P2P "$DST/repo"
git -C "$DST/repo" log -1 --format='P2P repo commit %H %cd' | tee "$DST/SOURCE.txt"
ls "$DST/repo"; ls "$DST/repo" | grep -i licen || echo "no LICENSE file in the repo"
GD="uv tool run --from gdown==5.2.0 gdown"  # uvx is not installed in core.sif
$GD --folder "https://drive.google.com/drive/folders/1CpcMFqaiyg3eVnSEItCi1N8hmEeDopkd" -O "$DST/data" --remaining-ok
$GD "https://drive.google.com/uc?id=11gGAz-7gM-41nCwhu8mebgR9Qm9IygU0" -O "$DST/baselines/" || echo "baselines download failed"
find "$DST/data" "$DST/baselines" -maxdepth 3 | head -80 || true
du -sh "$DST"/*

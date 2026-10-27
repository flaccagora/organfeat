#!/usr/bin/env bash
# P2I-LReg (Self-P2IR, arXiv 2504.15152) from Kaggle: junezhou001/p2i-lreg.
# Needs a Kaggle API token in ~/.kaggle/ (access_token, placed by the user; never read by scripts).
# Run on the serial partition (internet): sbatch -J dl_p2i_lreg scripts/leonardo/download.sbatch p2i_lreg
# Data is extracted to data/raw/p2i_lreg/; the audit (scripts/preprocess/audit_p2i_lreg.py) decides the
# dev / locked patient split before any descriptor is run on it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DST="${OFEAT_DATA:?source scripts/leonardo/env.sh}/raw/p2i_lreg"
mkdir -p "$DST"
cd "$DST"
# the host exports RHEL CA-bundle paths that do not exist in the Ubuntu container
CA=/etc/ssl/certs/ca-certificates.crt
for v in REQUESTS_CA_BUNDLE SSL_CERT_FILE CURL_CA_BUNDLE; do
  if [ -n "${!v:-}" ] && [ ! -f "${!v}" ]; then export "$v=$CA"; fi
done
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CA}"
"$ROOT/.venv/bin/kaggle" datasets download junezhou001/p2i-lreg -p "$DST"
for z in "$DST"/*.zip; do
  [ -e "$z" ] || continue
  python3 -m zipfile -e "$z" "$DST/" && rm "$z"
done
find "$DST" -maxdepth 2 | head -50 || true  # head closes the pipe early (SIGPIPE under pipefail)
du -sh "$DST"

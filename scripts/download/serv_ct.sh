#!/usr/bin/env bash
# SERV-CT (Edwards et al., MedIA 2022; figshare doi:10.5522/04/26352199, CC BY 4.0): 16 stereo-endoscopic
# pairs of two ex vivo porcine torsos with CT-derived reference depth, CT surfaces and the endoscope
# alignment. Candidate real-data test with RGB + CT ground truth (docs/r3b_datasets.md).
# Run on the serial partition (internet): sbatch -J dl_serv_ct scripts/leonardo/download.sbatch serv_ct
set -euo pipefail
DST="${OFEAT_DATA:?source scripts/leonardo/env.sh}/raw/serv_ct"
mkdir -p "$DST"
cd "$DST"
CA=/etc/ssl/certs/ca-certificates.crt
for v in REQUESTS_CA_BUNDLE SSL_CERT_FILE CURL_CA_BUNDLE; do
  if [ -n "${!v:-}" ] && [ ! -f "${!v}" ]; then export "$v=$CA"; fi
done
# SERV-CT-ALL.zip (389,586,784 bytes): the full version with CT surfaces and alignment software
[ -f SERV-CT-ALL.zip ] || curl -fL --retry 3 -o SERV-CT-ALL.zip https://ndownloader.figshare.com/files/47855530
[ "$(stat -c %s SERV-CT-ALL.zip)" = 389586784 ] || { echo "size mismatch"; exit 1; }
mkdir -p extracted
python3 -m zipfile -e SERV-CT-ALL.zip extracted/
find extracted -maxdepth 3 | head -60 || true
du -sh "$DST"

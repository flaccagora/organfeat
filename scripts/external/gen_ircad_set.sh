#!/usr/bin/env bash
# Generate one deformed-IRCAD set with the authors' generator (scripts/external/gen_deformed_ircad.py),
# dropping livers that cannot supply the crop's visible points (generator prints FAILLIVER; see
# docs/protocol.md, 2026-10-28) and retrying with the rest.
# Usage: bash scripts/external/gen_ircad_set.sh <crop 05|10|15|25> <per_liver> <seed_base> <out.h5> <livers...>
set -uo pipefail
CROP=$1 PER=$2 SEED=$3 OUT=$4; shift 4
L="$*"
LOG=jobs/out/$(basename "$OUT" .h5).log
for it in $(seq 1 20); do
  python scripts/external/gen_deformed_ircad.py --crop-ratio 0.$CROP --livers $L --per-liver "$PER" \
    --seed-base "$SEED" --out "$OUT" > "$LOG" 2>&1
  rc=$?
  [ $rc = 0 ] && { echo "$OUT: livers $L; redraws $(grep -c redraw "$LOG")"; exit 0; }
  f=$(grep -o "FAILLIVER [0-9]*" "$LOG" | awk '{print $2}')
  [ -n "$f" ] || { echo "$OUT: failure rc=$rc"; tail -5 "$LOG"; exit 1; }
  echo "$OUT: dropping liver $f"; L=$(echo $L | tr ' ' '\n' | grep -vx "$f" | tr '\n' ' ')
done
exit 1

# Base test set + all stress sets (paired cameras). sbatch -J views_test_sets -c 48 --mem 128G -t 02:00:00 scripts/leonardo/cpu.sbatch bash scripts/synth/make_test_sets.sh
set -e
for s in test test_tex_heldout test_tex_flat test_tools0 test_tools1 test_tools2 test_tools3 test_deform; do
  python scripts/synth/make_views.py --set $s --workers 46 | tail -n 2
done

#!/usr/bin/env bash
# One-off copy of the datasets already on disk (from the correspondences project's scratch) into this
# project's own data dir. Copies only what organfeat reads: no DePoLL videos / DICOM, no Project-10
# token caches. Resumable (existing files are skipped). Run: sbatch -J dl_copy_local scripts/leonardo/download.sbatch copy_local
set -euo pipefail
SRC=/leonardo_scratch/large/userexternal/$USER/correspondences/data
DST="${OFEAT_DATA:?source scripts/leonardo/env.sh}"

# DePoLL (LOCKED; CC BY-NC-ND 4.0): reference model + markers, transforms, per-state markers and surfaces
D=$DST/raw/depoll
mkdir -p "$D"
cp -a --update=none "$SRC/raw/depoll/"{LICENSE.txt,md5sums.txt,dataset_and_evaluation_description.pdf,coordinate_transforms} "$D/"
mkdir -p "$D/reference"
cp -a --update=none "$SRC/raw/depoll/reference/"{README.txt,referenceModel} "$D/reference/"
for s in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
  mkdir -p "$D/$s"
  cp -a --update=none "$SRC/raw/depoll/$s/"{README.txt,markers,surfaces} "$D/$s/"
done

# StereoMIS liver-bearing sequences: calibration, kinematics, tool masks, split csv; extracted left
# frames + SGBM depth + rectified tool masks + meta (correspondences scripts/preprocess/extract_stereomis.py, step 6)
for q in P2_0 P2_1 P2_2 P2_3 P2_4 P2_5 P3; do
  R=$DST/raw/stereomis/StereoMIS_0_0_1/$q
  mkdir -p "$R" "$DST/processed/stereomis/$q"
  cp -a --update=none "$SRC/raw/stereomis/StereoMIS_0_0_1/$q/"{StereoCalibration.ini,groundtruth.txt,masks} "$R/"
  cp -a --update=none "$SRC/raw/stereomis/StereoMIS_0_0_1/$q/"*_split.csv "$R/"
  cp -a --update=none "$SRC/processed/stereomis/$q/"{left,depth,tool,meta.json} "$DST/processed/stereomis/$q/"
done
du -sh "$D" "$DST/raw/stereomis" "$DST/processed/stereomis"

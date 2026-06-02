#!/bin/bash
#PBS -N era5_wind_profiles
#PBS -J 0-10
#PBS -l select=1:ncpus=1:mem=1GB
#PBS -l walltime=01:00:00
#PBS -q main
#PBS -A UMIC0126
#PBS -j oe
#PBS -o logs/era5_wind_05_14_26.out

cd "$PBS_O_WORKDIR"
mkdir -p logs

YEAR=$((2015 + PBS_ARRAY_INDEX))
# ── Environment ───────────────────────────────────────────────────────────────
module load conda
conda run -p /glade/work/rbhandarkar/conda-envs/assetra_gl python generate_wind_profiles_era5.py --year $YEAR

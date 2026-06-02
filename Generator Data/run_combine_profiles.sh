#!/bin/bash
#PBS -N combine_profiles
#PBS -J 0-18
#PBS -l select=1:ncpus=1:mem=5GB
#PBS -l walltime=01:00:00
#PBS -q main
#PBS -A UMIC0126
#PBS -j oe
#PBS -o logs/combine_profiles.out

cd "$PBS_O_WORKDIR"
mkdir -p logs

MEMBER=$(printf "%03d" $((PBS_ARRAY_INDEX + 1)))

module load conda
conda run -p /glade/work/rbhandarkar/conda-envs/assetra_gl \
    python combine_profiles.py \
        --base-dir   "/glade/work/rbhandarkar/Inputs/Wind Profiles/" \
        --output-dir "Wind Profiles Clean" \
        --ensemble   "$MEMBER"

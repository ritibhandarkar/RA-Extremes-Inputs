#!/bin/bash
#PBS -N wind_profiles
#PBS -J 0-588
#PBS -l select=1:ncpus=1:mem=1GB
#PBS -l walltime=00:10:00
#PBS -q main
#PBS -A UMIC0126
#PBS -j oe
#PBS -o logs/wind_05_08_26.out

cd "$PBS_O_WORKDIR"
mkdir -p logs

# RCP85: indices 0–278   (31 years × 9 members, 002–010)
# RCP60: indices 279–588 (31 years × 10 members, 001–010)
if [ "$PBS_ARRAY_INDEX" -lt 279 ]; then
    RCP=RCP85
    N_MEMBERS=9
    MEMBER_START=2
    REMAINDER=$PBS_ARRAY_INDEX
else
    RCP=RCP60
    N_MEMBERS=10
    MEMBER_START=1
    REMAINDER=$((PBS_ARRAY_INDEX - 279))
fi

YEAR=$((2015 + REMAINDER / N_MEMBERS))
MEMBER=$(printf "%03d" $((REMAINDER % N_MEMBERS + MEMBER_START)))

# ── Environment ───────────────────────────────────────────────────────────────
module load conda
conda run -p /glade/work/rbhandarkar/conda-envs/assetra_gl python generate_wind_profiles.py --year $YEAR --ensemble $MEMBER --rcp $RCP

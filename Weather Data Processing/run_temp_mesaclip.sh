#!/bin/bash
#PBS -N mesaclip_temp
#PBS -J 0-588
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=00:30:00
#PBS -q main
#PBS -A UMIC0126
#PBS -j oe
#PBS -o logs/temp_05_15_26.out

cd "$PBS_O_WORKDIR"
mkdir -p logs

# RCP85: indices 0–98    (11 years × 9 members, 002–010)
# RCP60: indices 99–208  (11 years × 10 members, 001–010)
if [ "$PBS_ARRAY_INDEX" -lt 99 ]; then
    RCP=RCP85
    N_MEMBERS=9
    MEMBER_START=2
    REMAINDER=$PBS_ARRAY_INDEX
else
    RCP=RCP60
    N_MEMBERS=10
    MEMBER_START=1
    REMAINDER=$((PBS_ARRAY_INDEX - 99))
fi

YEAR=$((2015 + REMAINDER / N_MEMBERS))
MEMBER=$(printf "%03d" $((REMAINDER % N_MEMBERS + MEMBER_START)))

cd "/glade/u/home/rbhandarkar/Inputs/Weather Data Processing"
module load conda
conda run -p /glade/work/rbhandarkar/conda-envs/assetra_gl python process_mesaclip.py --year $YEAR --rcp $RCP --ens $MEMBER

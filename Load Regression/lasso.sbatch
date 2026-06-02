#!/bin/bash
#SBATCH --job-name=lasso_cv
#SBATCH --array=0-47%36               # 6 zones × 8 folds = 48 tasks, max 36 concurrent
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --account=mtcraig98
#SBATCH --output=logs/lasso_%A_%a.out
#SBATCH --error=logs/lasso_%A_%a.err

# ── Environment ───────────────────────────────────────────────────────────────
module load python3.11-anaconda/2024.02
source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate assetra_gl

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="/nfs/turbo/seas-mtcraig-climate/ritib/Inputs/Load Regression"
DATA_DIR="${SCRIPT_DIR}"
OUT_DIR="${SCRIPT_DIR}/lasso_results"

mkdir -p "${OUT_DIR}"

# ── Run worker ────────────────────────────────────────────────────────────────
echo "Starting task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"

python "${SCRIPT_DIR}/lasso_worker.py" \
    --task-id  "${SLURM_ARRAY_TASK_ID}" \
    --data-dir "${DATA_DIR}" \
    --outdir   "${OUT_DIR}"

EXIT_CODE=$?
echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date) with exit code ${EXIT_CODE}"
sleep 10
echo "----- Resource usage for this task -----"
sacct -j ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
      --format=JobID,JobName,MaxRSS,Elapsed,State

exit ${EXIT_CODE}


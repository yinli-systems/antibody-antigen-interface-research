#!/bin/bash
#SBATCH --job-name=ab-corresp-v2
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
set -euo pipefail
: "${SOURCE_ROOT:?}" "${ATTEMPT_ROOT:?}" "${PLAN_SHA256:?}"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/ssd/scxi253/repr_extract/venv/bin/python3
cd "$SOURCE_ROOT"
"$PY" -u extensions/correspondence_v2/run.py run \
 --plan "$ATTEMPT_ROOT/PLAN.json" --plan-sha256 "$PLAN_SHA256" \
 --task "${SLURM_ARRAY_TASK_ID:?}" --output "$ATTEMPT_ROOT/out"

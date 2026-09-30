#!/bin/bash
#SBATCH --job-name=struct-plm-train
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00

set -euo pipefail
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${ATTEMPT_ROOT:?ATTEMPT_ROOT is required}"
: "${PLAN_SHA256:?PLAN_SHA256 is required}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array or set task ID explicitly}"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

PYTHON=/ssd/scxi253/repr_extract/venv/bin/python3
cd "${SOURCE_ROOT}"
"${PYTHON}" -u extensions/frontier_multimodel_v1/run_structural_training.py \
  --plan "${ATTEMPT_ROOT}/PLAN.json" \
  --plan-sha256 "${PLAN_SHA256}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --attempt-root "${ATTEMPT_ROOT}"

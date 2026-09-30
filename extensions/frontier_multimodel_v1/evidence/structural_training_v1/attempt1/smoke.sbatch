#!/bin/bash
#SBATCH --job-name=struct-plm-smoke
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt1/logs/smoke-%j.out
#SBATCH --error=/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt1/logs/smoke-%j.err
set -euo pipefail
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
S=/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/source_exact_58740ea
A=/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt1
cd "$S"
/ssd/scxi253/repr_extract/venv/bin/python3 -u extensions/frontier_multimodel_v1/run_structural_training.py \
  --plan "$A/PLAN.json" \
  --plan-sha256 895735a638b8774342bfd712b6095dc4a47ab5d7dd2de950a6e60d09869206ff \
  --task-id 6 --attempt-root "$A" --smoke

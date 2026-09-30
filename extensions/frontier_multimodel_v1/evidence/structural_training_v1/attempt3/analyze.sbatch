#!/bin/bash
#SBATCH --job-name=struct-plm-analyze
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt3/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt3/logs/analyze-%j.err
set -euo pipefail
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
cd "/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/source_exact_3eb4293"
/ssd/scxi253/repr_extract/venv/bin/python3 -u extensions/frontier_multimodel_v1/aggregate_structural_training.py --plan "/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt3/PLAN.json" --plan-sha256 24d6c6cfb2cee44b9cc289ac8721b3ad183321db11a059c14b29c878dd7cea12 --attempt-root "/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt3" --output "/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt3/ANALYSIS.json"

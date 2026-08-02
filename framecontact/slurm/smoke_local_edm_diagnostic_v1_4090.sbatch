#!/usr/bin/env bash
#SBATCH --job-name=fc-edm-smoke
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-edm-smoke-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-edm-smoke-%j.err

set -euo pipefail

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/edm-diagnostic-source-v1"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3
output="$base/smoke/local-edm-diagnostic-v1-4090-${SLURM_JOB_ID}.json"

cd "$source_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$source_root/src"

"$python_bin" scripts/smoke_local_edm_diagnostic.py --output "$output"
sha256sum "$output"

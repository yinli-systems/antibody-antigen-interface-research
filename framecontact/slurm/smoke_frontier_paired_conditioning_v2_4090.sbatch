#!/usr/bin/env bash
#SBATCH --job-name=fc-frontier-smoke-v2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-frontier-smoke-v2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-frontier-smoke-v2-%j.err

set -euo pipefail

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/frontier-source-v2"
source_manifest="$base/source-manifests/frontier-source-v2.json"
source_manifest_sha256="$1"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3
output="$base/smoke/frontier-paired-conditioning-v2-4090-${SLURM_JOB_ID}.json"

cd "$source_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$source_root/src"

"$python_bin" scripts/freeze_training_source_tree.py verify \
  --source-root "$source_root" \
  --manifest "$source_manifest" \
  --manifest-sha256 "$source_manifest_sha256"

"$python_bin" scripts/smoke_frontier_paired_conditioning.py \
  --output "$output" \
  --source-manifest-sha256 "$source_manifest_sha256" \
  --device cuda

sha256sum "$output"

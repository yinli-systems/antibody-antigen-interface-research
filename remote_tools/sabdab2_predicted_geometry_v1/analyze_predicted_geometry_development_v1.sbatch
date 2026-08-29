#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-analyze
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/analyze-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly analyzer=${root}/tools/sabdab2_predicted_geometry_v1/analyze_predicted_geometry_development_v1.py
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
"${python_bin}" "${analyzer}" \
  --campaign-root "${campaign}" \
  --bound-reference "${root}/results/development/sabdab2_edge_topk2_task_routed_full_v1/EDGE_TOPK2_TASK_ROUTED_FULL_DEVELOPMENT_ANALYSIS_V1.json" \
  --output "${campaign}/PREDICTED_GEOMETRY_DEVELOPMENT_ANALYSIS_V1.json" \
  --receipt-output "${campaign}/PREDICTED_GEOMETRY_DEVELOPMENT_ANALYSIS_V1_RECEIPT.json"

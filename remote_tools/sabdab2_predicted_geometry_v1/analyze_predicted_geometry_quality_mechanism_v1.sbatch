#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-mechanism
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/analyze-geometry-quality-mechanism-v1-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/analyze-geometry-quality-mechanism-v1-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly analyzer=${root}/tools/sabdab2_predicted_geometry_v1/analyze_predicted_geometry_quality_mechanism_v1.py

test "${SLURM_JOB_PARTITION}" = gpu_4090
"${python_bin}" "${analyzer}" \
  --predicted-structure-root "${campaign}/data/structure_cache_v1" \
  --bound-structure-root "${root}/data/framecontact_20260730/sabdab2/structure_cohort_v2" \
  --split "${campaign}/data/structure_cache_v1/DEVELOPMENT_SPLIT_VIEW.csv" \
  --predicted-analysis "${campaign}/PREDICTED_GEOMETRY_DEVELOPMENT_ANALYSIS_DISTANCE_LABEL_RECOVERY_V2.json" \
  --bound-analysis "${root}/results/development/sabdab2_edge_topk2_task_routed_full_v1/EDGE_TOPK2_TASK_ROUTED_FULL_DEVELOPMENT_ANALYSIS_V1.json" \
  --output "${campaign}/PREDICTED_GEOMETRY_QUALITY_MECHANISM_ANALYSIS_V1.json" \
  --receipt-output "${campaign}/PREDICTED_GEOMETRY_QUALITY_MECHANISM_ANALYSIS_V1_RECEIPT.json"

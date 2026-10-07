#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-plddt70-analyze
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/analyze-confidence-geometry-plddt70-v1-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/analyze-confidence-geometry-plddt70-v1-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools=${root}/remote_tools/sabdab2_confidence_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' \
  a23a3282788d8699239b0e5a54ed3d19926789587ad5099e65992b1d307a321d "${tools}/analyze_confidence_geometry_development_v1.py" \
  1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c "${campaign}/PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json" \
  | sha256sum --check --strict
"${python_bin}" "${tools}/analyze_confidence_geometry_development_v1.py" \
  --campaign-root "${campaign}" \
  --control-analysis "${campaign}/PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json" \
  --output "${campaign}/PREDICTED_GEOMETRY_CONFIDENCE_GEOMETRY_PLDDT70_ANALYSIS_V1.json" \
  --receipt-output "${campaign}/PREDICTED_GEOMETRY_CONFIDENCE_GEOMETRY_PLDDT70_ANALYSIS_V1_RECEIPT.json"

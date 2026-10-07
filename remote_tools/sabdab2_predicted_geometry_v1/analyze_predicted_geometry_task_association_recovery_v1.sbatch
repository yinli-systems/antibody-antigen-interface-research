#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-task-assoc-recovery
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/task-association-recovery-v1-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/task-association-recovery-v1-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools=${root}/tools/sabdab2_predicted_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly mechanism=${campaign}/PREDICTED_GEOMETRY_QUALITY_MECHANISM_ANALYSIS_V1.json
readonly mechanism_receipt=${campaign}/PREDICTED_GEOMETRY_QUALITY_MECHANISM_ANALYSIS_V1_RECEIPT.json
readonly output=${campaign}/PREDICTED_GEOMETRY_TASK_ASSOCIATION_ANALYSIS_V1.json
readonly receipt=${campaign}/PREDICTED_GEOMETRY_TASK_ASSOCIATION_RECEIPT_V1.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}"
test ! -e "${receipt}"
printf '%s  %s\n' \
  86ef7dff66b4f636f929f28728cf300f360041577741c891059ff601f9c5f272 "${tools}/GEOMETRY_TASK_ASSOCIATION_PLAN_V1.json" \
  b8c8f3ea0ae35afec9341c5145d9467c92faf5fdcda09e8776bd562247f5c4f5 "${mechanism}" \
  569b06bf732866c4022646943e730621bdf9d77d3f2441d79796b424875d7a95 "${mechanism_receipt}" \
  c9be09ae831f6e9a53c4e99e6186564a16c7337b44fdad45d2187c8e992e0f16 "${tools}/analyze_predicted_geometry_task_association_v1.py" \
  | sha256sum --check --strict

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${tools}/analyze_predicted_geometry_task_association_v1.py" \
  --plan "${tools}/GEOMETRY_TASK_ASSOCIATION_PLAN_V1.json" \
  --mechanism-analysis "${mechanism}" \
  --mechanism-receipt "${mechanism_receipt}" \
  --metrics-root "${campaign}/quality_task_metrics_v1" \
  --output "${output}" \
  --receipt-output "${receipt}"

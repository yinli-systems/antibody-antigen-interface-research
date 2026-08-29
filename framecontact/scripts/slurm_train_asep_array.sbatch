#!/usr/bin/env bash
#SBATCH --job-name=fc-asep-dev
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --array=0-39%8
#SBATCH --output=fc-asep-dev-%A_%a.out
#SBATCH --error=fc-asep-dev-%A_%a.err

set -euo pipefail

: "${FRAMECONTACT_REPOSITORY_ROOT:?missing FRAMECONTACT_REPOSITORY_ROOT}"
: "${FRAMECONTACT_ASEP_CACHE_ROOT:?missing FRAMECONTACT_ASEP_CACHE_ROOT}"
: "${FRAMECONTACT_ASEP_CACHE_MANIFEST_SHA256:?missing FRAMECONTACT_ASEP_CACHE_MANIFEST_SHA256}"
: "${FRAMECONTACT_PROTOCOL_RECEIPT_SHA256:?missing FRAMECONTACT_PROTOCOL_RECEIPT_SHA256}"
: "${FRAMECONTACT_TRAINING_PANEL:?missing FRAMECONTACT_TRAINING_PANEL}"
: "${FRAMECONTACT_TRAINING_PANEL_SHA256:?missing FRAMECONTACT_TRAINING_PANEL_SHA256}"
: "${FRAMECONTACT_VALIDATION_PANEL:?missing FRAMECONTACT_VALIDATION_PANEL}"
: "${FRAMECONTACT_VALIDATION_PANEL_SHA256:?missing FRAMECONTACT_VALIDATION_PANEL_SHA256}"
: "${FRAMECONTACT_RUN_ROOT:?missing FRAMECONTACT_RUN_ROOT}"
: "${FRAMECONTACT_PYTHON_BIN:?missing FRAMECONTACT_PYTHON_BIN}"

seeds=(17 42 73 101 137)
experiments=(
  sequence_dense
  spatial_topology
  distance
  distance_direction
  full_pose
  frozen_frame_ipa
  framecontact
  framecontact_interleaved
)
task_index="${SLURM_ARRAY_TASK_ID}"
if (( task_index < 0 || task_index >= 40 )); then
  echo "SLURM_ARRAY_TASK_ID is outside the frozen 0-39 matrix" >&2
  exit 2
fi
experiment="${experiments[$((task_index / 5))]}"
seed="${seeds[$((task_index % 5))]}"

repository_root="${FRAMECONTACT_REPOSITORY_ROOT}"
protocol_path="${FRAMECONTACT_PROTOCOL:-${repository_root}/protocols/framecontact_asep_epitope_group_v1.json}"
protocol_receipt="${FRAMECONTACT_PROTOCOL_RECEIPT:-${repository_root}/protocols/FRAMECONTACT_ASEP_EPITOPE_GROUP_V1_RECEIPT.json}"
output_root="${FRAMECONTACT_RUN_ROOT}/${experiment}/seed-${seed}"
if [[ -f "${output_root}/RUN_RECEIPT.json" && -f "${output_root}/FRAMECONTACT_DEVELOPMENT_RESULT.json" ]]; then
  exit 0
fi
if [[ -e "${output_root}" ]]; then
  echo "Partial output exists without a complete receipt: ${output_root}" >&2
  exit 2
fi

cd "${repository_root}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${FRAMECONTACT_PYTHON_BIN}" scripts/train_framecontact.py \
  --dataset-backend asep \
  --asep-cache-root "${FRAMECONTACT_ASEP_CACHE_ROOT}" \
  --asep-cache-manifest-sha256 "${FRAMECONTACT_ASEP_CACHE_MANIFEST_SHA256}" \
  --asep-feature-mode "${FRAMECONTACT_ASEP_FEATURE_MODE:-official_esm2_35m}" \
  --protocol "${protocol_path}" \
  --protocol-receipt "${protocol_receipt}" \
  --protocol-receipt-sha256 "${FRAMECONTACT_PROTOCOL_RECEIPT_SHA256}" \
  --training-partner-panel "${FRAMECONTACT_TRAINING_PANEL}" \
  --training-partner-panel-sha256 "${FRAMECONTACT_TRAINING_PANEL_SHA256}" \
  --validation-partner-panel "${FRAMECONTACT_VALIDATION_PANEL}" \
  --validation-partner-panel-sha256 "${FRAMECONTACT_VALIDATION_PANEL_SHA256}" \
  --output-root "${output_root}" \
  --seed "${seed}" \
  --experiment "${experiment}" \
  --epochs 100 \
  --early-stopping-patience 15 \
  --device cuda \
  --precision bf16

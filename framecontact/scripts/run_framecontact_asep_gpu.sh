#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  FRAMECONTACT_REPOSITORY_ROOT
  FRAMECONTACT_ASEP_CACHE_ROOT
  FRAMECONTACT_ASEP_CACHE_MANIFEST_SHA256
  FRAMECONTACT_PROTOCOL_RECEIPT_SHA256
  FRAMECONTACT_TRAINING_PANEL
  FRAMECONTACT_TRAINING_PANEL_SHA256
  FRAMECONTACT_VALIDATION_PANEL
  FRAMECONTACT_VALIDATION_PANEL_SHA256
  FRAMECONTACT_RUN_ROOT
  FRAMECONTACT_PYTHON_BIN
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Missing required environment variable: ${variable_name}" >&2
    exit 2
  fi
done

repository_root="${FRAMECONTACT_REPOSITORY_ROOT}"
protocol_path="${FRAMECONTACT_PROTOCOL:-${repository_root}/protocols/framecontact_asep_epitope_group_v1.json}"
protocol_receipt="${FRAMECONTACT_PROTOCOL_RECEIPT:-${repository_root}/protocols/FRAMECONTACT_ASEP_EPITOPE_GROUP_V1_RECEIPT.json}"
gpu_pattern="${FRAMECONTACT_GPU_PATTERN:-A100}"
read -r -a seeds <<< "${FRAMECONTACT_SEEDS:-17 42 73 101 137}"
read -r -a experiments <<< "${FRAMECONTACT_EXPERIMENTS:-sequence_dense spatial_topology distance distance_direction full_pose frozen_frame_ipa framecontact framecontact_interleaved}"
if [[ "${#seeds[@]}" -ne 5 ]]; then
  echo "Exactly five pre-registered seeds are required" >&2
  exit 2
fi

cd "${repository_root}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ ! "${gpu_name}" =~ ${gpu_pattern} ]]; then
  echo "GPU '${gpu_name}' does not match required pattern '${gpu_pattern}'" >&2
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
mkdir -p "${FRAMECONTACT_RUN_ROOT}"

if [[ "${FRAMECONTACT_VALIDATE:-1}" == "1" ]]; then
  PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${FRAMECONTACT_PYTHON_BIN}" -m pytest -q
  PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${FRAMECONTACT_PYTHON_BIN}" scripts/smoke_asep_framecontact_gpu.py \
      --cache-root "${FRAMECONTACT_ASEP_CACHE_ROOT}" \
      --cache-manifest-sha256 "${FRAMECONTACT_ASEP_CACHE_MANIFEST_SHA256}" \
      --output "${FRAMECONTACT_RUN_ROOT}/ASEP_GPU_SMOKE.json" \
      --device cuda --precision bf16
fi

for experiment in "${experiments[@]}"; do
for seed in "${seeds[@]}"; do
  output_root="${FRAMECONTACT_RUN_ROOT}/${experiment}/seed-${seed}"
  if [[ -f "${output_root}/RUN_RECEIPT.json" && -f "${output_root}/FRAMECONTACT_DEVELOPMENT_RESULT.json" ]]; then
    continue
  fi
  if [[ -e "${output_root}" ]]; then
    echo "Partial output exists without a complete receipt: ${output_root}" >&2
    exit 2
  fi
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
    --seed "${seed}" --experiment "${experiment}" \
    --epochs 100 --early-stopping-patience 15 \
    --device cuda --precision bf16
done
done

echo "All AsEP development experiment-by-seed paths are complete; official test was not opened."

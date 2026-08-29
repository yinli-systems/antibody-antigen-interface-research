#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  FRAMECONTACT_REPOSITORY_ROOT
  FRAMECONTACT_STRUCTURE_CACHE_ROOT
  FRAMECONTACT_CONTACT_CACHE_ROOT
  FRAMECONTACT_EMBEDDING_ROOT
  FRAMECONTACT_EMBEDDING_MANIFEST_SHA256
  FRAMECONTACT_SPLIT_MANIFEST
  FRAMECONTACT_SPLIT_MANIFEST_SHA256
  FRAMECONTACT_STRUCTURE_MANIFEST_SHA256
  FRAMECONTACT_CONTACT_MANIFEST_SHA256
  FRAMECONTACT_PROTOCOL_RECEIPT_SHA256
  FRAMECONTACT_SOURCE_MANIFEST
  FRAMECONTACT_SOURCE_MANIFEST_SHA256
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
protocol_path="${FRAMECONTACT_PROTOCOL:-${repository_root}/protocols/framecontact_primary_v5.json}"
protocol_receipt="${FRAMECONTACT_PROTOCOL_RECEIPT:-${repository_root}/protocols/FRAMECONTACT_PROTOCOL_V5_RECEIPT.json}"
gpu_pattern="${FRAMECONTACT_GPU_PATTERN:-A100}"
seed_text="${FRAMECONTACT_SEEDS:-17 42 73 101 137}"
read -r -a seeds <<< "${seed_text}"
experiment_text="${FRAMECONTACT_EXPERIMENTS:-sequence_dense spatial_topology distance distance_direction full_pose frozen_frame_ipa framecontact framecontact_interleaved}"
read -r -a experiments <<< "${experiment_text}"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${FRAMECONTACT_RUN_ROOT}"

if [[ "${FRAMECONTACT_VALIDATE:-1}" == "1" ]]; then
  PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${FRAMECONTACT_PYTHON_BIN}" -m pytest -q
  PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${FRAMECONTACT_PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
(x @ x.T).float().mean().backward()
torch.cuda.synchronize()
print({"cuda_smoke": "passed", "gpu": torch.cuda.get_device_name(0)})
PY
fi

for experiment in "${experiments[@]}"; do
for seed in "${seeds[@]}"; do
  output_root="${FRAMECONTACT_RUN_ROOT}/${experiment}/seed-${seed}"
  if [[ -f "${output_root}/RUN_RECEIPT.json" && -f "${output_root}/FRAMECONTACT_DEVELOPMENT_RESULT.json" ]]; then
    echo "Seed ${seed}: verified result paths already exist; skipping"
    continue
  fi
  if [[ -e "${output_root}" ]]; then
    echo "Seed ${seed}: partial output exists without a complete receipt: ${output_root}" >&2
    exit 2
  fi
  "${FRAMECONTACT_PYTHON_BIN}" scripts/train_framecontact.py \
    --dataset-backend sabdab2 \
    --structure-cache-root "${FRAMECONTACT_STRUCTURE_CACHE_ROOT}" \
    --contact-cache-root "${FRAMECONTACT_CONTACT_CACHE_ROOT}" \
    --embedding-root "${FRAMECONTACT_EMBEDDING_ROOT}" \
    --embedding-manifest-sha256 "${FRAMECONTACT_EMBEDDING_MANIFEST_SHA256}" \
    --split-manifest "${FRAMECONTACT_SPLIT_MANIFEST}" \
    --split-manifest-sha256 "${FRAMECONTACT_SPLIT_MANIFEST_SHA256}" \
    --structure-manifest-sha256 "${FRAMECONTACT_STRUCTURE_MANIFEST_SHA256}" \
    --contact-manifest-sha256 "${FRAMECONTACT_CONTACT_MANIFEST_SHA256}" \
    --protocol "${protocol_path}" \
    --protocol-receipt "${protocol_receipt}" \
    --protocol-receipt-sha256 "${FRAMECONTACT_PROTOCOL_RECEIPT_SHA256}" \
    --source-manifest "${FRAMECONTACT_SOURCE_MANIFEST}" \
    --source-manifest-sha256 "${FRAMECONTACT_SOURCE_MANIFEST_SHA256}" \
    --training-partner-panel "${FRAMECONTACT_TRAINING_PANEL}" \
    --training-partner-panel-sha256 "${FRAMECONTACT_TRAINING_PANEL_SHA256}" \
    --validation-partner-panel "${FRAMECONTACT_VALIDATION_PANEL}" \
    --validation-partner-panel-sha256 "${FRAMECONTACT_VALIDATION_PANEL_SHA256}" \
    --output-root "${output_root}" \
    --seed "${seed}" \
    --experiment "${experiment}" \
    --epochs 200 \
    --early-stopping-patience 20 \
    --batch-size 1 \
    --evaluation-batch-size 1 \
    --workers 4 \
    --d-model 128 \
    --num-heads 8 \
    --num-self-layers 2 \
    --num-rbf 16 \
    --neighbor-k 32 \
    --neighbor-radius 16.0 \
    --sequence-window 32 \
    --dropout 0.1 \
    --learning-rate 0.0003 \
    --weight-decay 0.0001 \
    --contact-loss-weight 1.0 \
    --paratope-loss-weight 0.25 \
    --epitope-loss-weight 0.25 \
    --specificity-loss-weight 0.25 \
    --specificity-margin 0.05 \
    --score-fraction 0.01 \
    --calibration-max-pairs 2000000 \
    --offload-saved-tensors \
    --device cuda \
    --precision bf16
done
done

echo "All requested experiment-by-seed development paths are complete. No test or external scorer was run."

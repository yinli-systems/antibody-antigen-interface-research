#!/bin/bash
#SBATCH --job-name=oas-prod-map-v1
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --array=0-31%16
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-prod-map-v1-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-prod-map-v1-%A_%a.err

set -euo pipefail
umask 077

: "${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}"
: "${OAS_DOWNLOAD_ROOT:?OAS_DOWNLOAD_ROOT is required}"
: "${OAS_PLAN:?OAS_PLAN is required}"
: "${OAS_DOWNLOAD_AUDIT:?OAS_DOWNLOAD_AUDIT is required}"
: "${OAS_OUTPUT_ROOT:?OAS_OUTPUT_ROOT is required}"
: "${OAS_PROTOCOL_SHA256:?OAS_PROTOCOL_SHA256 is required}"
: "${OAS_CONTROLLER_SHA256:?OAS_CONTROLLER_SHA256 is required}"
: "${OAS_PREPROCESSING_SHA256:?OAS_PREPROCESSING_SHA256 is required}"
: "${OAS_RECORD_SHA256:?OAS_RECORD_SHA256 is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${OAS_SOURCE_ROOT}/src"

exec python3 "${OAS_SOURCE_ROOT}/scripts/map_oas_production_records.py" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --task-count 32 \
  --download-root "${OAS_DOWNLOAD_ROOT}" \
  --plan "${OAS_PLAN}" \
  --download-audit "${OAS_DOWNLOAD_AUDIT}" \
  --filter-policy "${OAS_SOURCE_ROOT}/protocols/oas_sequence_filter_v1.json" \
  --protocol "${OAS_SOURCE_ROOT}/protocols/oas_production_materialization_v1.json" \
  --protocol-file-sha256 "${OAS_PROTOCOL_SHA256}" \
  --output-root "${OAS_OUTPUT_ROOT}" \
  --controller-source-sha256 "${OAS_CONTROLLER_SHA256}" \
  --preprocessing-source-sha256 "${OAS_PREPROCESSING_SHA256}" \
  --record-source-sha256 "${OAS_RECORD_SHA256}"

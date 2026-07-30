#!/bin/bash
#SBATCH --job-name=oas-prod-reduce-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --array=0-255%16
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-prod-reduce-v1-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-prod-reduce-v1-%A_%a.err

set -euo pipefail
umask 077

: "${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}"
: "${OAS_MAP_ROOT:?OAS_MAP_ROOT is required}"
: "${OAS_OUTPUT_ROOT:?OAS_OUTPUT_ROOT is required}"
: "${OAS_PROTOCOL_SHA256:?OAS_PROTOCOL_SHA256 is required}"
: "${OAS_REDUCER_SHA256:?OAS_REDUCER_SHA256 is required}"
: "${OAS_RECORD_SHA256:?OAS_RECORD_SHA256 is required}"
: "${OAS_TOKENIZER_SHA256:?OAS_TOKENIZER_SHA256 is required}"
: "${OAS_CPU_PARTITION:?OAS_CPU_PARTITION records the external CPU partition}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${OAS_SOURCE_ROOT}/src"

exec python3 "${OAS_SOURCE_ROOT}/scripts/reduce_oas_production_bucket.py" \
  --bucket "${SLURM_ARRAY_TASK_ID}" \
  --map-root "${OAS_MAP_ROOT}" \
  --protocol "${OAS_SOURCE_ROOT}/protocols/oas_production_materialization_v1.json" \
  --protocol-file-sha256 "${OAS_PROTOCOL_SHA256}" \
  --output-root "${OAS_OUTPUT_ROOT}" \
  --controller-source-sha256 "${OAS_REDUCER_SHA256}" \
  --record-source-sha256 "${OAS_RECORD_SHA256}" \
  --tokenizer-source-sha256 "${OAS_TOKENIZER_SHA256}"

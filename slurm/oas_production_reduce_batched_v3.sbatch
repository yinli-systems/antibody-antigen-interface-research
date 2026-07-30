#!/bin/bash
#SBATCH --job-name=oas-prod-reduce-v3
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-prod-reduce-v3-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-prod-reduce-v3-%j.err

set -euo pipefail
umask 077

: "${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}"
: "${OAS_MAP_ROOT:?OAS_MAP_ROOT is required}"
: "${OAS_OUTPUT_ROOT:?OAS_OUTPUT_ROOT is required}"
: "${OAS_BATCH_LOG_ROOT:?OAS_BATCH_LOG_ROOT is required}"
: "${OAS_PROTOCOL_SHA256:?OAS_PROTOCOL_SHA256 is required}"
: "${OAS_SUPERVISOR_SHA256:?OAS_SUPERVISOR_SHA256 is required}"
: "${OAS_REDUCER_SHA256:?OAS_REDUCER_SHA256 is required}"
: "${OAS_RECORD_SHA256:?OAS_RECORD_SHA256 is required}"
: "${OAS_TOKENIZER_SHA256:?OAS_TOKENIZER_SHA256 is required}"
: "${SLURM_JOB_GPUS:?One scheduler-reserved GPU is required}"

test "$(sha256sum "${OAS_SOURCE_ROOT}/scripts/run_oas_production_task_batch.py" | awk '{print $1}')" = "${OAS_SUPERVISOR_SHA256}"
test "$(sha256sum "${OAS_SOURCE_ROOT}/scripts/reduce_oas_production_bucket.py" | awk '{print $1}')" = "${OAS_REDUCER_SHA256}"

export CUDA_VISIBLE_DEVICES=
export OAS_SCHEDULER_RESERVED_GPU_COUNT=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${OAS_SOURCE_ROOT}/src"

exec python3 "${OAS_SOURCE_ROOT}/scripts/run_oas_production_task_batch.py" \
  --stage reduce \
  --task-count 256 \
  --worker-count 8 \
  --completion-template "${OAS_OUTPUT_ROOT}/reduce/bucket-{task_id:03d}/receipt.json" \
  --log-root "${OAS_BATCH_LOG_ROOT}" \
  -- \
  python3 "${OAS_SOURCE_ROOT}/scripts/reduce_oas_production_bucket.py" \
  --bucket "{task_id}" \
  --map-root "${OAS_MAP_ROOT}" \
  --protocol "${OAS_SOURCE_ROOT}/protocols/oas_production_materialization_v3.json" \
  --protocol-file-sha256 "${OAS_PROTOCOL_SHA256}" \
  --output-root "${OAS_OUTPUT_ROOT}" \
  --controller-source-sha256 "${OAS_REDUCER_SHA256}" \
  --record-source-sha256 "${OAS_RECORD_SHA256}" \
  --tokenizer-source-sha256 "${OAS_TOKENIZER_SHA256}"

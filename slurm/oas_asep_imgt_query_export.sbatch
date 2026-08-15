#!/bin/bash
#SBATCH --job-name=oas-imgt-query
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-imgt-query-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-imgt-query-%j.err

set -euo pipefail
umask 077

: "${OAS_QUERY_SOURCE_ROOT:?OAS_QUERY_SOURCE_ROOT is required}"
: "${OAS_QUERY_SOURCE_COMMIT:?OAS_QUERY_SOURCE_COMMIT is required}"
: "${OAS_QUERY_SCRIPT_SHA256:?OAS_QUERY_SCRIPT_SHA256 is required}"
: "${OAS_QUERY_MODULE_SHA256:?OAS_QUERY_MODULE_SHA256 is required}"
: "${OAS_ASEP_CACHE_ROOT:?OAS_ASEP_CACHE_ROOT is required}"
: "${OAS_ASEP_CACHE_MANIFEST_SHA256:?OAS_ASEP_CACHE_MANIFEST_SHA256 is required}"
: "${OAS_ANARCII_RUNTIME_ROOT:?OAS_ANARCII_RUNTIME_ROOT is required}"
: "${OAS_ANARCII_FREEZE_SHA256:?OAS_ANARCII_FREEZE_SHA256 is required}"
: "${OAS_ANARCII_WHEEL:?OAS_ANARCII_WHEEL is required}"
: "${OAS_QUERY_OUTPUT_ROOT:?OAS_QUERY_OUTPUT_ROOT is required}"
: "${SLURM_JOB_GPUS:?One scheduler-reserved GPU is required}"

test "$(cat "${OAS_QUERY_SOURCE_ROOT}/DEPLOYED_COMMIT")" = "${OAS_QUERY_SOURCE_COMMIT}"
test "$(sha256sum "${OAS_QUERY_SOURCE_ROOT}/scripts/export_asep_imgt_queries.py" | awk '{print $1}')" = "${OAS_QUERY_SCRIPT_SHA256}"
test "$(sha256sum "${OAS_QUERY_SOURCE_ROOT}/src/antibody_pretraining/imgt_query_export.py" | awk '{print $1}')" = "${OAS_QUERY_MODULE_SHA256}"
test "$(sha256sum "${OAS_ASEP_CACHE_ROOT}/ASEP_DEVELOPMENT_CACHE_MANIFEST.json" | awk '{print $1}')" = "${OAS_ASEP_CACHE_MANIFEST_SHA256}"
test "$(sha256sum "${OAS_ANARCII_RUNTIME_ROOT}/PIP_FREEZE.txt" | awk '{print $1}')" = "${OAS_ANARCII_FREEZE_SHA256}"
test ! -e "${OAS_QUERY_OUTPUT_ROOT}"

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${OAS_QUERY_SOURCE_ROOT}/src:${OAS_QUERY_SOURCE_ROOT}/framecontact/src"

exec "${OAS_ANARCII_RUNTIME_ROOT}/bin/python" \
  "${OAS_QUERY_SOURCE_ROOT}/scripts/export_asep_imgt_queries.py" \
  --asep-cache-root "${OAS_ASEP_CACHE_ROOT}" \
  --asep-cache-manifest-file-sha256 "${OAS_ASEP_CACHE_MANIFEST_SHA256}" \
  --anarcii-wheel "${OAS_ANARCII_WHEEL}" \
  --output-root "${OAS_QUERY_OUTPUT_ROOT}" \
  --expected-query-count 3106 \
  --batch-size 256 \
  --ncpu 8

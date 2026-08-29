#!/bin/bash
#SBATCH --job-name=oas-mmseqs-gap-finalize
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-mmseqs-gap-finalize-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-mmseqs-gap-finalize-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
SOURCE_COMMIT=${ABF_SOURCE_COMMIT:?ABF_SOURCE_COMMIT is required}
TARGET_OUTPUT_ROOT=${ABF_TARGET_OUTPUT_ROOT:?ABF_TARGET_OUTPUT_ROOT is required}
QUERY_ROOT=${ABF_QUERY_ROOT:?ABF_QUERY_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
CONTROLLER_SHA256=${ABF_MMSEQS_CONTROLLER_SHA256:?required}
CORE_SHA256=${ABF_MMSEQS_CORE_SHA256:?required}
MMSEQS_SHA256=${ABF_MMSEQS_BINARY_SHA256:?required}
PROTOCOL_SHA256=${ABF_NEAR_IDENTITY_PROTOCOL_SHA256:?required}

[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]]
for digest in \
  "${CONTROLLER_SHA256}" "${CORE_SHA256}" "${MMSEQS_SHA256}" "${PROTOCOL_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || { echo "Source digest is invalid" >&2; exit 2; }
done
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src
export OAS_SCHEDULER_RESERVED_GPU_COUNT=1
export CUDA_VISIBLE_DEVICES=NoDevFiles

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/finalize_oas_mmseqs_gap_search.py" \
  --output-root "${OUTPUT_ROOT}" \
  --target-output-root "${TARGET_OUTPUT_ROOT}" \
  --query-root "${QUERY_ROOT}" \
  --task-count 32 \
  --worker-count 4 \
  --expected-controller-sha256 "${CONTROLLER_SHA256}" \
  --expected-core-sha256 "${CORE_SHA256}" \
  --expected-mmseqs-sha256 "${MMSEQS_SHA256}" \
  --expected-protocol-sha256 "${PROTOCOL_SHA256}"

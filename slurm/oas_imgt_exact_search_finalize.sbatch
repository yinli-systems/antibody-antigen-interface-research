#!/bin/bash
#SBATCH --job-name=oas-imgt-search-finalize
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-imgt-search-finalize-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-imgt-search-finalize-%j.err

set -euo pipefail
umask 077

: "${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}"
: "${OAS_SOURCE_COMMIT:?OAS_SOURCE_COMMIT is required}"
: "${OAS_TARGET_OUTPUT_ROOT:?OAS_TARGET_OUTPUT_ROOT is required}"
: "${OAS_QUERY_ROOT:?OAS_QUERY_ROOT is required}"
: "${OAS_SEARCH_OUTPUT_ROOT:?OAS_SEARCH_OUTPUT_ROOT is required}"
: "${OAS_PYTHON_BIN:?OAS_PYTHON_BIN is required}"
: "${OAS_PYTHON_SHA256:?OAS_PYTHON_SHA256 is required}"
: "${OAS_SEARCH_FINALIZER_SHA256:?OAS_SEARCH_FINALIZER_SHA256 is required}"
: "${OAS_SEARCH_CONTROLLER_SHA256:?OAS_SEARCH_CONTROLLER_SHA256 is required}"
: "${OAS_SEARCH_KERNEL_SHA256:?OAS_SEARCH_KERNEL_SHA256 is required}"
: "${OAS_SEARCH_SCHEMA_SHA256:?OAS_SEARCH_SCHEMA_SHA256 is required}"
: "${SLURM_JOB_GPUS:?One scheduler-reserved idle GPU is required}"

for root in \
  "${OAS_SOURCE_ROOT}" \
  "${OAS_TARGET_OUTPUT_ROOT}" \
  "${OAS_QUERY_ROOT}" \
  "${OAS_SEARCH_OUTPUT_ROOT}"; do
  [[ "${root}" == /* ]] || { echo "All finalizer paths must be absolute" >&2; exit 2; }
done
[[ "${OAS_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
for digest in \
  "${OAS_PYTHON_SHA256}" \
  "${OAS_SEARCH_FINALIZER_SHA256}" \
  "${OAS_SEARCH_CONTROLLER_SHA256}" \
  "${OAS_SEARCH_KERNEL_SHA256}" \
  "${OAS_SEARCH_SCHEMA_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ "${OAS_PYTHON_BIN}" == /* && -x "${OAS_PYTHON_BIN}" ]]
PYTHON_REAL=$(readlink -f "${OAS_PYTHON_BIN}")
[[ -f "${PYTHON_REAL}" && ! -L "${PYTHON_REAL}" ]]
[[ "$(sha256sum "${PYTHON_REAL}" | awk '{print $1}')" == "${OAS_PYTHON_SHA256}" ]]

SOURCE_COMMIT_ANCHOR=${OAS_SOURCE_ROOT}/SOURCE_COMMIT
FINALIZER=${OAS_SOURCE_ROOT}/scripts/finalize_oas_imgt_exact_search.py
CONTROLLER=${OAS_SOURCE_ROOT}/scripts/run_oas_imgt_exact_search_task.py
KERNEL_SOURCE=${OAS_SOURCE_ROOT}/native/oas_imgt_exact_search.cu
MATCH_SCHEMA=${OAS_SOURCE_ROOT}/src/antibody_pretraining/imgt_search_output.py
for source in \
  "${SOURCE_COMMIT_ANCHOR}" \
  "${FINALIZER}" \
  "${CONTROLLER}" \
  "${KERNEL_SOURCE}" \
  "${MATCH_SCHEMA}"; do
  [[ -f "${source}" && ! -L "${source}" ]] \
    || { echo "Finalizer source is absent or symbolic: ${source}" >&2; exit 2; }
done
[[ "$(<"${SOURCE_COMMIT_ANCHOR}")" == "${OAS_SOURCE_COMMIT}" ]]
[[ "$(sha256sum "${FINALIZER}" | awk '{print $1}')" == "${OAS_SEARCH_FINALIZER_SHA256}" ]]
[[ "$(sha256sum "${CONTROLLER}" | awk '{print $1}')" == "${OAS_SEARCH_CONTROLLER_SHA256}" ]]
[[ "$(sha256sum "${KERNEL_SOURCE}" | awk '{print $1}')" == "${OAS_SEARCH_KERNEL_SHA256}" ]]
[[ "$(sha256sum "${MATCH_SCHEMA}" | awk '{print $1}')" == "${OAS_SEARCH_SCHEMA_SHA256}" ]]

IFS=',' read -r -a RESERVED_GPU_IDS <<< "${SLURM_JOB_GPUS}"
[[ "${#RESERVED_GPU_IDS[@]}" == 1 ]]
export CUDA_VISIBLE_DEVICES=
export OAS_SCHEDULER_RESERVED_GPU_COUNT=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${OAS_SOURCE_ROOT}/src

"${OAS_PYTHON_BIN}" "${FINALIZER}" \
  --search-output-root "${OAS_SEARCH_OUTPUT_ROOT}" \
  --target-output-root "${OAS_TARGET_OUTPUT_ROOT}" \
  --query-root "${OAS_QUERY_ROOT}" \
  --task-count 32 \
  --worker-count 8 \
  --finalizer-source-sha256 "${OAS_SEARCH_FINALIZER_SHA256}" \
  --search-controller-source-sha256 "${OAS_SEARCH_CONTROLLER_SHA256}" \
  --kernel-source-sha256 "${OAS_SEARCH_KERNEL_SHA256}" \
  --match-schema-source-sha256 "${OAS_SEARCH_SCHEMA_SHA256}"

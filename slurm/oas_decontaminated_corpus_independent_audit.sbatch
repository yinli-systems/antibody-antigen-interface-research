#!/bin/bash
#SBATCH --job-name=oas-decontam-audit
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-decontam-audit-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-decontam-audit-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${OAS_PYTHON_BIN:?OAS_PYTHON_BIN is required}
PYTHON_SHA256=${OAS_PYTHON_SHA256:?OAS_PYTHON_SHA256 is required}
SOURCE_ROOT=${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}
SOURCE_COMMIT=${OAS_SOURCE_COMMIT:?OAS_SOURCE_COMMIT is required}
CORPUS_ROOT=${OAS_CORPUS_ROOT:?OAS_CORPUS_ROOT is required}
EXCLUSION_ROOT=${OAS_EXCLUSION_ROOT:?OAS_EXCLUSION_ROOT is required}
OUTPUT_ROOT=${OAS_DECONTAMINATED_CORPUS_ROOT:?required}
PROTOCOL=${OAS_DECONTAMINATION_PROTOCOL:?required}
PROTOCOL_SHA256=${OAS_DECONTAMINATION_PROTOCOL_SHA256:?required}
AUDITOR_SHA256=${OAS_DECONTAMINATION_AUDITOR_SHA256:?required}

for path in \
  "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${EXCLUSION_ROOT}" "${OUTPUT_ROOT}" "${PROTOCOL}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
for digest in "${PYTHON_SHA256}" "${PROTOCOL_SHA256}" "${AUDITOR_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ -x "${PYTHON_BIN}" && -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]]
[[ -d "${CORPUS_ROOT}" && ! -L "${CORPUS_ROOT}" ]]
[[ -d "${EXCLUSION_ROOT}" && ! -L "${EXCLUSION_ROOT}" ]]
[[ -d "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
[[ -f "${PROTOCOL}" && ! -L "${PROTOCOL}" ]]
[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]]

AUDITOR=${SOURCE_ROOT}/scripts/audit_oas_decontaminated_corpus_independent.py
[[ "$(sha256sum "${PYTHON_BIN}" | awk '{print $1}')" == "${PYTHON_SHA256}" ]]
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]
[[ "$(sha256sum "${AUDITOR}" | awk '{print $1}')" == "${AUDITOR_SHA256}" ]]

IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Independent audit requires one scheduler-reserved GPU" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=
export OAS_SCHEDULER_RESERVED_GPU_COUNT=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src

"${PYTHON_BIN}" "${AUDITOR}" \
  --source-root "${CORPUS_ROOT}" \
  --exclusion-root "${EXCLUSION_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-file-sha256 "${PROTOCOL_SHA256}" \
  --auditor-source-sha256 "${AUDITOR_SHA256}" \
  --worker-count 8 \
  --chunk-records 131072

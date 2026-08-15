#!/bin/bash
#SBATCH --job-name=oas-decontam-finalize
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-decontam-finalize-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-decontam-finalize-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${OAS_PYTHON_BIN:?OAS_PYTHON_BIN is required}
PYTHON_SHA256=${OAS_PYTHON_SHA256:?OAS_PYTHON_SHA256 is required}
SOURCE_ROOT=${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}
SOURCE_COMMIT=${OAS_SOURCE_COMMIT:?OAS_SOURCE_COMMIT is required}
OUTPUT_ROOT=${OAS_DECONTAMINATED_CORPUS_ROOT:?required}
PROTOCOL=${OAS_DECONTAMINATION_PROTOCOL:?required}
PROTOCOL_SHA256=${OAS_DECONTAMINATION_PROTOCOL_SHA256:?required}
FINALIZER_SHA256=${OAS_DECONTAMINATION_FINALIZER_SHA256:?required}
CORE_SHA256=${OAS_FILTER_CORE_SHA256:?OAS_FILTER_CORE_SHA256 is required}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${OUTPUT_ROOT}" "${PROTOCOL}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
for digest in \
  "${PYTHON_SHA256}" "${PROTOCOL_SHA256}" \
  "${FINALIZER_SHA256}" "${CORE_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ -x "${PYTHON_BIN}" && -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]]
[[ -d "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
[[ -f "${PROTOCOL}" && ! -L "${PROTOCOL}" ]]
[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]]

FINALIZER=${SOURCE_ROOT}/scripts/finalize_oas_decontaminated_corpus.py
CORE=${SOURCE_ROOT}/src/antibody_pretraining/decontaminated_corpus.py
[[ "$(sha256sum "${PYTHON_BIN}" | awk '{print $1}')" == "${PYTHON_SHA256}" ]]
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]
[[ "$(sha256sum "${FINALIZER}" | awk '{print $1}')" == "${FINALIZER_SHA256}" ]]
[[ "$(sha256sum "${CORE}" | awk '{print $1}')" == "${CORE_SHA256}" ]]

IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Finalization requires one scheduler-reserved GPU" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=
export OAS_SCHEDULER_RESERVED_GPU_COUNT=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src

"${PYTHON_BIN}" "${FINALIZER}" \
  --output-root "${OUTPUT_ROOT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-file-sha256 "${PROTOCOL_SHA256}" \
  --finalizer-source-sha256 "${FINALIZER_SHA256}"

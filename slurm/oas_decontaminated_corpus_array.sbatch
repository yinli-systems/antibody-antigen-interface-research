#!/bin/bash
#SBATCH --job-name=oas-decontam
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-31%32
#SBATCH --time=1-00:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-decontam-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-decontam-%A_%a.err

set -euo pipefail
umask 077

PYTHON_BIN=${OAS_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}
SOURCE_COMMIT=${OAS_SOURCE_COMMIT:?OAS_SOURCE_COMMIT is required}
CORPUS_ROOT=${OAS_CORPUS_ROOT:?OAS_CORPUS_ROOT is required}
EXCLUSION_ROOT=${OAS_EXCLUSION_ROOT:?OAS_EXCLUSION_ROOT is required}
OUTPUT_ROOT=${OAS_DECONTAMINATED_CORPUS_ROOT:?required}
PROTOCOL=${OAS_DECONTAMINATION_PROTOCOL:?required}
PROTOCOL_SHA256=${OAS_DECONTAMINATION_PROTOCOL_SHA256:?required}
PYTHON_SHA256=${OAS_PYTHON_SHA256:?required}
CONTROLLER_SHA256=${OAS_FILTER_CONTROLLER_SHA256:?required}
CORE_SHA256=${OAS_FILTER_CORE_SHA256:?required}

for path in \
  "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${EXCLUSION_ROOT}" "${OUTPUT_ROOT}" "${PROTOCOL}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
for digest in \
  "${PROTOCOL_SHA256}" "${PYTHON_SHA256}" \
  "${CONTROLLER_SHA256}" "${CORE_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ -x "${PYTHON_BIN}" ]]
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]]
[[ -d "${CORPUS_ROOT}" && ! -L "${CORPUS_ROOT}" ]]
[[ -d "${EXCLUSION_ROOT}" && ! -L "${EXCLUSION_ROOT}" ]]
[[ -d "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
[[ -f "${PROTOCOL}" && ! -L "${PROTOCOL}" ]]
[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]]

CONTROLLER=${SOURCE_ROOT}/scripts/filter_oas_decontaminated_corpus_task.py
CORE=${SOURCE_ROOT}/src/antibody_pretraining/decontaminated_corpus.py
[[ "$(sha256sum "${PYTHON_BIN}" | awk '{print $1}')" == "${PYTHON_SHA256}" ]]
[[ "$(sha256sum "${CONTROLLER}" | awk '{print $1}')" == "${CONTROLLER_SHA256}" ]]
[[ "$(sha256sum "${CORE}" | awk '{print $1}')" == "${CORE_SHA256}" ]]
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]

IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Filtering requires one scheduler-reserved GPU" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=
export OAS_SCHEDULER_RESERVED_GPU_COUNT=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src
export TMPDIR=${OUTPUT_ROOT}/runtime-tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -m 700 -p "${TMPDIR}"

"${PYTHON_BIN}" "${CONTROLLER}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --task-count 32 \
  --bucket-count 256 \
  --source-root "${CORPUS_ROOT}" \
  --exclusion-root "${EXCLUSION_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-file-sha256 "${PROTOCOL_SHA256}" \
  --controller-source-sha256 "${CONTROLLER_SHA256}"

rmdir "${TMPDIR}"

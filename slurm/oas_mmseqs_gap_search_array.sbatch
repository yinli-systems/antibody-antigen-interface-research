#!/bin/bash
#SBATCH --job-name=oas-mmseqs-gap
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-31%32
#SBATCH --time=1-00:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-mmseqs-gap-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-mmseqs-gap-%A_%a.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
MMSEQS_BIN=${ABF_MMSEQS_BIN:?ABF_MMSEQS_BIN is required}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
SOURCE_COMMIT=${ABF_SOURCE_COMMIT:?ABF_SOURCE_COMMIT is required}
TARGET_OUTPUT_ROOT=${ABF_TARGET_OUTPUT_ROOT:?ABF_TARGET_OUTPUT_ROOT is required}
QUERY_ROOT=${ABF_QUERY_ROOT:?ABF_QUERY_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
PROTOCOL_SHA256=${ABF_NEAR_IDENTITY_PROTOCOL_SHA256:?required}
MAXIMUM_SEQUENCES=${ABF_MMSEQS_MAXIMUM_SEQUENCES:-500000}

for artifact_path in \
  "${PYTHON_BIN}" "${MMSEQS_BIN}" "${SOURCE_ROOT}" \
  "${TARGET_OUTPUT_ROOT}" "${QUERY_ROOT}" "${OUTPUT_ROOT}"; do
  [[ "${artifact_path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${PROTOCOL_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${MAXIMUM_SEQUENCES}" =~ ^[0-9]+$ ]]
[[ -x "${PYTHON_BIN}" && -x "${MMSEQS_BIN}" ]]
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]]
[[ -d "${TARGET_OUTPUT_ROOT}" && ! -L "${TARGET_OUTPUT_ROOT}" ]]
[[ -d "${QUERY_ROOT}" && ! -L "${QUERY_ROOT}" ]]
[[ -d "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]]
PROTOCOL=${SOURCE_ROOT}/protocols/oas_asep_near_identity_decontamination_v1.json
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]

IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Gap-aware task requires exactly one reserved GPU" >&2; exit 2; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src
export TMPDIR=${OUTPUT_ROOT}/runtime-tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export XDG_CACHE_HOME=${OUTPUT_ROOT}/runtime-cache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_oas_mmseqs_gap_search_task.py" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --task-count 32 \
  --target-output-root "${TARGET_OUTPUT_ROOT}" \
  --query-root "${QUERY_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --mmseqs-bin "${MMSEQS_BIN}" \
  --maximum-sequences "${MAXIMUM_SEQUENCES}" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${PROTOCOL_SHA256}"

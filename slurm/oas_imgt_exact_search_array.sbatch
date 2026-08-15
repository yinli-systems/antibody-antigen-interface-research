#!/bin/bash
#SBATCH --job-name=oas-imgt-exact
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-31%32
#SBATCH --time=12:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-imgt-exact-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-imgt-exact-%A_%a.err

set -euo pipefail
umask 077

: "${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}"
: "${OAS_SOURCE_COMMIT:?OAS_SOURCE_COMMIT is required}"
: "${OAS_TARGET_OUTPUT_ROOT:?OAS_TARGET_OUTPUT_ROOT is required}"
: "${OAS_QUERY_ROOT:?OAS_QUERY_ROOT is required}"
: "${OAS_SEARCH_OUTPUT_ROOT:?OAS_SEARCH_OUTPUT_ROOT is required}"
: "${OAS_SEARCH_CONTROLLER_SHA256:?OAS_SEARCH_CONTROLLER_SHA256 is required}"
: "${OAS_SEARCH_KERNEL_SHA256:?OAS_SEARCH_KERNEL_SHA256 is required}"
: "${OAS_SEARCH_SCHEMA_SHA256:?OAS_SEARCH_SCHEMA_SHA256 is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${SLURM_JOB_GPUS:?Exactly one scheduler GPU is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

for root in \
  "${OAS_SOURCE_ROOT}" \
  "${OAS_TARGET_OUTPUT_ROOT}" \
  "${OAS_QUERY_ROOT}" \
  "${OAS_SEARCH_OUTPUT_ROOT}"; do
  [[ "${root}" == /* ]] || { echo "All search paths must be absolute" >&2; exit 2; }
done
[[ "${OAS_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "Source commit is invalid" >&2; exit 2; }
for digest in \
  "${OAS_SEARCH_CONTROLLER_SHA256}" \
  "${OAS_SEARCH_KERNEL_SHA256}" \
  "${OAS_SEARCH_SCHEMA_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "Source digest is invalid" >&2; exit 2; }
done
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|[12][0-9]|3[01])$ ]] \
  || { echo "Array task is outside 0..31" >&2; exit 2; }

SOURCE_COMMIT_ANCHOR=${OAS_SOURCE_ROOT}/SOURCE_COMMIT
CONTROLLER=${OAS_SOURCE_ROOT}/scripts/run_oas_imgt_exact_search_task.py
KERNEL_SOURCE=${OAS_SOURCE_ROOT}/native/oas_imgt_exact_search.cu
MATCH_SCHEMA=${OAS_SOURCE_ROOT}/src/antibody_pretraining/imgt_search_output.py
for source in "${SOURCE_COMMIT_ANCHOR}" "${CONTROLLER}" "${KERNEL_SOURCE}" "${MATCH_SCHEMA}"; do
  [[ -f "${source}" && ! -L "${source}" ]] \
    || { echo "Search source is absent or symbolic: ${source}" >&2; exit 2; }
done
[[ "$(<"${SOURCE_COMMIT_ANCHOR}")" == "${OAS_SOURCE_COMMIT}" ]]
[[ "$(sha256sum "${CONTROLLER}" | awk '{print $1}')" == "${OAS_SEARCH_CONTROLLER_SHA256}" ]]
[[ "$(sha256sum "${KERNEL_SOURCE}" | awk '{print $1}')" == "${OAS_SEARCH_KERNEL_SHA256}" ]]
[[ "$(sha256sum "${MATCH_SCHEMA}" | awk '{print $1}')" == "${OAS_SEARCH_SCHEMA_SHA256}" ]]

IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a RESERVED_GPU_IDS <<< "${SLURM_JOB_GPUS}"
[[ "${#VISIBLE_GPU_IDS[@]}" == 1 && "${#RESERVED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Exact search requires exactly one visible scheduler GPU" >&2; exit 2; }

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8
NVCC_BIN=$(command -v nvcc)
[[ -x "${NVCC_BIN}" && ! -L "${NVCC_BIN}" ]] \
  || { echo "CUDA 12.8 nvcc is absent or symbolic" >&2; exit 2; }
NVCC_VERSION=$(nvcc --version)
[[ "${NVCC_VERSION}" == *"release 12.8"* ]] \
  || { echo "CUDA compiler release differs from 12.8" >&2; exit 2; }

SCRATCH_ROOT=${SLURM_TMPDIR:-/tmp}
KERNEL_BINARY=${SCRATCH_ROOT}/oas-imgt-exact-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
cleanup() {
  rm -f -- "${KERNEL_BINARY}"
}
trap cleanup EXIT

nvcc \
  -std=c++17 \
  -O3 \
  -lineinfo \
  -gencode arch=compute_120,code=sm_120 \
  "${KERNEL_SOURCE}" \
  -o "${KERNEL_BINARY}"
[[ -x "${KERNEL_BINARY}" && ! -L "${KERNEL_BINARY}" ]]
KERNEL_BINARY_SHA256=$(sha256sum "${KERNEL_BINARY}" | awk '{print $1}')

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${OAS_SOURCE_ROOT}/src

python3 "${CONTROLLER}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --target-output-root "${OAS_TARGET_OUTPUT_ROOT}" \
  --query-root "${OAS_QUERY_ROOT}" \
  --search-output-root "${OAS_SEARCH_OUTPUT_ROOT}" \
  --kernel-binary "${KERNEL_BINARY}" \
  --kernel-source "${KERNEL_SOURCE}" \
  --batch-records 32768 \
  --controller-source-sha256 "${OAS_SEARCH_CONTROLLER_SHA256}" \
  --kernel-binary-sha256 "${KERNEL_BINARY_SHA256}" \
  --kernel-source-sha256 "${OAS_SEARCH_KERNEL_SHA256}" \
  --match-schema-source-sha256 "${OAS_SEARCH_SCHEMA_SHA256}"

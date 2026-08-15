#!/bin/bash
#SBATCH --job-name=oas-imgt-canary
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-imgt-canary-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-imgt-canary-%j.err

set -euo pipefail
umask 077

: "${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}"
: "${OAS_SOURCE_COMMIT:?OAS_SOURCE_COMMIT is required}"
: "${OAS_CANARY_OUTPUT_ROOT:?OAS_CANARY_OUTPUT_ROOT is required}"
: "${OAS_CANARY_SCRIPT_SHA256:?OAS_CANARY_SCRIPT_SHA256 is required}"
: "${OAS_SEARCH_KERNEL_SHA256:?OAS_SEARCH_KERNEL_SHA256 is required}"
: "${SLURM_JOB_GPUS:?Exactly one scheduler GPU is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

[[ "${OAS_SOURCE_ROOT}" == /* && "${OAS_CANARY_OUTPUT_ROOT}" == /* ]] \
  || { echo "Canary paths must be absolute" >&2; exit 2; }
[[ "${OAS_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "Source commit is invalid" >&2; exit 2; }
[[ "${OAS_CANARY_SCRIPT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${OAS_SEARCH_KERNEL_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ ! -e "${OAS_CANARY_OUTPUT_ROOT}" && ! -L "${OAS_CANARY_OUTPUT_ROOT}" ]] \
  || { echo "Refusing to overwrite canary output" >&2; exit 2; }

SOURCE_COMMIT_ANCHOR=${OAS_SOURCE_ROOT}/SOURCE_COMMIT
CANARY_SCRIPT=${OAS_SOURCE_ROOT}/scripts/run_oas_imgt_exact_cuda_canary.py
KERNEL_SOURCE=${OAS_SOURCE_ROOT}/native/oas_imgt_exact_search.cu
for source in "${SOURCE_COMMIT_ANCHOR}" "${CANARY_SCRIPT}" "${KERNEL_SOURCE}"; do
  [[ -f "${source}" && ! -L "${source}" ]] \
    || { echo "Canary source is absent or symbolic: ${source}" >&2; exit 2; }
done
[[ "$(<"${SOURCE_COMMIT_ANCHOR}")" == "${OAS_SOURCE_COMMIT}" ]]
[[ "$(sha256sum "${CANARY_SCRIPT}" | awk '{print $1}')" == "${OAS_CANARY_SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${KERNEL_SOURCE}" | awk '{print $1}')" == "${OAS_SEARCH_KERNEL_SHA256}" ]]

IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a RESERVED_GPU_IDS <<< "${SLURM_JOB_GPUS}"
[[ "${#VISIBLE_GPU_IDS[@]}" == 1 && "${#RESERVED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Canary requires exactly one visible scheduler GPU" >&2; exit 2; }

source /etc/profile >/dev/null 2>&1
module purge
module load cuda/12.8
NVCC_BIN=$(command -v nvcc)
[[ -x "${NVCC_BIN}" && ! -L "${NVCC_BIN}" ]]
NVCC_VERSION=$(nvcc --version)
[[ "${NVCC_VERSION}" == *"release 12.8"* ]]

SCRATCH_ROOT=${SLURM_TMPDIR:-/tmp}
KERNEL_BINARY=${SCRATCH_ROOT}/oas-imgt-exact-canary-${SLURM_JOB_ID}
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

mkdir -p "$(dirname "${OAS_CANARY_OUTPUT_ROOT}")"
mkdir "${OAS_CANARY_OUTPUT_ROOT}"
FIXTURE_ROOT=${OAS_CANARY_OUTPUT_ROOT}/fixtures
ACTUAL_MATCHES=${OAS_CANARY_OUTPUT_ROOT}/actual_matches.bin
KERNEL_STATS=${OAS_CANARY_OUTPUT_ROOT}/kernel_stats.json

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${OAS_SOURCE_ROOT}/src
python3 "${CANARY_SCRIPT}" prepare --output-root "${FIXTURE_ROOT}" \
  > "${OAS_CANARY_OUTPUT_ROOT}/prepare.json"
"${KERNEL_BINARY}" \
  --queries "${FIXTURE_ROOT}/queries.bin" \
  --targets "${FIXTURE_ROOT}/targets.bin" \
  --matches "${ACTUAL_MATCHES}" \
  --stats "${KERNEL_STATS}" \
  --batch-records 7 \
  --device 0
python3 "${CANARY_SCRIPT}" audit \
  --fixture-root "${FIXTURE_ROOT}" \
  --actual-matches "${ACTUAL_MATCHES}" \
  --kernel-stats "${KERNEL_STATS}" \
  > "${OAS_CANARY_OUTPUT_ROOT}/AUDIT.json"

{
  echo "source_commit=${OAS_SOURCE_COMMIT}"
  echo "kernel_source_sha256=${OAS_SEARCH_KERNEL_SHA256}"
  echo "canary_script_sha256=${OAS_CANARY_SCRIPT_SHA256}"
  echo "kernel_binary_sha256=$(sha256sum "${KERNEL_BINARY}" | awk '{print $1}')"
  echo "nvcc_binary_sha256=$(sha256sum "${NVCC_BIN}" | awk '{print $1}')"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  printf '%s\n' "${NVCC_VERSION}"
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total \
    --format=csv,noheader
} > "${OAS_CANARY_OUTPUT_ROOT}/ENVIRONMENT.txt"
chmod -R a-w "${OAS_CANARY_OUTPUT_ROOT}"

#!/bin/bash
#SBATCH --job-name=oas-imgt-kernel-build
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-imgt-kernel-build-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-imgt-kernel-build-%j.err

set -euo pipefail
umask 077

: "${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}"
: "${OAS_SOURCE_COMMIT:?OAS_SOURCE_COMMIT is required}"
: "${OAS_KERNEL_BUNDLE_ROOT:?OAS_KERNEL_BUNDLE_ROOT is required}"
: "${OAS_PYTHON_BIN:?OAS_PYTHON_BIN is required}"
: "${OAS_PYTHON_SHA256:?OAS_PYTHON_SHA256 is required}"
: "${OAS_SEARCH_KERNEL_SHA256:?OAS_SEARCH_KERNEL_SHA256 is required}"
: "${OAS_CANARY_SCRIPT_SHA256:?OAS_CANARY_SCRIPT_SHA256 is required}"
: "${OAS_KERNEL_BUILD_LAUNCHER_SHA256:?OAS_KERNEL_BUILD_LAUNCHER_SHA256 is required}"
: "${OAS_KERNEL_BUNDLE_SEALER_SHA256:?OAS_KERNEL_BUNDLE_SEALER_SHA256 is required}"
: "${SLURM_JOB_GPUS:?Exactly one scheduler GPU is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

[[ "${OAS_SOURCE_ROOT}" == /* && "${OAS_KERNEL_BUNDLE_ROOT}" == /* ]]
[[ "${OAS_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
for digest in \
  "${OAS_PYTHON_SHA256}" \
  "${OAS_SEARCH_KERNEL_SHA256}" \
  "${OAS_CANARY_SCRIPT_SHA256}" \
  "${OAS_KERNEL_BUILD_LAUNCHER_SHA256}" \
  "${OAS_KERNEL_BUNDLE_SEALER_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ ! -e "${OAS_KERNEL_BUNDLE_ROOT}" && ! -L "${OAS_KERNEL_BUNDLE_ROOT}" ]] \
  || { echo "Refusing to overwrite a kernel bundle" >&2; exit 2; }
[[ "${OAS_PYTHON_BIN}" == /* && -x "${OAS_PYTHON_BIN}" ]]
PYTHON_REAL=$(readlink -f "${OAS_PYTHON_BIN}")
[[ -f "${PYTHON_REAL}" && ! -L "${PYTHON_REAL}" ]]
[[ "$(sha256sum "${PYTHON_REAL}" | awk '{print $1}')" == "${OAS_PYTHON_SHA256}" ]]

SOURCE_COMMIT_ANCHOR=${OAS_SOURCE_ROOT}/SOURCE_COMMIT
BUILD_LAUNCHER=${OAS_SOURCE_ROOT}/slurm/oas_imgt_exact_kernel_build.sbatch
SEALER=${OAS_SOURCE_ROOT}/scripts/seal_oas_imgt_exact_kernel_bundle.py
CANARY=${OAS_SOURCE_ROOT}/scripts/run_oas_imgt_exact_cuda_canary.py
KERNEL_SOURCE=${OAS_SOURCE_ROOT}/native/oas_imgt_exact_search.cu
for source in \
  "${SOURCE_COMMIT_ANCHOR}" \
  "${BUILD_LAUNCHER}" \
  "${SEALER}" \
  "${CANARY}" \
  "${KERNEL_SOURCE}"; do
  [[ -f "${source}" && ! -L "${source}" ]]
done
[[ "$(<"${SOURCE_COMMIT_ANCHOR}")" == "${OAS_SOURCE_COMMIT}" ]]
[[ "$(sha256sum "${BUILD_LAUNCHER}" | awk '{print $1}')" == "${OAS_KERNEL_BUILD_LAUNCHER_SHA256}" ]]
[[ "$(sha256sum "${SEALER}" | awk '{print $1}')" == "${OAS_KERNEL_BUNDLE_SEALER_SHA256}" ]]
[[ "$(sha256sum "${CANARY}" | awk '{print $1}')" == "${OAS_CANARY_SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${KERNEL_SOURCE}" | awk '{print $1}')" == "${OAS_SEARCH_KERNEL_SHA256}" ]]

IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a RESERVED_GPU_IDS <<< "${SLURM_JOB_GPUS}"
[[ "${#VISIBLE_GPU_IDS[@]}" == 1 && "${#RESERVED_GPU_IDS[@]}" == 1 ]]

source /etc/profile.d/modules.sh
module purge
module load cuda/12.8
NVCC_BIN=$(command -v nvcc)
[[ -x "${NVCC_BIN}" && ! -L "${NVCC_BIN}" ]]
NVCC_VERSION=$(nvcc --version)
[[ "${NVCC_VERSION}" == *"release 12.8"* ]]

PARENT=$(dirname "${OAS_KERNEL_BUNDLE_ROOT}")
mkdir -p "${PARENT}"
TEMPORARY=$(mktemp -d "${PARENT}/.oas-imgt-kernel-build-${SLURM_JOB_ID}.partial-XXXXXX")
cleanup() {
  if [[ -d "${TEMPORARY}" ]]; then
    chmod -R u+w "${TEMPORARY}" 2>/dev/null || true
    rm -rf -- "${TEMPORARY}"
  fi
}
trap cleanup EXIT

KERNEL_BINARY=${TEMPORARY}/oas_imgt_exact_search
nvcc \
  -std=c++17 \
  -O3 \
  -lineinfo \
  -gencode arch=compute_120,code=sm_120 \
  "${KERNEL_SOURCE}" \
  -o "${KERNEL_BINARY}"
[[ -x "${KERNEL_BINARY}" && ! -L "${KERNEL_BINARY}" ]]

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${OAS_SOURCE_ROOT}/src
"${OAS_PYTHON_BIN}" "${CANARY}" prepare --output-root "${TEMPORARY}/fixtures" \
  > "${TEMPORARY}/prepare.json"
"${KERNEL_BINARY}" \
  --queries "${TEMPORARY}/fixtures/queries.bin" \
  --targets "${TEMPORARY}/fixtures/targets.bin" \
  --matches "${TEMPORARY}/actual_matches.bin" \
  --stats "${TEMPORARY}/kernel_stats.json" \
  --batch-records 7 \
  --device 0
printf '%s\n' "${NVCC_VERSION}" > "${TEMPORARY}/nvcc-version.txt"
"${OAS_PYTHON_BIN}" "${SEALER}" \
  --bundle-root "${TEMPORARY}" \
  --kernel-source "${KERNEL_SOURCE}" \
  --canary-source "${CANARY}" \
  --nvcc-binary "${NVCC_BIN}" \
  --nvcc-version-file "${TEMPORARY}/nvcc-version.txt" \
  --kernel-source-sha256 "${OAS_SEARCH_KERNEL_SHA256}" \
  --canary-source-sha256 "${OAS_CANARY_SCRIPT_SHA256}" \
  --build-launcher-source-sha256 "${OAS_KERNEL_BUILD_LAUNCHER_SHA256}" \
  --sealer-source-sha256 "${OAS_KERNEL_BUNDLE_SEALER_SHA256}" \
  > "${TEMPORARY}/SEALER_STDOUT.json"

mv "${TEMPORARY}" "${OAS_KERNEL_BUNDLE_ROOT}"
TEMPORARY=
trap - EXIT
WRITABLE_ENTRY=$(find "${OAS_KERNEL_BUNDLE_ROOT}" -perm /222 -print -quit)
if [[ -n "${WRITABLE_ENTRY}" ]]; then
  echo "Sealed bundle remains writable: ${WRITABLE_ENTRY}" >&2
  exit 2
fi
exit 0

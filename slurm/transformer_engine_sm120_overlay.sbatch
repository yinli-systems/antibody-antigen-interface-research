#!/bin/bash
#SBATCH --job-name=te218-sm120-overlay
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/te218-sm120-overlay-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/te218-sm120-overlay-%j.err

set -euo pipefail
umask 077

SOURCE_ROOT=${TE_SOURCE_ROOT:?TE_SOURCE_ROOT is required}
SOURCE_COMMIT=${TE_SOURCE_COMMIT:?TE_SOURCE_COMMIT is required}
BASE_PYTHON=${TE_BASE_PYTHON:?TE_BASE_PYTHON is required}
BASE_PYTHON_SHA256=${TE_BASE_PYTHON_SHA256:?required}
CONTROLLER_SHA256=${TE_CONTROLLER_SHA256:?required}
OUTPUT_ROOT=${TE_OUTPUT_ROOT:?TE_OUTPUT_ROOT is required}
TE_VERSION=2.18.0
NVCC_VERSION=13.0.88
PARTIAL=${OUTPUT_ROOT}.partial-${SLURM_JOB_ID}
CONTROLLER=${SOURCE_ROOT}/scripts/probe_transformer_engine_sm120.py

for path in "${SOURCE_ROOT}" "${BASE_PYTHON}" "${OUTPUT_ROOT}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${BASE_PYTHON_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${CONTROLLER_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]]
[[ -x "${BASE_PYTHON}" && ! -L "${BASE_PYTHON}" ]]
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
[[ ! -e "${PARTIAL}" && ! -L "${PARTIAL}" ]]
[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]]
[[ "$(sha256sum "${BASE_PYTHON}" | awk '{print $1}')" == "${BASE_PYTHON_SHA256}" ]]
[[ "$(sha256sum "${CONTROLLER}" | awk '{print $1}')" == "${CONTROLLER_SHA256}" ]]
IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Overlay probe requires one scheduler GPU" >&2; exit 2; }

mkdir -m 700 "${PARTIAL}"
"${BASE_PYTHON}" -m pip install \
  --target "${PARTIAL}/site-packages" \
  --no-deps \
  --report "${PARTIAL}/install-report-nvcc.json" \
  "nvidia-cuda-nvcc==${NVCC_VERSION}"

export PYTHONPATH=${PARTIAL}/site-packages
export NVTE_BUILD_USE_NVIDIA_WHEELS=1
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
"${BASE_PYTHON}" -m pip install \
  --target "${PARTIAL}/site-packages" \
  --no-build-isolation \
  --no-deps \
  --report "${PARTIAL}/install-report-te.json" \
  "transformer-engine==${TE_VERSION}" \
  "transformer-engine-cu13==${TE_VERSION}" \
  "transformer-engine-torch==${TE_VERSION}"

"${BASE_PYTHON}" "${CONTROLLER}" \
  --expected-device-name "NVIDIA GeForce RTX 5090" \
  --expected-compute-capability 12.0 \
  --expected-transformer-engine-version "${TE_VERSION}" \
  --install-report-nvcc "${PARTIAL}/install-report-nvcc.json" \
  --install-report-te "${PARTIAL}/install-report-te.json" \
  --output "${PARTIAL}/gpu-capability-receipt.json"

sha256sum "${BASE_PYTHON}" > "${PARTIAL}/base-python-sha256.txt"
find "${PARTIAL}/site-packages" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${PARTIAL}/file-sha256s.txt"
chmod -R a-w "${PARTIAL}"
mv "${PARTIAL}" "${OUTPUT_ROOT}"
sha256sum \
  "${OUTPUT_ROOT}/install-report-nvcc.json" \
  "${OUTPUT_ROOT}/install-report-te.json" \
  "${OUTPUT_ROOT}/gpu-capability-receipt.json" \
  "${OUTPUT_ROOT}/file-sha256s.txt"

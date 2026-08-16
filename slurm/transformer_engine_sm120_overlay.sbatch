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
WHEELHOUSE=${TE_WHEELHOUSE:?TE_WHEELHOUSE is required}
WHEELHOUSE_MANIFEST_SHA256=${TE_WHEELHOUSE_MANIFEST_SHA256:?required}
WHEELHOUSE_ENTRY_COUNT=${TE_WHEELHOUSE_ENTRY_COUNT:?required}
OUTPUT_ROOT=${TE_OUTPUT_ROOT:?TE_OUTPUT_ROOT is required}
TE_VERSION=2.18.0
NVCC_VERSION=13.0.88
CUDA_CRT_VERSION=13.0.88
CUDA_RUNTIME_VERSION=13.0.96
PARTIAL=${OUTPUT_ROOT}.partial-${SLURM_JOB_ID}
CONTROLLER=${SOURCE_ROOT}/scripts/probe_transformer_engine_sm120.py

die() {
  echo "Transformer Engine overlay gate: $*" >&2
  exit 2
}

for path in "${SOURCE_ROOT}" "${BASE_PYTHON}" "${WHEELHOUSE}" "${OUTPUT_ROOT}"; do
  [[ "${path}" == /* ]] || die "all paths must be absolute: ${path}"
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "source commit is invalid"
[[ "${BASE_PYTHON_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "base Python SHA-256 is invalid"
[[ "${CONTROLLER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "controller SHA-256 is invalid"
[[ "${WHEELHOUSE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "wheelhouse manifest SHA-256 is invalid"
[[ "${WHEELHOUSE_ENTRY_COUNT}" =~ ^[1-9][0-9]*$ ]] \
  || die "wheelhouse entry count is invalid"
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]] || die "source root is absent or symbolic"
[[ -x "${BASE_PYTHON}" ]] || die "base Python is absent or not executable: ${BASE_PYTHON}"
BASE_PYTHON_RESOLVED=$(readlink -f -- "${BASE_PYTHON}")
[[ -n "${BASE_PYTHON_RESOLVED}" ]] || die "base Python cannot be resolved"
[[ -f "${BASE_PYTHON_RESOLVED}" && ! -L "${BASE_PYTHON_RESOLVED}" ]] \
  || die "resolved base Python is absent, symbolic, or not a regular file"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || die "output root already exists"
[[ ! -e "${PARTIAL}" && ! -L "${PARTIAL}" ]] || die "partial output already exists"
[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]] || die "source commit differs"
[[ "$(sha256sum "${BASE_PYTHON_RESOLVED}" | awk '{print $1}')" == "${BASE_PYTHON_SHA256}" ]] \
  || die "resolved base Python SHA-256 differs"
[[ "$(sha256sum "${CONTROLLER}" | awk '{print $1}')" == "${CONTROLLER_SHA256}" ]] \
  || die "controller SHA-256 differs"
[[ -d "${WHEELHOUSE}" && ! -L "${WHEELHOUSE}" ]] \
  || die "wheelhouse is absent or symbolic"
[[ -f "${WHEELHOUSE}/SHA256SUMS" && ! -L "${WHEELHOUSE}/SHA256SUMS" ]] \
  || die "wheelhouse manifest is absent or symbolic"
[[ "$(sha256sum "${WHEELHOUSE}/SHA256SUMS" | awk '{print $1}')" == "${WHEELHOUSE_MANIFEST_SHA256}" ]] \
  || die "wheelhouse manifest SHA-256 differs"
(
  cd "${WHEELHOUSE}"
  manifest_entries=0
  while read -r digest filename; do
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "wheelhouse entry digest is invalid"
    [[ -n "${filename}" && "${filename}" == "$(basename -- "${filename}")" ]] \
      || die "wheelhouse entry name is invalid"
    [[ -f "${filename}" && ! -L "${filename}" ]] \
      || die "wheelhouse entry is absent or symbolic"
    [[ "$(sha256sum "${filename}" | awk '{print $1}')" == "${digest}" ]] \
      || die "wheelhouse entry SHA-256 differs: ${filename}"
    manifest_entries=$((manifest_entries + 1))
  done < SHA256SUMS
  actual_entries=$(find . -maxdepth 1 -type f ! -name SHA256SUMS | wc -l)
  actual_children=$(find . -mindepth 1 -maxdepth 1 | wc -l)
  [[ "${manifest_entries}" -eq "${actual_entries}" \
      && "${manifest_entries}" -eq "${WHEELHOUSE_ENTRY_COUNT}" \
      && "${actual_children}" -eq $((WHEELHOUSE_ENTRY_COUNT + 1)) ]] \
    || die "wheelhouse inventory differs"
)
IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Overlay probe requires one scheduler GPU" >&2; exit 2; }

mkdir -m 700 "${PARTIAL}"
PIP_OFFLINE_ARGS=(--no-index --find-links "${WHEELHOUSE}")
"${BASE_PYTHON}" -m pip install \
  "${PIP_OFFLINE_ARGS[@]}" \
  --target "${PARTIAL}/site-packages" \
  --no-deps \
  --report "${PARTIAL}/install-report-nvcc.json" \
  "nvidia-cuda-crt==${CUDA_CRT_VERSION}" \
  "nvidia-cuda-nvcc==${NVCC_VERSION}" \
  "nvidia-cuda-runtime==${CUDA_RUNTIME_VERSION}"

export PYTHONPATH=${PARTIAL}/site-packages
export NVTE_BUILD_USE_NVIDIA_WHEELS=1
export NVTE_FRAMEWORK=pytorch
export NVTE_PYTORCH_FORCE_BUILD=TRUE
export NVTE_WITH_NCCL_EP=0
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
export CUDA_HOME=${PARTIAL}/site-packages/nvidia/cu13
[[ -x "${CUDA_HOME}/bin/nvcc" && -d "${CUDA_HOME}/include" ]] \
  || die "locked NVCC wheel did not materialize a CUDA toolkit"
export PATH=${CUDA_HOME}/bin:${PATH}
nvcc_version=$(nvcc -V | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)
[[ "${nvcc_version}" == "13.0" ]] || die "locked NVCC reports an unexpected CUDA version"
sha256sum "${CUDA_HOME}/bin/nvcc" > "${PARTIAL}/nvcc-sha256.txt"
"${BASE_PYTHON}" -m pip install \
  "${PIP_OFFLINE_ARGS[@]}" \
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

printf '%s\n' "${BASE_PYTHON}" > "${PARTIAL}/base-python-invocation-path.txt"
printf '%s\n' "${BASE_PYTHON_RESOLVED}" > "${PARTIAL}/base-python-resolved-path.txt"
sha256sum "${BASE_PYTHON_RESOLVED}" > "${PARTIAL}/base-python-sha256.txt"
sha256sum "${WHEELHOUSE}/SHA256SUMS" > "${PARTIAL}/wheelhouse-manifest-sha256.txt"
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

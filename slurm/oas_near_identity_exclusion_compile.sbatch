#!/bin/bash
#SBATCH --job-name=oas-nearid-exclusions
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-nearid-exclusions-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-nearid-exclusions-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${OAS_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}
SOURCE_COMMIT=${OAS_SOURCE_COMMIT:?OAS_SOURCE_COMMIT is required}
EXACT_ROOT=${OAS_EXACT_ROOT:?OAS_EXACT_ROOT is required}
GAP_ROOT=${OAS_GAP_ROOT:?OAS_GAP_ROOT is required}
TARGET_ROOT=${OAS_TARGET_ROOT:?OAS_TARGET_ROOT is required}
OUTPUT_ROOT=${OAS_EXCLUSION_OUTPUT_ROOT:?OAS_EXCLUSION_OUTPUT_ROOT is required}
EXPECTED_EXACT=${OAS_EXPECTED_EXACT_RECEIPT_SHA256:?required}
EXPECTED_GAP=${OAS_EXPECTED_GAP_RECEIPT_SHA256:?required}
EXPECTED_TARGET=${OAS_EXPECTED_TARGET_RECEIPT_SHA256:?required}
PYTHON_SHA256=${OAS_PYTHON_SHA256:?required}
CONTROLLER_SHA256=${OAS_CONTROLLER_SHA256:?required}
CORE_SHA256=${OAS_CORE_SHA256:?required}

for path in \
  "${PYTHON_BIN}" "${SOURCE_ROOT}" "${EXACT_ROOT}" "${GAP_ROOT}" \
  "${TARGET_ROOT}" "${OUTPUT_ROOT}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
for digest in \
  "${EXPECTED_EXACT}" "${EXPECTED_GAP}" "${EXPECTED_TARGET}" \
  "${PYTHON_SHA256}" "${CONTROLLER_SHA256}" "${CORE_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ -x "${PYTHON_BIN}" ]]
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]]
[[ -d "${EXACT_ROOT}" && ! -L "${EXACT_ROOT}" ]]
[[ -d "${GAP_ROOT}" && ! -L "${GAP_ROOT}" ]]
[[ -d "${TARGET_ROOT}" && ! -L "${TARGET_ROOT}" ]]
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]]

CONTROLLER=${SOURCE_ROOT}/scripts/compile_oas_near_identity_exclusions.py
CORE=${SOURCE_ROOT}/src/antibody_pretraining/near_identity_exclusion.py
[[ "$(sha256sum "${PYTHON_BIN}" | awk '{print $1}')" == "${PYTHON_SHA256}" ]]
[[ "$(sha256sum "${CONTROLLER}" | awk '{print $1}')" == "${CONTROLLER_SHA256}" ]]
[[ "$(sha256sum "${CORE}" | awk '{print $1}')" == "${CORE_SHA256}" ]]

IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Compiler requires exactly one scheduler-reserved GPU" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=
export OAS_SCHEDULER_RESERVED_GPU_COUNT=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src
export TMPDIR=${OUTPUT_ROOT}.runtime-tmp-${SLURM_JOB_ID}
mkdir -m 700 "${TMPDIR}"

"${PYTHON_BIN}" "${CONTROLLER}" \
  --exact-root "${EXACT_ROOT}" \
  --gap-root "${GAP_ROOT}" \
  --target-root "${TARGET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-exact-receipt-sha256 "${EXPECTED_EXACT}" \
  --expected-gap-receipt-sha256 "${EXPECTED_GAP}" \
  --expected-target-receipt-sha256 "${EXPECTED_TARGET}" \
  --controller-source-sha256 "${CONTROLLER_SHA256}" \
  --task-count 32

rmdir "${TMPDIR}"

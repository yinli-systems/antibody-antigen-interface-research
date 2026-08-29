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
: "${OAS_KERNEL_BUNDLE_ROOT:?OAS_KERNEL_BUNDLE_ROOT is required}"
: "${OAS_KERNEL_BINARY_SHA256:?OAS_KERNEL_BINARY_SHA256 is required}"
: "${OAS_KERNEL_BUILD_RECEIPT_FILE_SHA256:?OAS_KERNEL_BUILD_RECEIPT_FILE_SHA256 is required}"
: "${OAS_PYTHON_BIN:?OAS_PYTHON_BIN is required}"
: "${OAS_PYTHON_SHA256:?OAS_PYTHON_SHA256 is required}"
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
  "${OAS_SEARCH_OUTPUT_ROOT}" \
  "${OAS_KERNEL_BUNDLE_ROOT}"; do
  [[ "${root}" == /* ]] || { echo "All search paths must be absolute" >&2; exit 2; }
done
[[ "${OAS_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "Source commit is invalid" >&2; exit 2; }
for digest in \
  "${OAS_PYTHON_SHA256}" \
  "${OAS_SEARCH_CONTROLLER_SHA256}" \
  "${OAS_SEARCH_KERNEL_SHA256}" \
  "${OAS_SEARCH_SCHEMA_SHA256}" \
  "${OAS_KERNEL_BINARY_SHA256}" \
  "${OAS_KERNEL_BUILD_RECEIPT_FILE_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "Source digest is invalid" >&2; exit 2; }
done
[[ "${OAS_PYTHON_BIN}" == /* && -x "${OAS_PYTHON_BIN}" ]]
PYTHON_REAL=$(readlink -f "${OAS_PYTHON_BIN}")
[[ -f "${PYTHON_REAL}" && ! -L "${PYTHON_REAL}" ]]
[[ "$(sha256sum "${PYTHON_REAL}" | awk '{print $1}')" == "${OAS_PYTHON_SHA256}" ]]
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
KERNEL_BINARY=${OAS_KERNEL_BUNDLE_ROOT}/oas_imgt_exact_search
KERNEL_BUILD_RECEIPT=${OAS_KERNEL_BUNDLE_ROOT}/BUILD_RECEIPT.json
[[ -x "${KERNEL_BINARY}" && ! -L "${KERNEL_BINARY}" ]]
[[ -f "${KERNEL_BUILD_RECEIPT}" && ! -L "${KERNEL_BUILD_RECEIPT}" ]]
[[ "$(sha256sum "${KERNEL_BINARY}" | awk '{print $1}')" == "${OAS_KERNEL_BINARY_SHA256}" ]]
[[ "$(sha256sum "${KERNEL_BUILD_RECEIPT}" | awk '{print $1}')" == "${OAS_KERNEL_BUILD_RECEIPT_FILE_SHA256}" ]]
if find "${OAS_KERNEL_BUNDLE_ROOT}" -perm /222 -print -quit | grep -q .; then
  echo "Canonical kernel bundle remains writable" >&2
  exit 2
fi
"${OAS_PYTHON_BIN}" - \
  "${KERNEL_BUILD_RECEIPT}" \
  "${OAS_KERNEL_BINARY_SHA256}" \
  "${OAS_SEARCH_KERNEL_SHA256}" <<'PY'
import hashlib,json,sys
path, expected_binary, expected_source = sys.argv[1:]
raw = open(path, "rb").read()
receipt = json.loads(raw.decode("ascii"))
claimed = receipt.pop("receipt_sha256")
canonical = (json.dumps(receipt, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")
assert hashlib.sha256(canonical).hexdigest() == claimed
assert receipt["format_version"] == "oas_imgt_exact_kernel_bundle_v1"
assert receipt["claim_boundary"]["canonical_kernel_built_once"] is True
assert receipt["claim_boundary"]["canonical_kernel_canary_passed"] is True
assert receipt["claim_boundary"]["fixed_imgt_full_oas_search_complete"] is False
assert receipt["claim_boundary"]["large_scale_pretraining_authorized"] is False
assert receipt["test_access"]["frozen_official_external_test_accessed"] is False
assert receipt["kernel"]["binary_sha256"] == expected_binary
assert receipt["kernel"]["source_sha256"] == expected_source
PY

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${OAS_SOURCE_ROOT}/src

"${OAS_PYTHON_BIN}" "${CONTROLLER}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --target-output-root "${OAS_TARGET_OUTPUT_ROOT}" \
  --query-root "${OAS_QUERY_ROOT}" \
  --search-output-root "${OAS_SEARCH_OUTPUT_ROOT}" \
  --kernel-binary "${KERNEL_BINARY}" \
  --kernel-source "${KERNEL_SOURCE}" \
  --batch-records 32768 \
  --controller-source-sha256 "${OAS_SEARCH_CONTROLLER_SHA256}" \
  --kernel-binary-sha256 "${OAS_KERNEL_BINARY_SHA256}" \
  --kernel-source-sha256 "${OAS_SEARCH_KERNEL_SHA256}" \
  --match-schema-source-sha256 "${OAS_SEARCH_SCHEMA_SHA256}"

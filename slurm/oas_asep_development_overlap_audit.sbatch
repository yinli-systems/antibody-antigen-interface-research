#!/bin/bash
#SBATCH --job-name=oas-asep-dev-overlap
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/oas-asep-dev-overlap-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/oas-asep-dev-overlap-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
SOURCE_COMMIT=${ABF_SOURCE_COMMIT:?ABF_SOURCE_COMMIT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
ASEP_CACHE_ROOT=${ABF_ASEP_CACHE_ROOT:?ABF_ASEP_CACHE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
MANIFEST_FILE_SHA256=${ABF_CORPUS_MANIFEST_FILE_SHA256:?required}
MANIFEST_PAYLOAD_SHA256=${ABF_CORPUS_MANIFEST_PAYLOAD_SHA256:?required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
AUDIT_FILE_SHA256=${ABF_CORPUS_AUDIT_FILE_SHA256:?required}
AUDIT_PAYLOAD_SHA256=${ABF_CORPUS_AUDIT_PAYLOAD_SHA256:?required}
ASEP_CACHE_MANIFEST_FILE_SHA256=${ABF_ASEP_CACHE_MANIFEST_FILE_SHA256:?required}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${ASEP_CACHE_ROOT}" "${OUTPUT_ROOT}"; do
  [[ "${path}" == /* ]] || { echo "Audit paths must be absolute" >&2; exit 2; }
done
for digest in "${MANIFEST_FILE_SHA256}" "${MANIFEST_PAYLOAD_SHA256}" \
  "${CORPUS_PROTOCOL_FILE_SHA256}" "${AUDIT_FILE_SHA256}" \
  "${AUDIT_PAYLOAD_SHA256}" "${ASEP_CACHE_MANIFEST_FILE_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "Audit anchor is not a SHA-256" >&2; exit 2; }
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "Source commit is invalid" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" && -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]]
[[ -d "${CORPUS_ROOT}" && ! -L "${CORPUS_ROOT}" ]]
[[ -d "${ASEP_CACHE_ROOT}" && ! -L "${ASEP_CACHE_ROOT}" ]]
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] \
  || { echo "Audit output already exists" >&2; exit 2; }

MANIFEST_PATH=${CORPUS_ROOT}/corpus_manifest.json
AUDIT_PATH=${CORPUS_ROOT}/independent_audit_receipt.json
ASEP_CACHE_MANIFEST=${ASEP_CACHE_ROOT}/ASEP_DEVELOPMENT_CACHE_MANIFEST.json
for path in "${MANIFEST_PATH}" "${AUDIT_PATH}" "${ASEP_CACHE_MANIFEST}"; do
  [[ -f "${path}" && ! -L "${path}" ]] \
    || { echo "Audit evidence is unavailable or a symlink" >&2; exit 2; }
done
[[ "$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')" == "${MANIFEST_FILE_SHA256}" ]]
[[ "$(sha256sum "${AUDIT_PATH}" | awk '{print $1}')" == "${AUDIT_FILE_SHA256}" ]]
[[ "$(sha256sum "${ASEP_CACHE_MANIFEST}" | awk '{print $1}')" == \
  "${ASEP_CACHE_MANIFEST_FILE_SHA256}" ]]

IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "Overlap audit requires exactly one scheduler-reserved GPU" >&2; exit 2; }
mkdir -p "$(dirname "${OUTPUT_ROOT}")"
mkdir "${OUTPUT_ROOT}"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "scheduler_reserved_gpu_count=1"
  echo "gpu_compute_permitted=false"
  echo "development_sequence_fields_only=true"
  echo "official_test_accessed=false"
  date --iso-8601=seconds
  nvidia-smi -i "${ALLOCATED_GPU_IDS[0]}" \
    --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total \
    --format=csv
} > "${OUTPUT_ROOT}/environment.txt"

export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/framecontact/src
"${PYTHON_BIN}" -c \
  'import torch; raise SystemExit(0 if torch.cuda.device_count() == 0 else 1)'

RESULT=${OUTPUT_ROOT}/OAS_ASEP_DEVELOPMENT_EXACT_OVERLAP.json
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/audit_oas_asep_development_overlap.py" \
  --production-corpus-root "${CORPUS_ROOT}" \
  --manifest-file-sha256 "${MANIFEST_FILE_SHA256}" \
  --manifest-payload-sha256 "${MANIFEST_PAYLOAD_SHA256}" \
  --corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}" \
  --independent-audit "${AUDIT_PATH}" \
  --independent-audit-file-sha256 "${AUDIT_FILE_SHA256}" \
  --independent-audit-payload-sha256 "${AUDIT_PAYLOAD_SHA256}" \
  --asep-cache-root "${ASEP_CACHE_ROOT}" \
  --asep-cache-manifest-file-sha256 "${ASEP_CACHE_MANIFEST_FILE_SHA256}" \
  --output "${RESULT}" \
  --maximum-open-arrays 128

date --iso-8601=seconds >> "${OUTPUT_ROOT}/environment.txt"
sha256sum "${RESULT}" > "${OUTPUT_ROOT}/result.sha256"
chmod -R a-w "${OUTPUT_ROOT}"
echo "oas_asep_development_exact_overlap_audit=PASS"
echo "result=${RESULT}"

#!/bin/bash
#SBATCH --job-name=abf-oas-4g-pipeline
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-pipeline-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-pipeline-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${OUTPUT_ROOT}" "${RUNTIME_MANIFEST}"; do
  [[ "${path}" == /* ]] || { echo "Pipeline paths must be absolute" >&2; exit 2; }
done
for digest in "${CORPUS_PROTOCOL_FILE_SHA256}" "${RUNTIME_MANIFEST_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "Pipeline anchor is not a SHA-256" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Output root already exists" >&2; exit 2; }

MANIFEST_PATH=${CORPUS_ROOT}/corpus_manifest.json
AUDIT_PATH=${CORPUS_ROOT}/independent_audit_receipt.json
for path in "${MANIFEST_PATH}" "${AUDIT_PATH}"; do
  [[ -f "${path}" && ! -L "${path}" ]] \
    || { echo "Corpus evidence is unavailable or a symlink" >&2; exit 2; }
done
mkdir -p "${OUTPUT_ROOT}"
GATE_PROTOCOL=${OUTPUT_ROOT}/gate_protocol.json
MANIFEST_FILE_SHA256=$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')
AUDIT_FILE_SHA256=$(sha256sum "${AUDIT_PATH}" | awk '{print $1}')
MANIFEST_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["manifest_sha256"])' \
  "${MANIFEST_PATH}")
AUDIT_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["receipt_sha256"])' \
  "${AUDIT_PATH}")
for digest in "${MANIFEST_FILE_SHA256}" "${AUDIT_FILE_SHA256}" \
  "${MANIFEST_PAYLOAD_SHA256}" "${AUDIT_PAYLOAD_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "Corpus evidence has an invalid SHA-256" >&2; exit 2; }
done

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/freeze_antibody_four_gpu_recovery_gate.py" \
  --source-root "${SOURCE_ROOT}" \
  --corpus-root "${CORPUS_ROOT}" \
  --manifest-file-sha256 "${MANIFEST_FILE_SHA256}" \
  --manifest-payload-sha256 "${MANIFEST_PAYLOAD_SHA256}" \
  --corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}" \
  --independent-audit "${AUDIT_PATH}" \
  --independent-audit-file-sha256 "${AUDIT_FILE_SHA256}" \
  --independent-audit-payload-sha256 "${AUDIT_PAYLOAD_SHA256}" \
  --output "${GATE_PROTOCOL}"
GATE_PROTOCOL_SHA256=$(sha256sum "${GATE_PROTOCOL}" | awk '{print $1}')

export ABF_SOURCE_ROOT=${SOURCE_ROOT}
export ABF_OUTPUT_ROOT=${OUTPUT_ROOT}
export ABF_CORPUS_ROOT=${CORPUS_ROOT}
export ABF_CORPUS_MANIFEST_FILE_SHA256=${MANIFEST_FILE_SHA256}
export ABF_CORPUS_MANIFEST_PAYLOAD_SHA256=${MANIFEST_PAYLOAD_SHA256}
export ABF_CORPUS_PROTOCOL_FILE_SHA256=${CORPUS_PROTOCOL_FILE_SHA256}
export ABF_CORPUS_AUDIT_PATH=${AUDIT_PATH}
export ABF_CORPUS_AUDIT_FILE_SHA256=${AUDIT_FILE_SHA256}
export ABF_CORPUS_AUDIT_PAYLOAD_SHA256=${AUDIT_PAYLOAD_SHA256}
export ABF_GATE_PROTOCOL=${GATE_PROTOCOL}
export ABF_GATE_PROTOCOL_SHA256=${GATE_PROTOCOL_SHA256}
export ABF_RUNTIME_MANIFEST=${RUNTIME_MANIFEST}
export ABF_RUNTIME_MANIFEST_SHA256=${RUNTIME_MANIFEST_SHA256}

bash "${SOURCE_ROOT}/slurm/antibody_four_gpu_recovery_gate.sbatch"
RUN_ROOT=${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/audit_antibody_four_gpu_recovery_gate.py" \
  --gate-protocol "${GATE_PROTOCOL}" \
  --continuous "${RUN_ROOT}/continuous" \
  --stage1 "${RUN_ROOT}/stage1" \
  --resumed "${RUN_ROOT}/resumed" \
  --artifact-root "${RUN_ROOT}" \
  --output "${OUTPUT_ROOT}/independent-audit.json"
chmod a-w "${OUTPUT_ROOT}/independent-audit.json" "${OUTPUT_ROOT}/gate_protocol.json"

echo "four_gpu_recovery_pipeline=PASS"
echo "gate_protocol_sha256=${GATE_PROTOCOL_SHA256}"
echo "independent_audit=${OUTPUT_ROOT}/independent-audit.json"

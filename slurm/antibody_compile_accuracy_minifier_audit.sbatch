#!/bin/bash
#SBATCH --job-name=abf-compile-minifier-audit
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/abf-compile-minifier-audit-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/abf-compile-minifier-audit-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
TRAIN_JOB_ID=${ABF_TRAIN_JOB_ID:?ABF_TRAIN_JOB_ID is required}
PROTOCOL_SHA256=${ABF_MINIFIER_PROTOCOL_SHA256:?required}
PREDECESSOR_AUDIT=${ABF_PREDECESSOR_AUDIT:?required}
PREDECESSOR_AUDIT_SHA256=${ABF_PREDECESSOR_AUDIT_SHA256:?required}
AUDITOR_SHA256=${ABF_MINIFIER_AUDITOR_SHA256:?required}
PROTOCOL=${SOURCE_ROOT}/protocols/antibody_compile_accuracy_minifier_v1.json
RUN_ROOT=${OUTPUT_ROOT}/slurm-${TRAIN_JOB_ID}
OUTPUT=${OUTPUT_ROOT}/independent_audit.json

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${OUTPUT_ROOT}" \
  "${PREDECESSOR_AUDIT}" "${RUN_ROOT}"; do
  if [[ "${path}" != /* ]]; then
    echo "All audit paths must be absolute" >&2
    exit 2
  fi
done
test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" = "${PROTOCOL_SHA256}"
test "$(sha256sum "${PREDECESSOR_AUDIT}" | awk '{print $1}')" = \
  "${PREDECESSOR_AUDIT_SHA256}"
test "$(sha256sum "${SOURCE_ROOT}/scripts/audit_antibody_compile_accuracy_minifier.py" \
  | awk '{print $1}')" = "${AUDITOR_SHA256}"
EXPECTED_AUDIT_LAUNCHER_SHA256=$(${PYTHON_BIN} -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["inputs"]["audit_launcher_sha256"])' \
  "${PROTOCOL}")
test "$(sha256sum "${SOURCE_ROOT}/slurm/antibody_compile_accuracy_minifier_audit.sbatch" \
  | awk '{print $1}')" = "${EXPECTED_AUDIT_LAUNCHER_SHA256}"
test -f "${RUN_ROOT}/artifact_sha256.txt"
test ! -w "${RUN_ROOT}/artifact_sha256.txt"
test ! -e "${OUTPUT}"

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
ARTIFACT_MANIFEST_SHA256=$(sha256sum "${RUN_ROOT}/artifact_sha256.txt" \
  | awk '{print $1}')
${PYTHON_BIN} "${SOURCE_ROOT}/scripts/audit_antibody_compile_accuracy_minifier.py" \
  --protocol "${PROTOCOL}" \
  --protocol-file-sha256 "${PROTOCOL_SHA256}" \
  --predecessor-audit "${PREDECESSOR_AUDIT}" \
  --predecessor-audit-file-sha256 "${PREDECESSOR_AUDIT_SHA256}" \
  --run-root "${RUN_ROOT}" \
  --artifact-manifest-file-sha256 "${ARTIFACT_MANIFEST_SHA256}" \
  --auditor-sha256 "${AUDITOR_SHA256}" \
  --output "${OUTPUT}"
chmod a-w "${OUTPUT}"

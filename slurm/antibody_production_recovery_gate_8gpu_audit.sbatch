#!/bin/bash
#SBATCH --job-name=abf-oas-recovery-8g-audit
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/abf-oas-recovery-8g-audit-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/abf-oas-recovery-8g-audit-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
RUN_ROOT=${ABF_RUN_ROOT:?ABF_RUN_ROOT is required}
GATE_PROTOCOL=${ABF_GATE_PROTOCOL:?ABF_GATE_PROTOCOL is required}
CONFIRMATION_PROTOCOL=${ABF_CONFIRMATION_PROTOCOL:?ABF_CONFIRMATION_PROTOCOL is required}
CONFIRMATION_AUDIT=${ABF_CONFIRMATION_AUDIT:?ABF_CONFIRMATION_AUDIT is required}
OUTPUT=${ABF_AUDIT_OUTPUT:?ABF_AUDIT_OUTPUT is required}
GATE_PROTOCOL_SHA256=${ABF_GATE_PROTOCOL_SHA256:?required}
CONFIRMATION_PROTOCOL_SHA256=${ABF_CONFIRMATION_PROTOCOL_SHA256:?required}
CONFIRMATION_AUDIT_SHA256=${ABF_CONFIRMATION_AUDIT_SHA256:?required}
AUDITOR_SHA256=${ABF_AUDITOR_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}
AUDITOR=${SOURCE_ROOT}/scripts/audit_antibody_production_recovery_gate_8gpu.py

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${RUN_ROOT}" \
  "${GATE_PROTOCOL}" "${CONFIRMATION_PROTOCOL}" "${CONFIRMATION_AUDIT}" \
  "${OUTPUT}" "${RUNTIME_MANIFEST}"; do
  if [[ "${path}" != /* ]]; then
    echo "All audit paths must be absolute" >&2
    exit 2
  fi
done
for digest in "${GATE_PROTOCOL_SHA256}" "${CONFIRMATION_PROTOCOL_SHA256}" \
  "${CONFIRMATION_AUDIT_SHA256}" "${AUDITOR_SHA256}" \
  "${RUNTIME_MANIFEST_SHA256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "All audit anchors must be lowercase SHA-256 digests" >&2
    exit 2
  fi
done

test -d "${SOURCE_ROOT}" && test ! -L "${SOURCE_ROOT}"
test -d "${RUN_ROOT}" && test ! -L "${RUN_ROOT}"
test -f "${RUN_ROOT}/artifact_sha256.txt"
test ! -e "${OUTPUT}"
test "$(sha256sum "${GATE_PROTOCOL}" | awk '{print $1}')" = "${GATE_PROTOCOL_SHA256}"
test "$(sha256sum "${CONFIRMATION_PROTOCOL}" | awk '{print $1}')" = "${CONFIRMATION_PROTOCOL_SHA256}"
test "$(sha256sum "${CONFIRMATION_AUDIT}" | awk '{print $1}')" = "${CONFIRMATION_AUDIT_SHA256}"
test "$(sha256sum "${AUDITOR}" | awk '{print $1}')" = "${AUDITOR_SHA256}"

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/scripts:${SOURCE_ROOT}/src

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}"

exec "${PYTHON_BIN}" "${AUDITOR}" \
  --gate-protocol "${GATE_PROTOCOL}" \
  --continuous "${RUN_ROOT}/continuous" \
  --stage1 "${RUN_ROOT}/stage1" \
  --resumed "${RUN_ROOT}/resumed" \
  --confirmation-protocol "${CONFIRMATION_PROTOCOL}" \
  --confirmation-audit "${CONFIRMATION_AUDIT}" \
  --artifact-root "${RUN_ROOT}" \
  --output "${OUTPUT}"

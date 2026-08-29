#!/bin/bash
#SBATCH --job-name=abf-dynamo-block-audit
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/abf-dynamo-block-audit-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/abf-dynamo-block-audit-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
RUN_ROOT=${ABF_RUN_ROOT:?ABF_RUN_ROOT is required}
PROTOCOL=${ABF_BLOCK_PROTOCOL:?ABF_BLOCK_PROTOCOL is required}
PREDECESSOR_PROTOCOL=${ABF_PREDECESSOR_PROTOCOL:?required}
PREDECESSOR_AUDIT=${ABF_PREDECESSOR_AUDIT:?required}
OUTPUT=${ABF_AUDIT_OUTPUT:?ABF_AUDIT_OUTPUT is required}
PROTOCOL_SHA256=${ABF_BLOCK_PROTOCOL_SHA256:?required}
AUDITOR_SHA256=${ABF_BLOCK_AUDITOR_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}
AUDITOR=${SOURCE_ROOT}/scripts/audit_antibody_dynamo_block_bisection.py

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${RUN_ROOT}" "${PROTOCOL}" \
  "${PREDECESSOR_PROTOCOL}" "${PREDECESSOR_AUDIT}" "${OUTPUT}" \
  "${RUNTIME_MANIFEST}"; do
  if [[ "${path}" != /* ]]; then
    echo "All audit paths must be absolute" >&2
    exit 2
  fi
done
for digest in "${PROTOCOL_SHA256}" "${AUDITOR_SHA256}" \
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
test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" = "${PROTOCOL_SHA256}"
test "$(sha256sum "${AUDITOR}" | awk '{print $1}')" = "${AUDITOR_SHA256}"

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/scripts:${SOURCE_ROOT}/src

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}"

exec "${PYTHON_BIN}" "${AUDITOR}" \
  --run-root "${RUN_ROOT}" \
  --protocol "${PROTOCOL}" \
  --predecessor-protocol "${PREDECESSOR_PROTOCOL}" \
  --predecessor-audit "${PREDECESSOR_AUDIT}" \
  --output "${OUTPUT}"

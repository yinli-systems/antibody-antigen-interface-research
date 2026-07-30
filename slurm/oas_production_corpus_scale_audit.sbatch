#!/bin/bash
#SBATCH --job-name=oas-corpus-scale
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-corpus-scale-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-corpus-scale-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
SOURCE_ROOT=${OAS_SCALE_SOURCE_ROOT:?OAS_SCALE_SOURCE_ROOT is required}
CORPUS_ROOT=${OAS_SCALE_CORPUS_ROOT:?OAS_SCALE_CORPUS_ROOT is required}
INDEPENDENT_AUDIT=${OAS_SCALE_INDEPENDENT_AUDIT:?OAS_SCALE_INDEPENDENT_AUDIT is required}
INDEPENDENT_AUDIT_FILE_SHA256=${OAS_SCALE_INDEPENDENT_AUDIT_FILE_SHA256:?required}
INDEPENDENT_AUDIT_PAYLOAD_SHA256=${OAS_SCALE_INDEPENDENT_AUDIT_PAYLOAD_SHA256:?required}
OUTPUT=${OAS_SCALE_OUTPUT:?OAS_SCALE_OUTPUT is required}
PROTOCOL_SHA256=${OAS_SCALE_PROTOCOL_SHA256:?OAS_SCALE_PROTOCOL_SHA256 is required}
AUDITOR_SHA256=${OAS_SCALE_AUDITOR_SHA256:?OAS_SCALE_AUDITOR_SHA256 is required}
RUNTIME_MANIFEST=${OAS_SCALE_RUNTIME_MANIFEST:?OAS_SCALE_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${OAS_SCALE_RUNTIME_MANIFEST_SHA256:?required}
PROTOCOL=${SOURCE_ROOT}/protocols/oas_production_corpus_scale_audit_v1.json
AUDITOR=${SOURCE_ROOT}/scripts/audit_oas_production_corpus_scale.py

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${INDEPENDENT_AUDIT}" "${OUTPUT}" "${RUNTIME_MANIFEST}"; do
  if [[ "${path}" != /* ]]; then
    echo "All scale-audit paths must be absolute" >&2
    exit 2
  fi
done
for digest in "${INDEPENDENT_AUDIT_FILE_SHA256}" \
  "${INDEPENDENT_AUDIT_PAYLOAD_SHA256}" "${PROTOCOL_SHA256}" \
  "${AUDITOR_SHA256}" "${RUNTIME_MANIFEST_SHA256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "All scale-audit anchors must be lowercase SHA-256 digests" >&2
    exit 2
  fi
done

test -x "${PYTHON_BIN}"
test -d "${SOURCE_ROOT}" && test ! -L "${SOURCE_ROOT}"
test -d "${CORPUS_ROOT}" && test ! -L "${CORPUS_ROOT}"
test -f "${INDEPENDENT_AUDIT}" && test ! -L "${INDEPENDENT_AUDIT}"
test ! -e "${OUTPUT}"
test "$(sha256sum "${INDEPENDENT_AUDIT}" | awk '{print $1}')" = "${INDEPENDENT_AUDIT_FILE_SHA256}"
test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" = "${PROTOCOL_SHA256}"
test "$(sha256sum "${AUDITOR}" | awk '{print $1}')" = "${AUDITOR_SHA256}"

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}"

exec "${PYTHON_BIN}" "${AUDITOR}" \
  --source-root "${SOURCE_ROOT}" \
  --corpus-root "${CORPUS_ROOT}" \
  --protocol "${PROTOCOL}" \
  --independent-audit "${INDEPENDENT_AUDIT}" \
  --independent-audit-file-sha256 "${INDEPENDENT_AUDIT_FILE_SHA256}" \
  --independent-audit-payload-sha256 "${INDEPENDENT_AUDIT_PAYLOAD_SHA256}" \
  --worker-count 8 \
  --output "${OUTPUT}"

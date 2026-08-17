#!/bin/bash
#SBATCH --job-name=abf-oas-chain
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-chain-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-chain-%j.err

set -euo pipefail
umask 077

# This is a control-plane continuation job.  It never changes a frozen gate,
# substitutes hardware, or launches a later stage unless the preceding audit
# is present and passes its registered predicate.
PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
RECOVERY_ROOT=${ABF_RECOVERY_ROOT:?ABF_RECOVERY_ROOT is required}
SHAPE_ROOT=${ABF_SHAPE_ROOT:?ABF_SHAPE_ROOT is required}
EFFICIENCY_ROOT=${ABF_EFFICIENCY_ROOT:?ABF_EFFICIENCY_ROOT is required}
CAMPAIGN_ROOT=${ABF_CAMPAIGN_ROOT:?ABF_CAMPAIGN_ROOT is required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?ABF_RUNTIME_MANIFEST_SHA256 is required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?ABF_CORPUS_PROTOCOL_FILE_SHA256 is required}
CHAIN_STAGE=${ABF_CHAIN_STAGE:?ABF_CHAIN_STAGE is required}
CHAIN_SCRIPT=${ABF_CHAIN_SCRIPT:-${SOURCE_ROOT}/slurm/antibody_scale_ladder_chain.sbatch}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${RECOVERY_ROOT}" "${RUNTIME_MANIFEST}" "${CHAIN_SCRIPT}"; do
  [[ "${path}" == /* ]] || { echo "Chain paths must be absolute" >&2; exit 2; }
done
[[ "${CHAIN_STAGE}" == shape || "${CHAIN_STAGE}" == efficiency || \
  "${CHAIN_STAGE}" == campaign ]] || { echo "Unknown chain stage" >&2; exit 2; }

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}
json_field() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="ascii"))
for component in sys.argv[2].split("."):
    value = value[component]
print(value)
PY
}
require_file() {
  [[ -f "$1" && ! -L "$1" ]] || {
    echo "Required audit artifact is unavailable: $1" >&2
    exit 2
  }
}
require_stable_json() {
  local path=$1
  local first second
  # A dependency can release in the same scheduler tick that the producer
  # closes its receipt.  Require a complete, valid, unchanged JSON receipt
  # before deriving any digest; this makes the chain fail closed instead of
  # interpreting a transient/partial read as an anchor mismatch.
  for _ in $(seq 1 30); do
    require_file "${path}"
    if "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="ascii") as handle:
    value = json.load(handle)
receipt = value.get("receipt_sha256")
if not isinstance(receipt, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt):
    raise SystemExit(1)
PY
    then
      first=$(sha256_file "${path}")
      sleep 1
      second=$(sha256_file "${path}")
      if [[ "${first}" == "${second}" ]]; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "Required JSON receipt did not become stable: ${path}" >&2
  exit 2
}
submit_next() {
  local next_stage=$1
  sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
    --export="ALL,ABF_CHAIN_STAGE=${next_stage},ABF_CHAIN_SCRIPT=${CHAIN_SCRIPT}" \
    "${CHAIN_SCRIPT}"
}

RECOVERY_AUDIT=${RECOVERY_ROOT}/independent-audit.json
RECOVERY_GATE=${RECOVERY_ROOT}/gate_protocol.json
SHAPE_AUDIT=${SHAPE_ROOT}/independent-audit.json
SHAPE_PROTOCOL=${SHAPE_ROOT}/protocol-bundle/protocol.json
EFFICIENCY_AUDIT=${EFFICIENCY_ROOT}/independent-audit.json
EFFICIENCY_PROTOCOL=${EFFICIENCY_ROOT}/protocol-bundle/protocol.json
CAMPAIGN_GATE=${CAMPAIGN_ROOT}/campaign_gate.json

case "${CHAIN_STAGE}" in
  shape)
    require_stable_json "${RECOVERY_AUDIT}"
    export ABF_OUTPUT_ROOT=${SHAPE_ROOT}
    export ABF_RECOVERY_AUDIT=${RECOVERY_AUDIT}
    export ABF_RECOVERY_AUDIT_FILE_SHA256=$(sha256_file "${RECOVERY_AUDIT}")
    export ABF_RECOVERY_AUDIT_PAYLOAD_SHA256=$(json_field "${RECOVERY_AUDIT}" receipt_sha256)
    bash "${SOURCE_ROOT}/slurm/antibody_four_gpu_shape_stability_probe_v2.sbatch"
    require_file "${SHAPE_AUDIT}"
    [[ "$(json_field "${SHAPE_AUDIT}" probe_passed)" == True ]] || {
      echo "Shape probe did not pass; efficiency is not submitted" >&2
      exit 3
    }
    next_job=$(submit_next efficiency)
    echo "shape_passed=true"
    echo "next_efficiency_job=${next_job}"
    ;;
  efficiency)
    require_stable_json "${RECOVERY_AUDIT}"
    require_file "${RECOVERY_GATE}"
    require_stable_json "${SHAPE_AUDIT}"
    require_file "${SHAPE_PROTOCOL}"
    export ABF_OUTPUT_ROOT=${EFFICIENCY_ROOT}
    export ABF_RECOVERY_AUDIT=${RECOVERY_AUDIT}
    export ABF_RECOVERY_AUDIT_FILE_SHA256=$(sha256_file "${RECOVERY_AUDIT}")
    export ABF_RECOVERY_AUDIT_PAYLOAD_SHA256=$(json_field "${RECOVERY_AUDIT}" receipt_sha256)
    export ABF_SHAPE_PROBE_AUDIT=${SHAPE_AUDIT}
    export ABF_SHAPE_PROBE_AUDIT_FILE_SHA256=$(sha256_file "${SHAPE_AUDIT}")
    export ABF_SHAPE_PROBE_AUDIT_PAYLOAD_SHA256=$(json_field "${SHAPE_AUDIT}" receipt_sha256)
    export ABF_SHAPE_PROBE_PROTOCOL_FILE_SHA256=$(sha256_file "${SHAPE_PROTOCOL}")
    export ABF_EFFICIENCY_GATE_VERSION=v2
    export ABF_COMPILED_LENGTH_BUCKET_BOUNDARIES=194
    bash "${SOURCE_ROOT}/slurm/antibody_four_gpu_efficiency_pipeline_v2.sbatch"
    require_file "${EFFICIENCY_AUDIT}"
    [[ "$(json_field "${EFFICIENCY_AUDIT}" gate_passed)" == True ]] || {
      echo "Efficiency gate did not pass; campaign is not frozen" >&2
      exit 3
    }
    require_file "${RECOVERY_GATE}"
    "${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/freeze_antibody_four_gpu_pretraining_campaign.py" \
      --source-root "${SOURCE_ROOT}" \
      --recovery-gate "${RECOVERY_GATE}" \
      --recovery-gate-file-sha256 "$(sha256_file "${RECOVERY_GATE}")" \
      --recovery-audit "${RECOVERY_AUDIT}" \
      --recovery-audit-file-sha256 "$(sha256_file "${RECOVERY_AUDIT}")" \
      --recovery-audit-payload-sha256 "$(json_field "${RECOVERY_AUDIT}" receipt_sha256)" \
      --efficiency-protocol "${EFFICIENCY_PROTOCOL}" \
      --efficiency-protocol-file-sha256 "$(sha256_file "${EFFICIENCY_PROTOCOL}")" \
      --efficiency-audit "${EFFICIENCY_AUDIT}" \
      --efficiency-audit-file-sha256 "$(sha256_file "${EFFICIENCY_AUDIT}")" \
      --efficiency-audit-payload-sha256 "$(json_field "${EFFICIENCY_AUDIT}" receipt_sha256)" \
      --output "${CAMPAIGN_GATE}"
    require_file "${CAMPAIGN_GATE}"
    next_job=$(submit_next campaign)
    echo "efficiency_gate_passed=true"
    echo "campaign_gate=${CAMPAIGN_GATE}"
    echo "next_campaign_job=${next_job}"
    ;;
  campaign)
    require_file "${CAMPAIGN_GATE}"
    export ABF_CAMPAIGN_ROOT=${CAMPAIGN_ROOT}
    export ABF_GATE_PROTOCOL=${CAMPAIGN_GATE}
    export ABF_GATE_PROTOCOL_SHA256=$(sha256_file "${CAMPAIGN_GATE}")
    export ABF_SEGMENT_INDEX=0
    unset ABF_PREVIOUS_SEGMENT_ROOT ABF_PREVIOUS_SEGMENT_AUDIT || true
    bash "${SOURCE_ROOT}/slurm/antibody_four_gpu_pretraining_segment.sbatch"
    ;;
esac

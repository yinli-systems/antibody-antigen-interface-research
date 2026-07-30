#!/bin/bash
#SBATCH --job-name=abf-oas-gate-pipe
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-gate-pipe-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-gate-pipe-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
MANIFEST_PATH=$CORPUS_ROOT/corpus_manifest.json
AUDIT_PATH=$CORPUS_ROOT/independent_audit_receipt.json
GATE_PROTOCOL=$OUTPUT_ROOT/gate_protocol.json

test ! -e "$OUTPUT_ROOT"
test -f "$MANIFEST_PATH" && test ! -L "$MANIFEST_PATH"
test -f "$AUDIT_PATH" && test ! -L "$AUDIT_PATH"
mkdir -p "$OUTPUT_ROOT"

MANIFEST_FILE_SHA256=$(sha256sum "$MANIFEST_PATH" | awk '{print $1}')
AUDIT_FILE_SHA256=$(sha256sum "$AUDIT_PATH" | awk '{print $1}')
MANIFEST_PAYLOAD_SHA256=$(
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["manifest_sha256"])' "$MANIFEST_PATH"
)
AUDIT_PAYLOAD_SHA256=$(
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["receipt_sha256"])' "$AUDIT_PATH"
)
for digest in \
  "$MANIFEST_FILE_SHA256" \
  "$MANIFEST_PAYLOAD_SHA256" \
  "$AUDIT_FILE_SHA256" \
  "$AUDIT_PAYLOAD_SHA256" \
  "$CORPUS_PROTOCOL_FILE_SHA256"; do
  test "${#digest}" = 64
  test "$digest" = "$(printf '%s' "$digest" | tr -cd '0-9a-f')"
done

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src:$SOURCE_ROOT/scripts
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/freeze_antibody_production_recovery_gate.py" \
  --source-root "$SOURCE_ROOT" \
  --corpus-root "$CORPUS_ROOT" \
  --manifest-file-sha256 "$MANIFEST_FILE_SHA256" \
  --manifest-payload-sha256 "$MANIFEST_PAYLOAD_SHA256" \
  --corpus-protocol-file-sha256 "$CORPUS_PROTOCOL_FILE_SHA256" \
  --independent-audit "$AUDIT_PATH" \
  --independent-audit-file-sha256 "$AUDIT_FILE_SHA256" \
  --independent-audit-payload-sha256 "$AUDIT_PAYLOAD_SHA256" \
  --output "$GATE_PROTOCOL"

export ABF_SOURCE_ROOT=$SOURCE_ROOT
export ABF_OUTPUT_ROOT=$OUTPUT_ROOT
export ABF_CORPUS_ROOT=$CORPUS_ROOT
export ABF_CORPUS_MANIFEST_FILE_SHA256=$MANIFEST_FILE_SHA256
export ABF_CORPUS_MANIFEST_PAYLOAD_SHA256=$MANIFEST_PAYLOAD_SHA256
export ABF_CORPUS_PROTOCOL_FILE_SHA256=$CORPUS_PROTOCOL_FILE_SHA256
export ABF_CORPUS_AUDIT_PATH=$AUDIT_PATH
export ABF_CORPUS_AUDIT_FILE_SHA256=$AUDIT_FILE_SHA256
export ABF_CORPUS_AUDIT_PAYLOAD_SHA256=$AUDIT_PAYLOAD_SHA256
export ABF_GATE_PROTOCOL=$GATE_PROTOCOL

bash "$SOURCE_ROOT/slurm/antibody_production_loader_recovery_gate.sbatch"

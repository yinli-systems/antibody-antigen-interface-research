#!/bin/bash
# Submit the 8-GPU pilot only after the 4-GPU audit is materialized.
# A failed 4-GPU gate never authorizes production training; this wrapper only
# follows the registered safe-eager fallback into a recovery-only pilot.
#SBATCH --job-name=abf-8g-gated-submit
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-8g-gated-submit-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-8g-gated-submit-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:?ABF_PYTHON_BIN is required}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
FOUR_GPU_OUTPUT=${ABF_FOUR_GPU_OUTPUT:?ABF_FOUR_GPU_OUTPUT is required}
PILOT_OUTPUT_ROOT=${ABF_PILOT_OUTPUT_ROOT:?ABF_PILOT_OUTPUT_ROOT is required}
PILOT_SCRIPT=${ABF_PILOT_SCRIPT:?ABF_PILOT_SCRIPT is required}
BUILD_SCRIPT=${ABF_BUILD_SCRIPT:?ABF_BUILD_SCRIPT is required}
ELIGIBILITY_SCRIPT=${ABF_ELIGIBILITY_SCRIPT:?ABF_ELIGIBILITY_SCRIPT is required}
MODEL_FAMILY=${ABF_MODEL_FAMILY:?ABF_MODEL_FAMILY is required}
TRAINING_PROTOCOL=${ABF_TRAINING_PROTOCOL:?ABF_TRAINING_PROTOCOL is required}
EFFICIENCY_PROTOCOL=${ABF_EFFICIENCY_PROTOCOL:?ABF_EFFICIENCY_PROTOCOL is required}
TARGET_PARTITION=${ABF_TARGET_PARTITION:-gpu_5090}
TARGET_GPU_MODEL=${ABF_TARGET_GPU_MODEL:-NVIDIA GeForce RTX 5090}
TARGET_GPU_COUNT=${ABF_TARGET_GPU_COUNT:-8}

AUDIT=${FOUR_GPU_OUTPUT}/independent-audit.json
SOURCE_PROTOCOL=${FOUR_GPU_OUTPUT}/protocol-bundle/safe_eager_gate.json
DECISION=${PILOT_OUTPUT_ROOT}/gated-submission-${SLURM_JOB_ID}.json
mkdir -p "$PILOT_OUTPUT_ROOT"

if [[ ! -f "$AUDIT" || ! -f "$SOURCE_PROTOCOL" ]]; then
  printf '{"format_version":"antibody_eight_gpu_gated_submission_v1","submitted":false,"reason":"four_gpu_audit_missing","four_gpu_output":"%s"}\n' "$FOUR_GPU_OUTPUT" > "$DECISION"
  exit 0
fi
ELIGIBILITY=${PILOT_OUTPUT_ROOT}/recovery-eligibility-${SLURM_JOB_ID}.json
"$PYTHON_BIN" "$ELIGIBILITY_SCRIPT" --audit "$AUDIT" --output "$ELIGIBILITY"
RECOVERY_ELIGIBLE=$("$PYTHON_BIN" -c 'import json,sys; print(str(json.load(open(sys.argv[1], encoding="ascii"))["eligible"]).lower())' "$ELIGIBILITY")
if [[ "$RECOVERY_ELIGIBLE" != true ]]; then
  printf '{"format_version":"antibody_eight_gpu_gated_submission_v1","submitted":false,"reason":"recovery_eligibility_failed","four_gpu_audit":"%s","recovery_eligibility":"%s"}\n' "$AUDIT" "$ELIGIBILITY" > "$DECISION"
  exit 0
fi

GATE_PROTOCOL=${PILOT_OUTPUT_ROOT}/gate_protocol.json
"$PYTHON_BIN" "$BUILD_SCRIPT" --source "$SOURCE_PROTOCOL" --output "$GATE_PROTOCOL" --target-gpu-model "$TARGET_GPU_MODEL" --target-world-size "$TARGET_GPU_COUNT"

MANIFEST=${CORPUS_ROOT}/corpus_manifest.json
CORPUS_AUDIT=${CORPUS_ROOT}/independent_audit_receipt.json
CORPUS_PROTOCOL=${CORPUS_ROOT}/decontamination_protocol.json
MANIFEST_FILE_SHA256=$(sha256sum "$MANIFEST" | awk '{print $1}')
MANIFEST_PAYLOAD_SHA256=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["manifest_sha256"])' "$MANIFEST")
CORPUS_PROTOCOL_FILE_SHA256=$(sha256sum "$CORPUS_PROTOCOL" | awk '{print $1}')
CORPUS_AUDIT_FILE_SHA256=$(sha256sum "$CORPUS_AUDIT" | awk '{print $1}')
CORPUS_AUDIT_PAYLOAD_SHA256=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["receipt_sha256"])' "$CORPUS_AUDIT")

JOB_ID=$(sbatch --parsable --partition="$TARGET_PARTITION" --nodes=1 --ntasks=1 --gres=gpu:"$TARGET_GPU_COUNT" --cpus-per-task=32 --time=04:00:00 \
  --export=ALL,ABF_PYTHON_BIN="$PYTHON_BIN",ABF_SOURCE_ROOT="$SOURCE_ROOT",ABF_CORPUS_ROOT="$CORPUS_ROOT",ABF_OUTPUT_ROOT="$PILOT_OUTPUT_ROOT",ABF_GATE_PROTOCOL="$GATE_PROTOCOL",ABF_CORPUS_PROTOCOL_FILE_SHA256="$CORPUS_PROTOCOL_FILE_SHA256",ABF_CORPUS_MANIFEST_FILE_SHA256="$MANIFEST_FILE_SHA256",ABF_CORPUS_MANIFEST_PAYLOAD_SHA256="$MANIFEST_PAYLOAD_SHA256",ABF_CORPUS_AUDIT="$CORPUS_AUDIT",ABF_CORPUS_AUDIT_FILE_SHA256="$CORPUS_AUDIT_FILE_SHA256",ABF_CORPUS_AUDIT_PAYLOAD_SHA256="$CORPUS_AUDIT_PAYLOAD_SHA256",ABF_MODEL_FAMILY="$MODEL_FAMILY",ABF_TRAINING_PROTOCOL="$TRAINING_PROTOCOL",ABF_EFFICIENCY_PROTOCOL="$EFFICIENCY_PROTOCOL" \
  "$PILOT_SCRIPT")
printf '{"format_version":"antibody_scale_gpu_gated_submission_v1","submitted":true,"four_gpu_audit":"%s","recovery_eligibility":"%s","pilot_job_id":"%s","pilot_output_root":"%s","gate_protocol":"%s","target_partition":"%s","target_gpu_model":"%s","target_gpu_count":%s,"production_pretraining_authorized":false}\n' "$AUDIT" "$ELIGIBILITY" "$JOB_ID" "$PILOT_OUTPUT_ROOT" "$GATE_PROTOCOL" "$TARGET_PARTITION" "$TARGET_GPU_MODEL" "$TARGET_GPU_COUNT" > "$DECISION"
echo "eight_gpu_pilot_job_id=$JOB_ID"

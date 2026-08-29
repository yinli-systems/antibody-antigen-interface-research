#!/bin/bash
#SBATCH --job-name=abf-elastic-4to2to4
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-elastic-4to2to4-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-elastic-4to2to4-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/contact/bin/python3.10}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
OUTPUT_BASE=${ABF_OUTPUT_BASE:?ABF_OUTPUT_BASE is required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}

for path in "$PYTHON_BIN" "$SOURCE_ROOT" "$CORPUS_ROOT" "$OUTPUT_BASE" "$RUNTIME_MANIFEST"; do
  [[ "$path" == /* ]] || { echo "All elastic-gate paths must be absolute" >&2; exit 2; }
done
[[ -x "$PYTHON_BIN" && -f "$RUNTIME_MANIFEST" ]] \
  || { echo "Elastic-gate runtime evidence is unavailable" >&2; exit 2; }
[[ "$RUNTIME_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "Runtime manifest digest is invalid" >&2; exit 2; }
[[ "$CORPUS_PROTOCOL_FILE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "Corpus protocol digest is invalid" >&2; exit 2; }

RUN_ROOT=$OUTPUT_BASE/slurm-$SLURM_JOB_ID
[[ ! -e "$RUN_ROOT" ]] || { echo "Run root must be create-only" >&2; exit 2; }
mkdir -p "$OUTPUT_BASE"
mkdir "$RUN_ROOT"
MONITOR_PID=

seal_run() {
  if [[ -d "$RUN_ROOT" && ! -e "$RUN_ROOT/artifact_sha256.txt" ]]; then
    find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum \
      > "$RUN_ROOT/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "$RUN_ROOT" 2>/dev/null || true
}
cleanup() {
  local exit_code=$?
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=
  fi
  seal_run
  return "$exit_code"
}
trap cleanup EXIT

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src:$SOURCE_ROOT/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
ulimit -Sn 4096

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/generate_runtime_manifest.py" verify \
  --manifest "$RUNTIME_MANIFEST" \
  --expected-file-sha256 "$RUNTIME_MANIFEST_SHA256" \
  > "$RUN_ROOT/runtime_verification.json"
[[ "$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')" == 4 ]] \
  || { echo "Elastic gate requires exactly four visible GPUs" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader \
  | awk '$0 == "NVIDIA GeForce RTX 4090" {n++} END {print n+0}')" == 4 ]] \
  || { echo "Elastic gate requires four RTX 4090 GPUs" >&2; exit 2; }

MANIFEST_PATH=$CORPUS_ROOT/corpus_manifest.json
AUDIT_PATH=$CORPUS_ROOT/independent_audit_receipt.json
MANIFEST_FILE_SHA256=$(sha256sum "$MANIFEST_PATH" | awk '{print $1}')
AUDIT_FILE_SHA256=$(sha256sum "$AUDIT_PATH" | awk '{print $1}')
MANIFEST_PAYLOAD_SHA256=$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["manifest_sha256"])' \
  "$MANIFEST_PATH")
AUDIT_PAYLOAD_SHA256=$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["receipt_sha256"])' \
  "$AUDIT_PATH")
GATE_PROTOCOL=$RUN_ROOT/gate_protocol.json
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/freeze_antibody_elastic_recovery_gate.py" \
  --source-root "$SOURCE_ROOT" \
  --corpus-root "$CORPUS_ROOT" \
  --manifest-file-sha256 "$MANIFEST_FILE_SHA256" \
  --manifest-payload-sha256 "$MANIFEST_PAYLOAD_SHA256" \
  --corpus-protocol-file-sha256 "$CORPUS_PROTOCOL_FILE_SHA256" \
  --independent-audit "$AUDIT_PATH" \
  --independent-audit-file-sha256 "$AUDIT_FILE_SHA256" \
  --independent-audit-payload-sha256 "$AUDIT_PAYLOAD_SHA256" \
  --output "$GATE_PROTOCOL"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "source_root=$SOURCE_ROOT"
  echo "run_root=$RUN_ROOT"
  echo "runtime_manifest_sha256=$RUNTIME_MANIFEST_SHA256"
  echo "gate_protocol_sha256=$(sha256sum "$GATE_PROTOCOL" | awk '{print $1}')"
  echo "claim_boundary=real-OAS elastic recovery systems gate only"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
  nvidia-smi topo -m
} > "$RUN_ROOT/environment.txt"
nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,pstate \
  --format=csv -l 5 > "$RUN_ROOT/gpu.csv" &
MONITOR_PID=$!

COMMON_ARGS=(
  --production-corpus-root "$CORPUS_ROOT"
  --production-corpus-manifest-file-sha256 "$MANIFEST_FILE_SHA256"
  --production-corpus-manifest-payload-sha256 "$MANIFEST_PAYLOAD_SHA256"
  --production-corpus-protocol-file-sha256 "$CORPUS_PROTOCOL_FILE_SHA256"
  --production-corpus-independent-audit "$AUDIT_PATH"
  --production-corpus-independent-audit-file-sha256 "$AUDIT_FILE_SHA256"
  --production-corpus-independent-audit-payload-sha256 "$AUDIT_PAYLOAD_SHA256"
  --gate-protocol "$GATE_PROTOCOL"
  --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json"
  --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json"
  --efficiency-protocol "$SOURCE_ROOT/protocols/antibody_efficiency_gate_v1.json"
  --preset abf_35m
  --micro-batch 16
  --target-tokens 16384
  --batch-quantum 128
  --run-seed 17
  --length-bucket-boundaries 194
  --ddp-bucket-cap-mb 25
  --maximum-open-arrays 1024
  --minimum-soft-open-file-limit 4096
  --fused-adamw --pinned-staging --bucketed-padding --length-bucketing
)

run_stage() {
  local visible_devices=$1
  local process_count=$2
  local run_key=$3
  local run_role=$4
  local step_count=$5
  local output_root=$6
  local resume_root=${7:-}
  local resume_args=()
  if [[ -n "$resume_root" ]]; then
    resume_args=(--resume-root "$resume_root")
  fi
  CUDA_VISIBLE_DEVICES=$visible_devices "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc-per-node="$process_count" \
    "$SOURCE_ROOT/scripts/run_antibody_pretraining_production_gate.py" \
    "${COMMON_ARGS[@]}" \
    --output "$output_root" \
    --run-role "$run_role" \
    --elastic-run-key "$run_key" \
    --steps "$step_count" \
    "${resume_args[@]}" \
    2>&1 | tee "$RUN_ROOT/$run_key.log"
}

ALL_VISIBLE=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}
FIRST_TWO_VISIBLE=$(printf '%s\n' "$ALL_VISIBLE" | cut -d, -f1,2)
[[ "$ALL_VISIBLE" == *,*,*,* && "$FIRST_TWO_VISIBLE" == *,* ]] \
  || { echo "Slurm visible-device inventory differs" >&2; exit 2; }

run_stage "$ALL_VISIBLE" 4 continuous_4 continuous 4 "$RUN_ROOT/continuous_4"
run_stage "$ALL_VISIBLE" 4 stage1_4 stage1 2 "$RUN_ROOT/stage1_4"
run_stage "$FIRST_TWO_VISIBLE" 2 stage2_2 resumed 1 "$RUN_ROOT/stage2_2" \
  "$RUN_ROOT/stage1_4"
run_stage "$ALL_VISIBLE" 4 stage3_4 resumed 1 "$RUN_ROOT/stage3_4" \
  "$RUN_ROOT/stage2_2"

OPTIMIZER_DIAGNOSTIC=$RUN_ROOT/optimizer_diagnostic.json
CUDA_VISIBLE_DEVICES=$FIRST_TWO_VISIBLE "$PYTHON_BIN" \
  "$SOURCE_ROOT/scripts/diagnose_antibody_optimizer_state.py" \
  --left "$RUN_ROOT/continuous_4/checkpoint" \
  --right "$RUN_ROOT/stage3_4/checkpoint" \
  --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json" \
  --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json" \
  --preset abf_35m \
  --output "$OPTIMIZER_DIAGNOSTIC"

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/audit_antibody_elastic_recovery.py" \
  --gate-protocol "$GATE_PROTOCOL" \
  --continuous "$RUN_ROOT/continuous_4" \
  --stage1 "$RUN_ROOT/stage1_4" \
  --stage2 "$RUN_ROOT/stage2_2" \
  --stage3 "$RUN_ROOT/stage3_4" \
  --optimizer-diagnostic "$OPTIMIZER_DIAGNOSTIC" \
  --output "$RUN_ROOT/audit.json"

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=
date --iso-8601=seconds >> "$RUN_ROOT/environment.txt"
find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"
trap - EXIT
echo "elastic_recovery_audit=PASS"
echo "elastic_recovery_audit_path=$RUN_ROOT/audit.json"

#!/bin/bash
# Development-only 8-GPU real-OAS recovery pilot; no production authorization.
# All source, corpus, and protocol anchors are supplied explicitly by the caller.
#SBATCH --job-name=abf-oas-8g-pilot
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-8g-pilot-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-8g-pilot-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:?ABF_PYTHON_BIN is required}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
GATE_PROTOCOL=${ABF_GATE_PROTOCOL:?ABF_GATE_PROTOCOL is required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
CORPUS_MANIFEST_FILE_SHA256=${ABF_CORPUS_MANIFEST_FILE_SHA256:?required}
CORPUS_MANIFEST_PAYLOAD_SHA256=${ABF_CORPUS_MANIFEST_PAYLOAD_SHA256:?required}
CORPUS_AUDIT=${ABF_CORPUS_AUDIT:?ABF_CORPUS_AUDIT is required}
CORPUS_AUDIT_FILE_SHA256=${ABF_CORPUS_AUDIT_FILE_SHA256:?required}
CORPUS_AUDIT_PAYLOAD_SHA256=${ABF_CORPUS_AUDIT_PAYLOAD_SHA256:?required}
MODEL_FAMILY=${ABF_MODEL_FAMILY:?ABF_MODEL_FAMILY is required}
TRAINING_PROTOCOL=${ABF_TRAINING_PROTOCOL:?ABF_TRAINING_PROTOCOL is required}
EFFICIENCY_PROTOCOL=${ABF_EFFICIENCY_PROTOCOL:?ABF_EFFICIENCY_PROTOCOL is required}
STEPS_CONTINUOUS=${ABF_STEPS_CONTINUOUS:-200}
STEPS_STAGE1=${ABF_STEPS_STAGE1:-100}

for path in "$PYTHON_BIN" "$SOURCE_ROOT" "$CORPUS_ROOT" "$OUTPUT_ROOT" "$GATE_PROTOCOL" "$CORPUS_AUDIT" "$MODEL_FAMILY" "$TRAINING_PROTOCOL" "$EFFICIENCY_PROTOCOL"; do
  [[ "$path" == /* ]] || { echo "all paths must be absolute" >&2; exit 2; }
done
WORLD_SIZE=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["execution"]["world_size"])' "$GATE_PROTOCOL")
[[ "$WORLD_SIZE" == 6 || "$WORLD_SIZE" == 8 ]] || { echo "unsupported scale-pilot world size" >&2; exit 2; }
TARGET_TOKENS=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["pilot"]["target_residue_tokens_per_update"])' "$GATE_PROTOCOL")
BATCH_QUANTUM=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["pilot"]["batch_quantum_sequences"])' "$GATE_PROTOCOL")
[[ "$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')" == "$WORLD_SIZE" ]] || { echo "visible GPU count differs from protocol" >&2; exit 2; }
EXPECTED_GPU_MODEL=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["hardware_identity"]["gpu_model"])' "$GATE_PROTOCOL")
[[ "$EXPECTED_GPU_MODEL" == "NVIDIA GeForce RTX 4090" || "$EXPECTED_GPU_MODEL" == "NVIDIA GeForce RTX 5090" ]] || { echo "unsupported target GPU model" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | awk -v expected="$EXPECTED_GPU_MODEL" '$0 == expected {n++} END {print n+0}')" == "$WORLD_SIZE" ]] || { echo "target GPU identity/count differs" >&2; exit 2; }
NCCL_ALGO=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["pilot"]["collective_determinism"]["nccl_algo"])' "$GATE_PROTOCOL")
NCCL_PROTO=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["pilot"]["collective_determinism"]["nccl_proto"])' "$GATE_PROTOCOL")
NCCL_MIN_NCHANNELS=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["pilot"]["collective_determinism"]["nccl_min_nchannels"])' "$GATE_PROTOCOL")
NCCL_MAX_NCHANNELS=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["pilot"]["collective_determinism"]["nccl_max_nchannels"])' "$GATE_PROTOCOL")
[[ "$NCCL_ALGO:$NCCL_PROTO:$NCCL_MIN_NCHANNELS:$NCCL_MAX_NCHANNELS" == "Ring:Simple:1:1" ]] || { echo "collective determinism values differ" >&2; exit 2; }

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT/src:$SOURCE_ROOT/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN ABF_TELEMETRY_INTERVAL_SECONDS=5
export NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
ulimit -Sn 4096

RUN_ROOT="$OUTPUT_ROOT/slurm-${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" ]] || { echo "run root exists" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
cp "$GATE_PROTOCOL" "$RUN_ROOT/protocol.json"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv > "$RUN_ROOT/environment.txt"
printf 'job_id=%s\nnode=%s\nworld_size=%s\ngpu_model=%s\nclaim_boundary=development-only real-OAS recovery pilot\n' "$SLURM_JOB_ID" "$(hostname)" "$WORLD_SIZE" "$EXPECTED_GPU_MODEL" >> "$RUN_ROOT/environment.txt"

MONITOR_PID=
stop_monitor() { if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; MONITOR_PID=; fi; }
trap 'stop_monitor' EXIT
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate --format=csv -l 5 > "$RUN_ROOT/gpu.csv" & MONITOR_PID=$!

COMMON=(
  --production-corpus-root "$CORPUS_ROOT"
  --production-corpus-manifest-file-sha256 "$CORPUS_MANIFEST_FILE_SHA256"
  --production-corpus-manifest-payload-sha256 "$CORPUS_MANIFEST_PAYLOAD_SHA256"
  --production-corpus-protocol-file-sha256 "$CORPUS_PROTOCOL_FILE_SHA256"
  --production-corpus-independent-audit "$CORPUS_AUDIT"
  --production-corpus-independent-audit-file-sha256 "$CORPUS_AUDIT_FILE_SHA256"
  --production-corpus-independent-audit-payload-sha256 "$CORPUS_AUDIT_PAYLOAD_SHA256"
  --gate-protocol "$GATE_PROTOCOL" --model-family "$MODEL_FAMILY"
  --training-protocol "$TRAINING_PROTOCOL" --efficiency-protocol "$EFFICIENCY_PROTOCOL"
  --preset abf_650m --micro-batch 16 --target-tokens "$TARGET_TOKENS" --batch-quantum "$BATCH_QUANTUM"
  --run-seed 17 --length-bucket-boundaries 96,128,160,194
  --ddp-bucket-cap-mb 25 --maximum-open-arrays 1024 --minimum-soft-open-file-limit 4096
)
run_role() {
  local role="$1" steps="$2" output="$RUN_ROOT/$1"
  local extra=()
  if [[ "$role" == resumed ]]; then extra+=(--resume-root "$RUN_ROOT/stage1/run"); fi
  if [[ "$role" != stage1 ]]; then extra+=(--skip-checkpoint); fi
  mkdir -p "$output"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="$WORLD_SIZE" \
    "$SOURCE_ROOT/scripts/run_antibody_pretraining_production_gate.py" \
    "${COMMON[@]}" --run-role "$role" --steps "$steps" --output "$output/run" \
    "${extra[@]}" --authorization-mode recovery_gate 2>&1 | tee "$output/training.log"
}
run_role continuous "$STEPS_CONTINUOUS"
run_role stage1 "$STEPS_STAGE1"
run_role resumed "$STEPS_STAGE1"
stop_monitor
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/verify_antibody_eight_gpu_recovery.py" \
  --protocol "$GATE_PROTOCOL" --continuous "$RUN_ROOT/continuous/run" \
  --stage1 "$RUN_ROOT/stage1/run" --resumed "$RUN_ROOT/resumed/run" \
  --output "$RUN_ROOT/independent-audit.json"
find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"
trap - EXIT
echo "eight_gpu_real_oas_recovery_pilot=PASS"

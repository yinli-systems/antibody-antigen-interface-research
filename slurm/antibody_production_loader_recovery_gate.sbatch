#!/bin/bash
#SBATCH --job-name=abf-oas-recovery
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-recovery-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-recovery-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
CORPUS_MANIFEST_FILE_SHA256=${ABF_CORPUS_MANIFEST_FILE_SHA256:?required}
CORPUS_MANIFEST_PAYLOAD_SHA256=${ABF_CORPUS_MANIFEST_PAYLOAD_SHA256:?required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
CORPUS_AUDIT_PATH=${ABF_CORPUS_AUDIT_PATH:?ABF_CORPUS_AUDIT_PATH is required}
CORPUS_AUDIT_FILE_SHA256=${ABF_CORPUS_AUDIT_FILE_SHA256:?required}
CORPUS_AUDIT_PAYLOAD_SHA256=${ABF_CORPUS_AUDIT_PAYLOAD_SHA256:?required}
GATE_PROTOCOL=${ABF_GATE_PROTOCOL:?ABF_GATE_PROTOCOL is required}
PRESET=${ABF_PRESET:-abf_650m}
MICRO_BATCH=${ABF_MICRO_BATCH:-16}
TARGET_TOKENS=${ABF_TARGET_TOKENS:-16384}
BATCH_QUANTUM=${ABF_BATCH_QUANTUM:-128}
RUN_SEED=${ABF_RUN_SEED:-17}
LENGTH_BUCKET_BOUNDARIES=${ABF_LENGTH_BUCKET_BOUNDARIES:-96,128,160,194}
DDP_BUCKET_CAP_MB=${ABF_DDP_BUCKET_CAP_MB:-25}

RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SOURCE_ROOT/src:$SOURCE_ROOT/scripts
LOCAL_CACHE=${SLURM_TMPDIR:-/tmp}/abf-oas-recovery-$SLURM_JOB_ID
mkdir -p "$LOCAL_CACHE/inductor" "$LOCAL_CACHE/triton" "$LOCAL_CACHE/torch_extensions"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_CACHE/inductor
export TRITON_CACHE_DIR=$LOCAL_CACHE/triton
export TORCH_EXTENSIONS_DIR=$LOCAL_CACHE/torch_extensions

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 > "$RUN_ROOT/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

COMMON_ARGS=(
  --production-corpus-root "$CORPUS_ROOT"
  --production-corpus-manifest-file-sha256 "$CORPUS_MANIFEST_FILE_SHA256"
  --production-corpus-manifest-payload-sha256 "$CORPUS_MANIFEST_PAYLOAD_SHA256"
  --production-corpus-protocol-file-sha256 "$CORPUS_PROTOCOL_FILE_SHA256"
  --production-corpus-independent-audit "$CORPUS_AUDIT_PATH"
  --production-corpus-independent-audit-file-sha256 "$CORPUS_AUDIT_FILE_SHA256"
  --production-corpus-independent-audit-payload-sha256 "$CORPUS_AUDIT_PAYLOAD_SHA256"
  --gate-protocol "$GATE_PROTOCOL"
  --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json"
  --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json"
  --efficiency-protocol "$SOURCE_ROOT/protocols/antibody_efficiency_gate_v1.json"
  --preset "$PRESET"
  --micro-batch "$MICRO_BATCH"
  --target-tokens "$TARGET_TOKENS"
  --batch-quantum "$BATCH_QUANTUM"
  --run-seed "$RUN_SEED"
  --length-bucket-boundaries "$LENGTH_BUCKET_BOUNDARIES"
  --ddp-bucket-cap-mb "$DDP_BUCKET_CAP_MB"
  --compile-blocks
  --disable-compiled-cudagraphs
  --compile-prewarm-buckets
  --fused-adamw
  --pinned-staging
  --bucketed-padding
  --length-bucketing
)

run_gate() {
  local role=$1
  local steps=$2
  local output=$3
  shift 3
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=1 \
    "$SOURCE_ROOT/scripts/run_antibody_pretraining_production_gate.py" \
    "${COMMON_ARGS[@]}" --run-role "$role" --steps "$steps" \
    --output "$output" "$@"
}

run_gate continuous 4 "$RUN_ROOT/continuous" --skip-checkpoint
run_gate stage1 2 "$RUN_ROOT/stage1"
run_gate resumed 2 "$RUN_ROOT/resumed" \
  --resume-root "$RUN_ROOT/stage1" --skip-checkpoint

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/audit_antibody_production_loader_recovery.py" \
  --gate-protocol "$GATE_PROTOCOL" \
  --continuous "$RUN_ROOT/continuous" \
  --stage1 "$RUN_ROOT/stage1" \
  --resumed "$RUN_ROOT/resumed" \
  --output "$RUN_ROOT/independent-audit.json"

chmod -R a-w "$RUN_ROOT"

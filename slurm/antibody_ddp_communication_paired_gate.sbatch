#!/bin/bash
#SBATCH --job-name=abf-ddp-comm-gate
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-ddp-comm-gate-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-ddp-comm-gate-%j.err

set -euo pipefail
umask 077
PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
COMMUNICATION_PROTOCOL=$SOURCE_ROOT/protocols/antibody_ddp_communication_optimization_v1.json
PAIRED_PROTOCOL=$SOURCE_ROOT/protocols/antibody_ddp_communication_paired_gate_v1.json
RUN_ORDER=safe_eager_r1,fp32_compiled_r1,bf16_selected_r1,bf16_selected_r2,fp32_compiled_r2,safe_eager_r2
PAIR_ROOT=$OUTPUT_ROOT/paired-$SLURM_JOB_ID
test ! -e "$PAIR_ROOT"
mkdir -p "$PAIR_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SOURCE_ROOT/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
PAIRED_PROTOCOL_SHA256=$(sha256sum "$PAIRED_PROTOCOL" | awk '{print $1}')
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/capture_antibody_communication_allocation.py" \
  --output "$PAIR_ROOT/allocation.json" \
  --protocol-sha256 "$PAIRED_PROTOCOL_SHA256" \
  --run-order "$RUN_ORDER"

MONITOR_PID=
cleanup_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=
  fi
}
trap cleanup_monitor EXIT

run_mode() {
  local label=$1
  local mode_root=$PAIR_ROOT/$label
  local cache_root=${SLURM_TMPDIR:-/tmp}/abf-ddp-comm-$SLURM_JOB_ID-$label
  mkdir -p "$mode_root" "$cache_root/inductor" "$cache_root/triton" \
    "$cache_root/torch_extensions"
  export TORCHINDUCTOR_CACHE_DIR=$cache_root/inductor
  export TRITON_CACHE_DIR=$cache_root/triton
  export TORCH_EXTENSIONS_DIR=$cache_root/torch_extensions
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
    --format=csv -l 5 > "$PAIR_ROOT/gpu-$label.csv" &
  MONITOR_PID=$!
  local args=(
    --shard "$SHARD_ROOT"
    --manifest-sha256 "$MANIFEST_SHA256"
    --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json"
    --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json"
    --efficiency-protocol "$SOURCE_ROOT/protocols/antibody_efficiency_gate_v1.json"
    --output "$mode_root/run"
    --preset abf_650m
    --steps 500
    --micro-batch 16
    --target-tokens 16384
    --batch-quantum 128
    --backend ddp
    --ddp-bucket-cap-mb 25
    --fsdp-output-dtype bfloat16
    --length-bucket-boundaries 96,128,144,194
    --deterministic-algorithms
    --audit-gradient-rank-parity
    --skip-checkpoint
  )
  case "$label" in
    safe_eager_r1|safe_eager_r2) ;;
    fp32_compiled_r1|fp32_compiled_r2)
      args+=(
        --ddp-communication-protocol "$COMMUNICATION_PROTOCOL"
        --compile-blocks
        --compile-prewarm-buckets
        --fused-adamw
        --pinned-staging
        --bucketed-padding
      )
      ;;
    bf16_selected_r1|bf16_selected_r2)
      args+=(
        --ddp-communication-protocol "$COMMUNICATION_PROTOCOL"
        --ddp-communication-dtype bfloat16
        --ddp-batched-grad-copy
        --compile-blocks
        --compile-prewarm-buckets
        --fused-adamw
        --pinned-staging
        --bucketed-padding
      )
      ;;
    *) exit 2 ;;
  esac
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "$SOURCE_ROOT/scripts/run_antibody_pretraining_optimized.py" "${args[@]}"
  cleanup_monitor
}

IFS=',' read -r -a run_labels <<< "$RUN_ORDER"
for label in "${run_labels[@]}"; do
  run_mode "$label"
done

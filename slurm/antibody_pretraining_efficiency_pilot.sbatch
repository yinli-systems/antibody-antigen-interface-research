#!/bin/bash
#SBATCH --job-name=abf-efficiency
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-efficiency-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-efficiency-%j.err

set -euo pipefail
umask 077
PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
BACKEND=${ABF_BACKEND:-ddp}
PRESET=${ABF_PRESET:-abf_650m}
STEPS=${ABF_STEPS:-20}
MICRO_BATCH=${ABF_MICRO_BATCH:-16}
TARGET_TOKENS=${ABF_TARGET_TOKENS:-16384}
BATCH_QUANTUM=${ABF_BATCH_QUANTUM:-128}
DDP_BUCKET_CAP_MB=${ABF_DDP_BUCKET_CAP_MB:-25}
FSDP_OUTPUT_DTYPE=${ABF_FSDP_OUTPUT_DTYPE:-bfloat16}
COMPILE_BLOCKS=${ABF_COMPILE_BLOCKS:-0}
FUSED_ADAMW=${ABF_FUSED_ADAMW:-0}
PINNED_STAGING=${ABF_PINNED_STAGING:-0}
LENGTH_BUCKETING=${ABF_LENGTH_BUCKETING:-0}
SKIP_CHECKPOINT=${ABF_SKIP_CHECKPOINT:-1}
case "$BACKEND" in ddp|fsdp2) ;; *) exit 2 ;; esac
case "$PRESET" in abf_300m|abf_650m) ;; *) exit 2 ;; esac
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID-$PRESET-$BACKEND
test ! -e "$RUN_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SOURCE_ROOT/src
LOCAL_CACHE=${SLURM_TMPDIR:-/tmp}/abf-efficiency-$SLURM_JOB_ID
mkdir -p "$LOCAL_CACHE/inductor" "$LOCAL_CACHE/triton" "$LOCAL_CACHE/torch_extensions"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_CACHE/inductor
export TRITON_CACHE_DIR=$LOCAL_CACHE/triton
export TORCH_EXTENSIONS_DIR=$LOCAL_CACHE/torch_extensions
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 2 > "$OUTPUT_ROOT/gpu-$SLURM_JOB_ID-$PRESET-$BACKEND.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
ARGS=(
  --shard "$SHARD_ROOT"
  --manifest-sha256 "$MANIFEST_SHA256"
  --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json"
  --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json"
  --efficiency-protocol "$SOURCE_ROOT/protocols/antibody_efficiency_gate_v1.json"
  --output "$RUN_ROOT"
  --preset "$PRESET"
  --steps "$STEPS"
  --micro-batch "$MICRO_BATCH"
  --target-tokens "$TARGET_TOKENS"
  --batch-quantum "$BATCH_QUANTUM"
  --backend "$BACKEND"
  --ddp-bucket-cap-mb "$DDP_BUCKET_CAP_MB"
  --fsdp-output-dtype "$FSDP_OUTPUT_DTYPE"
)
test "$COMPILE_BLOCKS" = 0 || ARGS+=(--compile-blocks)
test "$FUSED_ADAMW" = 0 || ARGS+=(--fused-adamw)
test "$PINNED_STAGING" = 0 || ARGS+=(--pinned-staging)
test "$LENGTH_BUCKETING" = 0 || ARGS+=(--length-bucketing)
test "$SKIP_CHECKPOINT" = 0 || ARGS+=(--skip-checkpoint)
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "$SOURCE_ROOT/scripts/run_antibody_pretraining_optimized.py" "${ARGS[@]}"

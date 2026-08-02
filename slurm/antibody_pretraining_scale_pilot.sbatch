#!/bin/bash
#SBATCH --job-name=abf-scale-pilot
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-scale-pilot-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-scale-pilot-%j.err

set -euo pipefail
umask 077
PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/deepspeed0190_torch213_v3/bin/python}
TORCH_SITE=/ssd/scxi253/model_deployment/envs/frontier_torch213/lib/python3.10/site-packages
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
BACKEND=${ABF_BACKEND:?ABF_BACKEND is required}
PRESET=${ABF_PRESET:-abf_300m}
STEPS=${ABF_STEPS:-3}
MICRO_BATCH=${ABF_MICRO_BATCH:-2}
case "$BACKEND" in ddp|fsdp2|zero2|zero3) ;; *) exit 2 ;; esac
case "$PRESET" in abf_300m|abf_650m) ;; *) exit 2 ;; esac
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID-$PRESET-$BACKEND
test ! -e "$RUN_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SOURCE_ROOT/src:$TORCH_SITE
LOCAL_CACHE=${SLURM_TMPDIR:-/tmp}/abf-scale-pilot-$SLURM_JOB_ID
mkdir -p "$LOCAL_CACHE/triton" "$LOCAL_CACHE/torch_extensions"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DS_BUILD_OPS=0 DS_IGNORE_CUDA_DETECTION=1 TRITON_HOME=$LOCAL_CACHE/triton
export TRITON_CACHE_DIR=$LOCAL_CACHE/triton TORCH_EXTENSIONS_DIR=$LOCAL_CACHE/torch_extensions
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 2 > "$OUTPUT_ROOT/gpu-$SLURM_JOB_ID-$PRESET-$BACKEND.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "$SOURCE_ROOT/scripts/run_antibody_pretraining.py" \
  --shard "$SHARD_ROOT" --manifest-sha256 "$MANIFEST_SHA256" \
  --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json" \
  --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json" \
  --output "$RUN_ROOT" --preset "$PRESET" --steps "$STEPS" \
  --micro-batch "$MICRO_BATCH" --target-tokens 16384 --batch-quantum 128 \
  --backend "$BACKEND" --skip-checkpoint

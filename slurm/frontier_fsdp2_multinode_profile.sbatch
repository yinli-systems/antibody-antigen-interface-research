#!/bin/bash
#SBATCH --job-name=frontier-multinode-profile
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=/tmp/frontier-multinode-profile-%j.out
#SBATCH --error=/tmp/frontier-multinode-profile-%j.err

set -euo pipefail
umask 077

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${FRONTIER_PYTHON_BIN:-$DEPLOY_ROOT/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?FRONTIER_SOURCE_ROOT must identify sealed source}
OUTPUT_ROOT=${FRONTIER_OUTPUT_ROOT:?FRONTIER_OUTPUT_ROOT must be create-only}
SOURCE_SHA256=${FRONTIER_SOURCE_SHA256:?FRONTIER_SOURCE_SHA256 is required}
RUNTIME_MANIFEST=${FRONTIER_RUNTIME_MANIFEST:?FRONTIER_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${FRONTIER_RUNTIME_MANIFEST_SHA256:?FRONTIER_RUNTIME_MANIFEST_SHA256 is required}
PRECISION=${FRONTIER_PRECISION:-bfloat16}
NPROC_PER_NODE=${FRONTIER_NPROC_PER_NODE:-8}
WORLD_SIZE=${FRONTIER_WORLD_SIZE:?FRONTIER_WORLD_SIZE must be 32 or 64}
GLOBAL_BATCH_SIZE=${FRONTIER_GLOBAL_BATCH_SIZE:-64}
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}

if [[ "$WORLD_SIZE" != "32" && "$WORLD_SIZE" != "64" ]]; then
  echo "FRONTIER_WORLD_SIZE must be 32 or 64" >&2
  exit 2
fi
if [[ "$NPROC_PER_NODE" != "8" ]] || (( SLURM_JOB_NUM_NODES * NPROC_PER_NODE != WORLD_SIZE )); then
  echo "Allocated nodes times eight GPUs must equal FRONTIER_WORLD_SIZE" >&2
  exit 2
fi
if [[ "$GLOBAL_BATCH_SIZE" != "64" ]] || (( GLOBAL_BATCH_SIZE % WORLD_SIZE != 0 )); then
  echo "Strong-scaling profile requires fixed global batch 64" >&2
  exit 2
fi
BATCH_SIZE_PER_RANK=$((GLOBAL_BATCH_SIZE / WORLD_SIZE))
for absolute_path in "$PYTHON_BIN" "$SOURCE_ROOT" "$OUTPUT_ROOT" "$RUNTIME_MANIFEST" "$LOCAL_BASE"; do
  if [[ "$absolute_path" != /* ]]; then
    echo "All runtime, source, output, and scratch paths must be absolute" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" || ! -f "$RUNTIME_MANIFEST" ]]; then
  echo "The Python runtime or runtime manifest is unavailable" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite an existing run root: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
mkdir "$RUN_ROOT"

seal_run() {
  if [[ -d "$RUN_ROOT" && ! -e "$RUN_ROOT/artifact_sha256.txt" ]]; then
    find "$RUN_ROOT" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$RUN_ROOT/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "$RUN_ROOT" 2>/dev/null || true
}
trap seal_run EXIT

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src:$SOURCE_ROOT/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

VISIBLE_GPU_COUNT=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
if [[ "$VISIBLE_GPU_COUNT" != "8" ]]; then
  echo "Every allocated node must expose exactly eight GPUs" >&2
  exit 2
fi
OBSERVED_SOURCE_SHA256=$("$PYTHON_BIN" -c "import pathlib,sys; p=pathlib.Path('$SOURCE_ROOT'); sys.path.insert(0,str(p/'scripts')); from frontier_fsdp2_profile import _source_digest; print(_source_digest(p))")
if [[ "$OBSERVED_SOURCE_SHA256" != "$SOURCE_SHA256" ]]; then
  echo "Sealed source digest differs from FRONTIER_SOURCE_SHA256" >&2
  exit 2
fi

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/generate_runtime_manifest.py" verify --manifest "$RUNTIME_MANIFEST" --expected-file-sha256 "$RUNTIME_MANIFEST_SHA256" > "$RUN_ROOT/runtime_verification.json"
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/distributed_launch_contract.py" \
  --output "$RUN_ROOT/launch_manifest.json" \
  --source-sha256 "$SOURCE_SHA256" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --runtime-verification "$RUN_ROOT/runtime_verification.json" \
  --nodes "$SLURM_JOB_NUM_NODES" \
  --gpus-per-node 8 \
  --processes-per-node 8 \
  --world-size "$WORLD_SIZE" \
  --backend nccl \
  --parallelism fsdp2 \
  --precision "$PRECISION" \
  --scope synthetic_fsdp2_multinode_strong_scaling_profile \
  > "$RUN_ROOT/launch_manifest_creation.json"
LAUNCH_MANIFEST_SHA256=$(sha256sum "$RUN_ROOT/launch_manifest.json" | awk '{print $1}')
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --label nvidia-smi -L > "$RUN_ROOT/gpu_inventory.txt"
{
  echo "job_id=$SLURM_JOB_ID"
  echo "nodes=$SLURM_JOB_NUM_NODES"
  echo "node_list=$SLURM_JOB_NODELIST"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "source_root=$SOURCE_ROOT"
  echo "run_root=$RUN_ROOT"
  echo "source_sha256=$SOURCE_SHA256"
  echo "runtime_manifest_sha256=$RUNTIME_MANIFEST_SHA256"
  echo "launch_manifest_sha256=$LAUNCH_MANIFEST_SHA256"
  echo "precision=$PRECISION"
  echo "world_size=$WORLD_SIZE"
  echo "fixed_global_batch_size=$GLOBAL_BATCH_SIZE"
  echo "batch_size_per_rank=$BATCH_SIZE_PER_RANK"
  echo "warmup_steps=50"
  echo "measured_steps=500"
  echo "claim_boundary=reduced synthetic fixed-batch strong-scaling measurement only"
  date --iso-8601=seconds
} > "$RUN_ROOT/environment.txt"

export MASTER_ADDR MASTER_PORT WORLD_SIZE BATCH_SIZE_PER_RANK
export PROFILE_OUTPUT_ROOT=$RUN_ROOT/profile
export PROFILE_LAUNCH_MANIFEST=$RUN_ROOT/launch_manifest.json
export PROFILE_LAUNCH_MANIFEST_SHA256=$LAUNCH_MANIFEST_SHA256
export PROFILE_SOURCE_SHA256=$SOURCE_SHA256
export PROFILE_PRECISION=$PRECISION

srun \
  --nodes="$SLURM_JOB_NUM_NODES" \
  --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 \
  --cpu-bind=none \
  bash -c '
    set -euo pipefail
    export TMPDIR=${SLURM_TMPDIR:-/tmp}
    "$0" -m torch.distributed.run \
      --nnodes="$SLURM_JOB_NUM_NODES" \
      --nproc-per-node=8 \
      --node-rank="$SLURM_PROCID" \
      --rdzv-backend=c10d \
      --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
      "$1/scripts/frontier_fsdp2_profile.py" \
      --output-root "$PROFILE_OUTPUT_ROOT" \
      --warmup-steps 50 \
      --measured-steps 500 \
      --batch-size-per-rank "$BATCH_SIZE_PER_RANK" \
      --seed 20260730 \
      --precision "$PROFILE_PRECISION" \
      --parallelism fsdp2 \
      --expected-world-size "$WORLD_SIZE" \
      --expected-source-sha256 "$PROFILE_SOURCE_SHA256" \
      --launch-manifest "$PROFILE_LAUNCH_MANIFEST" \
      --expected-launch-manifest-sha256 "$PROFILE_LAUNCH_MANIFEST_SHA256" \
      --synthetic-only
  ' "$PYTHON_BIN" "$SOURCE_ROOT" 2>&1 | tee "$RUN_ROOT/profile.log"

date --iso-8601=seconds >> "$RUN_ROOT/environment.txt"
find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"

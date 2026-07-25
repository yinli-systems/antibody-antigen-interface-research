#!/bin/bash
#SBATCH --job-name=frontier-r2-smoke
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/tmp/frontier-r2-smoke-%j.out
#SBATCH --error=/tmp/frontier-r2-smoke-%j.err

set -euo pipefail
umask 077

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${FRONTIER_PYTHON_BIN:-$DEPLOY_ROOT/envs/mipe/bin/python}
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?FRONTIER_SOURCE_ROOT must identify sealed source}
COARSE_ROOT=${FRONTIER_COARSE_ROOT:?FRONTIER_COARSE_ROOT must identify a verified D0 run}
OUTPUT_ROOT=${FRONTIER_OUTPUT_ROOT:?FRONTIER_OUTPUT_ROOT must be create-only}
SOURCE_SHA256=${FRONTIER_SOURCE_SHA256:?FRONTIER_SOURCE_SHA256 is required}
RUNTIME_MANIFEST=${FRONTIER_RUNTIME_MANIFEST:?FRONTIER_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${FRONTIER_RUNTIME_MANIFEST_SHA256:?FRONTIER_RUNTIME_MANIFEST_SHA256 is required}
PRECISION=${FRONTIER_PRECISION:-bfloat16}
NPROC=${FRONTIER_NPROC:-8}
DDP_INITIALIZATION=${FRONTIER_DDP_INITIALIZATION:-default}
PARALLELISM=${FRONTIER_PARALLELISM:-ddp}
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}

for absolute_path in \
  "$PYTHON_BIN" "$SOURCE_ROOT" "$COARSE_ROOT" "$OUTPUT_ROOT" \
  "$RUNTIME_MANIFEST" "$LOCAL_BASE"; do
  if [[ "$absolute_path" != /* ]]; then
    echo "All runtime, source, checkpoint, output, and scratch paths must be absolute" >&2
    exit 2
  fi
done
if [[ "$OUTPUT_ROOT" == "/" || "$SOURCE_ROOT" == "/" ]]; then
  echo "Source and output roots cannot be the filesystem root" >&2
  exit 2
fi
case "$OUTPUT_ROOT/" in
  "$SOURCE_ROOT/"*)
    echo "Output root cannot be inside the sealed source tree" >&2
    exit 2
    ;;
esac
if [[ "$PARALLELISM" != "ddp" ]]; then
  echo "This diagnostic implements DDP only; FSDP2 is not implied" >&2
  exit 2
fi
if [[ "$PRECISION" != "float32" && "$PRECISION" != "bfloat16" ]]; then
  echo "FRONTIER_PRECISION must be float32 or bfloat16" >&2
  exit 2
fi
if [[ "$DDP_INITIALIZATION" != "default" \
  && "$DDP_INITIALIZATION" != "verified-equal" ]]; then
  echo "FRONTIER_DDP_INITIALIZATION is unsupported" >&2
  exit 2
fi
if [[ ! "$NPROC" =~ ^[1-8]$ ]]; then
  echo "FRONTIER_NPROC must be an integer from one through eight" >&2
  exit 2
fi
if [[ ! "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$RUNTIME_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Source and runtime anchors must be lowercase SHA-256 digests" >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-}" != "1" ]]; then
  echo "This standalone torchrun launcher requires exactly one Slurm node" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$RUNTIME_MANIFEST" ]]; then
  echo "The Python runtime or runtime manifest is unavailable" >&2
  exit 2
fi
for required_source in \
  scripts/distributed_launch_contract.py \
  scripts/frontier_ddp_smoke.py \
  scripts/frontier_ddp_staged_smoke.py \
  scripts/generate_runtime_manifest.py; do
  if [[ ! -f "$SOURCE_ROOT/$required_source" ]]; then
    echo "Sealed source is missing $required_source" >&2
    exit 2
  fi
done
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite an existing run root: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
mkdir "$RUN_ROOT"
LOCAL_ROOT=$(mktemp -d "$LOCAL_BASE/frontier-r2-smoke-$SLURM_JOB_ID.XXXXXX")
mkdir "$LOCAL_ROOT/tmp"
MONITOR_PID=
stop_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=
  fi
}
seal_run() {
  if [[ -d "$RUN_ROOT" && ! -e "$RUN_ROOT/artifact_sha256.txt" ]]; then
    find "$RUN_ROOT" -type f -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum \
      > "$RUN_ROOT/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "$RUN_ROOT" 2>/dev/null || true
}
cleanup() {
  local status=$?
  stop_monitor
  seal_run
  chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true
  rm -rf -- "$LOCAL_ROOT"
  return "$status"
}
trap cleanup EXIT

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src
export TMPDIR=$LOCAL_ROOT/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

VISIBLE_GPU_COUNT=$(
  "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())'
)
if [[ ! "$VISIBLE_GPU_COUNT" =~ ^[1-8]$ ]] \
  || (( NPROC != VISIBLE_GPU_COUNT )); then
  echo "NPROC must equal the allocated visible GPU count in [1, 8]" >&2
  exit 2
fi

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/generate_runtime_manifest.py" verify \
  --manifest "$RUNTIME_MANIFEST" \
  --expected-file-sha256 "$RUNTIME_MANIFEST_SHA256" \
  > "$RUN_ROOT/runtime_verification.json"
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/distributed_launch_contract.py" \
  --output "$RUN_ROOT/launch_manifest.json" \
  --source-sha256 "$SOURCE_SHA256" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --runtime-verification "$RUN_ROOT/runtime_verification.json" \
  --nodes 1 \
  --gpus-per-node "$VISIBLE_GPU_COUNT" \
  --processes-per-node "$NPROC" \
  --world-size "$NPROC" \
  --backend nccl \
  --parallelism "$PARALLELISM" \
  --precision "$PRECISION" \
  --scope synthetic_r2_smoke \
  > "$RUN_ROOT/launch_manifest_creation.json"
LAUNCH_MANIFEST_SHA256=$(
  sha256sum "$RUN_ROOT/launch_manifest.json" | awk '{print $1}'
)

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 2 > "$RUN_ROOT/gpu.csv" &
MONITOR_PID=$!

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "source_root=$SOURCE_ROOT"
  echo "coarse_root=$COARSE_ROOT"
  echo "run_root=$RUN_ROOT"
  echo "ddp_initialization=$DDP_INITIALIZATION"
  echo "source_sha256=$SOURCE_SHA256"
  echo "runtime_manifest_sha256=$RUNTIME_MANIFEST_SHA256"
  echo "launch_manifest_sha256=$LAUNCH_MANIFEST_SHA256"
  echo "parallelism=$PARALLELISM"
  echo "precision=$PRECISION"
  echo "world_size=$NPROC"
  echo "visible_gpu_count=$VISIBLE_GPU_COUNT"
  echo "nccl_debug=${NCCL_DEBUG:-<unset>}"
  echo "nccl_p2p_disable=${NCCL_P2P_DISABLE:-<unset>}"
  echo "nccl_ib_disable=${NCCL_IB_DISABLE:-<unset>}"
  echo "nccl_shm_disable=${NCCL_SHM_DISABLE:-<unset>}"
  echo "nccl_p2p_level=${NCCL_P2P_LEVEL:-<unset>}"
  echo "nccl_cumem_host_enable=${NCCL_CUMEM_HOST_ENABLE:-<unset>}"
  echo "claim_boundary=synthetic correctness only; no efficiency or model metric claim"
  date --iso-8601=seconds
  nvidia-smi
} > "$RUN_ROOT/environment.txt"

DDP_ARGUMENTS=()
if [[ "$DDP_INITIALIZATION" != "default" ]]; then
  DDP_ARGUMENTS=(--ddp-initialization "$DDP_INITIALIZATION")
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NPROC" \
  "$SOURCE_ROOT/scripts/frontier_ddp_staged_smoke.py" \
  --stage r2 \
  --output-root "$RUN_ROOT/r2" \
  --coarse-root "$COARSE_ROOT" \
  --steps 3 \
  --batch-size-per-rank 3 \
  --seed 20260724 \
  --precision "$PRECISION" \
  --parallelism "$PARALLELISM" \
  --expected-world-size "$NPROC" \
  --expected-source-sha256 "$SOURCE_SHA256" \
  --launch-manifest "$RUN_ROOT/launch_manifest.json" \
  --expected-launch-manifest-sha256 "$LAUNCH_MANIFEST_SHA256" \
  --require-cuda \
  "${DDP_ARGUMENTS[@]}" \
  --synthetic-only \
  2>&1 | tee "$RUN_ROOT/r2.log"

date --iso-8601=seconds >> "$RUN_ROOT/environment.txt"
stop_monitor
find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 -r sha256sum \
  > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"

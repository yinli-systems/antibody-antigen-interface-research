#!/bin/bash
#SBATCH --job-name=frontier-nccl-backward
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/tmp/frontier-nccl-backward-%j.out
#SBATCH --error=/tmp/frontier-nccl-backward-%j.err

set -euo pipefail
umask 077

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${FRONTIER_PYTHON_BIN:-$DEPLOY_ROOT/envs/mipe/bin/python}
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?FRONTIER_SOURCE_ROOT is required}
OUTPUT_ROOT=${FRONTIER_OUTPUT_ROOT:?FRONTIER_OUTPUT_ROOT is required}
MODE=${FRONTIER_NCCL_DIAG_MODE:?FRONTIER_NCCL_DIAG_MODE is required}
SOURCE_SHA256=${FRONTIER_SOURCE_SHA256:?FRONTIER_SOURCE_SHA256 is required}
RUNTIME_MANIFEST=${FRONTIER_RUNTIME_MANIFEST:?FRONTIER_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${FRONTIER_RUNTIME_MANIFEST_SHA256:?FRONTIER_RUNTIME_MANIFEST_SHA256 is required}
NPROC=${FRONTIER_NPROC:-2}
RUN_ROOT=$OUTPUT_ROOT/$MODE/slurm-$SLURM_JOB_ID

for absolute_path in \
  "$PYTHON_BIN" "$SOURCE_ROOT" "$OUTPUT_ROOT" "$RUNTIME_MANIFEST"; do
  if [[ "$absolute_path" != /* ]]; then
    echo "All runtime, source, and output paths must be absolute" >&2
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
case "$MODE" in
  ddp_only|autograd_all_gather_only)
    ;;
  *)
    echo "Unsupported FRONTIER_NCCL_DIAG_MODE: $MODE" >&2
    exit 2
    ;;
esac
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite diagnostic run root: $RUN_ROOT" >&2
  exit 2
fi
if [[ ! "$NPROC" =~ ^[2-8]$ ]]; then
  echo "FRONTIER_NPROC must be an integer from two through eight" >&2
  exit 2
fi
if [[ ! "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$RUNTIME_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Source and runtime anchors must be lowercase SHA-256 digests" >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-}" != "1" ]]; then
  echo "This standalone diagnostic requires exactly one Slurm node" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$RUNTIME_MANIFEST" ]]; then
  echo "The Python runtime or runtime manifest is unavailable" >&2
  exit 2
fi
for required_source in \
  scripts/distributed_launch_contract.py \
  scripts/generate_runtime_manifest.py \
  scripts/nccl_backward_isolation.py; do
  if [[ ! -f "$SOURCE_ROOT/$required_source" ]]; then
    echo "Sealed source is missing $required_source" >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_ROOT/$MODE"
mkdir "$RUN_ROOT"
seal_run() {
  local status=$?
  if [[ ! -e "$RUN_ROOT/artifact_sha256.txt" ]]; then
    find "$RUN_ROOT" -type f -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum \
      > "$RUN_ROOT/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "$RUN_ROOT" 2>/dev/null || true
  return "$status"
}
trap seal_run EXIT

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src
export OMP_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
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
  --parallelism collective_diagnostic \
  --precision float32 \
  --scope synthetic_nccl_backward \
  > "$RUN_ROOT/launch_manifest_creation.json"
LAUNCH_MANIFEST_SHA256=$(
  sha256sum "$RUN_ROOT/launch_manifest.json" | awk '{print $1}'
)

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  "--nproc-per-node=$NPROC" \
  "$SOURCE_ROOT/scripts/nccl_backward_isolation.py" \
  --output-root "$RUN_ROOT" \
  --mode "$MODE" \
  --expected-world-size "$NPROC" \
  --expected-source-sha256 "$SOURCE_SHA256" \
  --launch-manifest "$RUN_ROOT/launch_manifest.json" \
  --expected-launch-manifest-sha256 "$LAUNCH_MANIFEST_SHA256" \
  --synthetic-only \
  2>&1 | tee "$RUN_ROOT/diagnostic.log"

find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 -r sha256sum \
  > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"

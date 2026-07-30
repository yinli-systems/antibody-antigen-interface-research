#!/bin/bash
#SBATCH --job-name=frontier-reshard
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --output=/tmp/frontier-reshard-%j.out
#SBATCH --error=/tmp/frontier-reshard-%j.err

set -euo pipefail
umask 077

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${FRONTIER_PYTHON_BIN:-$DEPLOY_ROOT/envs/mipe/bin/python}
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?FRONTIER_SOURCE_ROOT must identify sealed source}
OUTPUT_ROOT=${FRONTIER_OUTPUT_ROOT:?FRONTIER_OUTPUT_ROOT must be create-only}
SOURCE_SHA256=${FRONTIER_SOURCE_SHA256:?FRONTIER_SOURCE_SHA256 is required}
RUNTIME_MANIFEST=${FRONTIER_RUNTIME_MANIFEST:?FRONTIER_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${FRONTIER_RUNTIME_MANIFEST_SHA256:?FRONTIER_RUNTIME_MANIFEST_SHA256 is required}
MODE=${FRONTIER_RESHARD_MODE:?FRONTIER_RESHARD_MODE must be save or load}
PRECISION=${FRONTIER_PRECISION:-float32}
NPROC=${FRONTIER_NPROC:-8}
ABSOLUTE_TOLERANCE=${FRONTIER_ABSOLUTE_TOLERANCE:-2e-5}
RELATIVE_TOLERANCE=${FRONTIER_RELATIVE_TOLERANCE:-2e-4}
SOURCE_SAVE_RESULT=${FRONTIER_SOURCE_SAVE_RESULT:-}
SOURCE_SAVE_RESULT_SHA256=${FRONTIER_SOURCE_SAVE_RESULT_SHA256:-}
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}

if [[ "$MODE" != "save" && "$MODE" != "load" ]]; then
  echo "FRONTIER_RESHARD_MODE must be save or load" >&2
  exit 2
fi
if [[ ! "$NPROC" =~ ^[1-8]$ ]] || (( 8 % NPROC != 0 )); then
  echo "FRONTIER_NPROC must be one, two, four, or eight" >&2
  exit 2
fi
if [[ "$MODE" == "load" ]]; then
  if [[ "$SOURCE_SAVE_RESULT" != /* ]] \
    || [[ ! "$SOURCE_SAVE_RESULT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Load mode requires an absolute save result and its SHA-256" >&2
    exit 2
  fi
fi
for absolute_path in "$PYTHON_BIN" "$SOURCE_ROOT" "$OUTPUT_ROOT" "$RUNTIME_MANIFEST" "$LOCAL_BASE"; do
  if [[ "$absolute_path" != /* ]]; then
    echo "All runtime, source, output, and scratch paths must be absolute" >&2
    exit 2
  fi
done
if [[ "$OUTPUT_ROOT" == "/" || "$SOURCE_ROOT" == "/" ]]; then
  echo "Source and output roots cannot be the filesystem root" >&2
  exit 2
fi
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
LOCAL_ROOT=$(mktemp -d "$LOCAL_BASE/frontier-reshard-$SLURM_JOB_ID.XXXXXX")
mkdir "$LOCAL_ROOT/tmp"

seal_run() {
  if [[ -d "$RUN_ROOT" && ! -e "$RUN_ROOT/artifact_sha256.txt" ]]; then
    find "$RUN_ROOT" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$RUN_ROOT/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "$RUN_ROOT" 2>/dev/null || true
}

cleanup() {
  local status=$?
  seal_run
  chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true
  rm -rf -- "$LOCAL_ROOT"
  return "$status"
}
trap cleanup EXIT

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src:$SOURCE_ROOT/scripts
export TMPDIR=$LOCAL_ROOT/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

VISIBLE_GPU_COUNT=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
if [[ ! "$VISIBLE_GPU_COUNT" =~ ^[1-8]$ ]] || (( NPROC != VISIBLE_GPU_COUNT )); then
  echo "NPROC must equal the allocated visible GPU count" >&2
  exit 2
fi
OBSERVED_SOURCE_SHA256=$("$PYTHON_BIN" -c "import pathlib,sys; p=pathlib.Path('$SOURCE_ROOT'); sys.path.insert(0,str(p/'scripts')); from frontier_fsdp2_reshard import _source_digest; print(_source_digest(p))")
if [[ "$OBSERVED_SOURCE_SHA256" != "$SOURCE_SHA256" ]]; then
  echo "Sealed source digest differs from FRONTIER_SOURCE_SHA256" >&2
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
  --parallelism fsdp2 \
  --precision "$PRECISION" \
  --scope synthetic_fsdp2_cross_world_reshard \
  > "$RUN_ROOT/launch_manifest_creation.json"
LAUNCH_MANIFEST_SHA256=$(sha256sum "$RUN_ROOT/launch_manifest.json" | awk '{print $1}')

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "mode=$MODE"
  echo "source_root=$SOURCE_ROOT"
  echo "run_root=$RUN_ROOT"
  echo "source_sha256=$SOURCE_SHA256"
  echo "runtime_manifest_sha256=$RUNTIME_MANIFEST_SHA256"
  echo "launch_manifest_sha256=$LAUNCH_MANIFEST_SHA256"
  echo "precision=$PRECISION"
  echo "world_size=$NPROC"
  echo "absolute_tolerance=$ABSOLUTE_TOLERANCE"
  echo "relative_tolerance=$RELATIVE_TOLERANCE"
  echo "source_save_result=$SOURCE_SAVE_RESULT"
  echo "source_save_result_sha256=$SOURCE_SAVE_RESULT_SHA256"
  echo "claim_boundary=synthetic cross-world FSDP2 DCP correctness only"
  date --iso-8601=seconds
  nvidia-smi
  nvidia-smi topo -m
} > "$RUN_ROOT/environment.txt"

LOAD_ARGUMENTS=()
if [[ "$MODE" == "load" ]]; then
  LOAD_ARGUMENTS+=(
    --source-save-result "$SOURCE_SAVE_RESULT"
    --expected-save-result-sha256 "$SOURCE_SAVE_RESULT_SHA256"
  )
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NPROC" \
  "$SOURCE_ROOT/scripts/frontier_fsdp2_reshard.py" \
  --mode "$MODE" \
  --output-root "$RUN_ROOT/reshard" \
  --steps-before-checkpoint 3 \
  --precision "$PRECISION" \
  --parallelism fsdp2 \
  --expected-world-size "$NPROC" \
  --expected-source-sha256 "$SOURCE_SHA256" \
  --launch-manifest "$RUN_ROOT/launch_manifest.json" \
  --expected-launch-manifest-sha256 "$LAUNCH_MANIFEST_SHA256" \
  --absolute-tolerance "$ABSOLUTE_TOLERANCE" \
  --relative-tolerance "$RELATIVE_TOLERANCE" \
  --synthetic-only \
  "${LOAD_ARGUMENTS[@]}" \
  2>&1 | tee "$RUN_ROOT/reshard.log"

date --iso-8601=seconds >> "$RUN_ROOT/environment.txt"
find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"

#!/bin/bash
#SBATCH --job-name=frontier-mn-gate
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/tmp/frontier-mn-gate-%j.out
#SBATCH --error=/tmp/frontier-mn-gate-%j.err

set -euo pipefail
umask 077

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${FRONTIER_PYTHON_BIN:-$DEPLOY_ROOT/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?FRONTIER_SOURCE_ROOT must identify sealed source}
OUTPUT_ROOT=${FRONTIER_OUTPUT_ROOT:?FRONTIER_OUTPUT_ROOT must be create-only}
SOURCE_SHA256=${FRONTIER_SOURCE_SHA256:?FRONTIER_SOURCE_SHA256 is required}
RUNTIME_MANIFEST=${FRONTIER_RUNTIME_MANIFEST:?FRONTIER_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${FRONTIER_RUNTIME_MANIFEST_SHA256:?FRONTIER_RUNTIME_MANIFEST_SHA256 is required}
MODE=${FRONTIER_GATE_MODE:?FRONTIER_GATE_MODE is required}
PRECISION=${FRONTIER_PRECISION:-bfloat16}
SOURCE_SAVE_RESULT=${FRONTIER_SOURCE_SAVE_RESULT:-}
SOURCE_SAVE_RESULT_SHA256=${FRONTIER_SOURCE_SAVE_RESULT_SHA256:-}
WORLD_SIZE=8
NPROC_PER_NODE=2
GLOBAL_BATCH_SIZE=64
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID

if [[ "$PRECISION" == "bfloat16" ]]; then
  ABSOLUTE_TOLERANCE=0.005
  RELATIVE_TOLERANCE=0.02
else
  ABSOLUTE_TOLERANCE=2e-5
  RELATIVE_TOLERANCE=2e-4
fi

if [[ ! "$MODE" =~ ^(dcp_smoke|reshard_save|reshard_load|profile)$ ]]; then
  echo "FRONTIER_GATE_MODE is outside the reviewed set" >&2
  exit 2
fi
if [[ "$PRECISION" != "float32" && "$PRECISION" != "bfloat16" ]]; then
  echo "FRONTIER_PRECISION must be float32 or bfloat16" >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-}" != "4" ]]; then
  echo "This gate requires exactly four Slurm nodes" >&2
  exit 2
fi
for absolute_path in "$PYTHON_BIN" "$SOURCE_ROOT" "$OUTPUT_ROOT" "$RUNTIME_MANIFEST"; do
  if [[ "$absolute_path" != /* ]]; then
    echo "All source, runtime, and output paths must be absolute" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" || ! -f "$RUNTIME_MANIFEST" ]]; then
  echo "The Python runtime or runtime manifest is unavailable" >&2
  exit 2
fi
if [[ ! "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$RUNTIME_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Source and runtime anchors must be lowercase SHA-256 digests" >&2
  exit 2
fi
if [[ "$MODE" == "reshard_load" ]] \
  && { [[ "$SOURCE_SAVE_RESULT" != /* ]] \
    || [[ ! "$SOURCE_SAVE_RESULT_SHA256" =~ ^[0-9a-f]{64}$ ]]; }; then
  echo "Reshard load requires an absolute save receipt and SHA-256" >&2
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
    find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum \
      > "$RUN_ROOT/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "$RUN_ROOT" 2>/dev/null || true
}
trap seal_run EXIT

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src:$SOURCE_ROOT/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=${FRONTIER_NCCL_SOCKET_IFNAME:-=bond0}
export NCCL_IB_DISABLE=${FRONTIER_NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${FRONTIER_NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${FRONTIER_NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}

case "$MODE" in
  dcp_smoke)
    ENTRYPOINT=scripts/frontier_fsdp2_dcp_smoke.py
    SOURCE_MODULE=frontier_fsdp2_dcp_smoke
    SOURCE_FUNCTION=_source_digest
    SCOPE=synthetic_fsdp2_multinode_dcp_correctness
    RUN_ARGUMENTS=(
      --output-root "$RUN_ROOT/fsdp2"
      --steps-before-checkpoint 3
      --batch-size-per-rank 2
      --seed 20260730
      --precision "$PRECISION"
      --parallelism fsdp2
      --expected-world-size "$WORLD_SIZE"
      --expected-source-sha256 "$SOURCE_SHA256"
      --absolute-tolerance "$ABSOLUTE_TOLERANCE"
      --relative-tolerance "$RELATIVE_TOLERANCE"
      --synthetic-only
    )
    ;;
  reshard_save|reshard_load)
    ENTRYPOINT=scripts/frontier_fsdp2_reshard.py
    SOURCE_MODULE=frontier_fsdp2_reshard
    SOURCE_FUNCTION=_source_digest
    SCOPE=synthetic_fsdp2_multinode_cross_world_reshard
    RESHARD_MODE=${MODE#reshard_}
    RUN_ARGUMENTS=(
      --mode "$RESHARD_MODE"
      --output-root "$RUN_ROOT/reshard"
      --steps-before-checkpoint 3
      --precision "$PRECISION"
      --parallelism fsdp2
      --expected-world-size "$WORLD_SIZE"
      --expected-source-sha256 "$SOURCE_SHA256"
      --absolute-tolerance "$ABSOLUTE_TOLERANCE"
      --relative-tolerance "$RELATIVE_TOLERANCE"
      --synthetic-only
    )
    if [[ "$MODE" == "reshard_load" ]]; then
      RUN_ARGUMENTS+=(
        --source-save-result "$SOURCE_SAVE_RESULT"
        --expected-save-result-sha256 "$SOURCE_SAVE_RESULT_SHA256"
      )
    fi
    ;;
  profile)
    ENTRYPOINT=scripts/frontier_fsdp2_profile.py
    SOURCE_MODULE=frontier_fsdp2_profile
    SOURCE_FUNCTION=_source_digest
    SCOPE=synthetic_fsdp2_multinode_world8_profile
    RUN_ARGUMENTS=(
      --output-root "$RUN_ROOT/profile"
      --warmup-steps 50
      --measured-steps 500
      --batch-size-per-rank 8
      --seed 20260730
      --precision "$PRECISION"
      --parallelism fsdp2
      --expected-world-size "$WORLD_SIZE"
      --expected-source-sha256 "$SOURCE_SHA256"
      --synthetic-only
    )
    ;;
esac

OBSERVED_SOURCE_SHA256=$(
  "$PYTHON_BIN" -c \
    "import pathlib,sys; p=pathlib.Path('$SOURCE_ROOT'); sys.path.insert(0,str(p/'scripts')); from $SOURCE_MODULE import $SOURCE_FUNCTION; print($SOURCE_FUNCTION(p))"
)
if [[ "$OBSERVED_SOURCE_SHA256" != "$SOURCE_SHA256" ]]; then
  echo "Sealed source digest differs from FRONTIER_SOURCE_SHA256" >&2
  exit 2
fi

export FRONTIER_VERIFY_ROOT=$RUN_ROOT/runtime_by_node
mkdir "$FRONTIER_VERIFY_ROOT"
export FRONTIER_PYTHON_BIN=$PYTHON_BIN
export FRONTIER_SOURCE_ROOT=$SOURCE_ROOT
export FRONTIER_RUNTIME_MANIFEST=$RUNTIME_MANIFEST
export FRONTIER_RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256
RUNTIME_VERIFY_STATUS=116
RUNTIME_VERIFY_SUCCESS_ROOT=
for launch_attempt in 1 2 3; do
  export FRONTIER_VERIFY_ATTEMPT_ROOT=$FRONTIER_VERIFY_ROOT/launch-$launch_attempt
  mkdir "$FRONTIER_VERIFY_ATTEMPT_ROOT"
  set +e
  srun \
    --overlap \
    --nodes=4 \
    --ntasks=4 \
    --ntasks-per-node=1 \
    --cpu-bind=none \
    bash -c '
      set -euo pipefail
      node=$(hostname)
      receipt="$FRONTIER_VERIFY_ATTEMPT_ROOT/runtime-$SLURM_PROCID-$node.json"
      status=116
      for attempt in 1 2 3; do
        payload="$FRONTIER_VERIFY_ATTEMPT_ROOT/runtime-$SLURM_PROCID-$node.attempt-$attempt.payload"
        stderr="$FRONTIER_VERIFY_ATTEMPT_ROOT/runtime-$SLURM_PROCID-$node.attempt-$attempt.stderr"
        attempt_status="$FRONTIER_VERIFY_ATTEMPT_ROOT/runtime-$SLURM_PROCID-$node.attempt-$attempt.status"
        set +e
        "$FRONTIER_PYTHON_BIN" \
          "$FRONTIER_SOURCE_ROOT/scripts/generate_runtime_manifest.py" verify \
          --manifest "$FRONTIER_RUNTIME_MANIFEST" \
          --expected-file-sha256 "$FRONTIER_RUNTIME_MANIFEST_SHA256" \
          > "$payload" 2> "$stderr"
        status=$?
        set -e
        printf "attempt=%s\nexit_code=%s\n" "$attempt" "$status" > "$attempt_status"
        if [[ "$status" == "0" ]]; then
          ln "$payload" "$receipt"
          exit 0
        fi
        if [[ "$status" != "116" ]]; then
          exit "$status"
        fi
        sleep "$((attempt * 2))"
      done
      exit "$status"
    '
  RUNTIME_VERIFY_STATUS=$?
  set -e
  printf "launch_attempt=%s\nexit_code=%s\n" \
    "$launch_attempt" "$RUNTIME_VERIFY_STATUS" \
    > "$FRONTIER_VERIFY_ROOT/launch-$launch_attempt.status"
  if [[ "$RUNTIME_VERIFY_STATUS" == "0" ]]; then
    RUNTIME_VERIFY_SUCCESS_ROOT=$FRONTIER_VERIFY_ATTEMPT_ROOT
    break
  fi
  if [[ "$RUNTIME_VERIFY_STATUS" != "116" ]]; then
    exit "$RUNTIME_VERIFY_STATUS"
  fi
  sleep "$((launch_attempt * 2))"
done
if [[ "$RUNTIME_VERIFY_STATUS" != "0" || -z "$RUNTIME_VERIFY_SUCCESS_ROOT" ]]; then
  exit "$RUNTIME_VERIFY_STATUS"
fi

"$PYTHON_BIN" - "$RUNTIME_VERIFY_SUCCESS_ROOT" "$RUNTIME_MANIFEST_SHA256" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
files = sorted(root.glob("runtime-*.json"))
if len(files) != 4:
    raise SystemExit("Runtime verification did not cover all four nodes")
states = set()
for path in files:
    payload = json.loads(path.read_text(encoding="ascii"))
    if payload.get("valid") is not True or payload.get("file_sha256") != expected:
        raise SystemExit(f"Runtime verification failed: {path}")
    states.add(
        (
            payload.get("content_sha256"),
            payload.get("runtime_state_sha256"),
        )
    )
if len(states) != 1:
    raise SystemExit("Runtime state differs across allocated nodes")
PY

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/distributed_launch_contract.py" \
  --output "$RUN_ROOT/launch_manifest.json" \
  --source-sha256 "$SOURCE_SHA256" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --runtime-verification "$(find "$RUNTIME_VERIFY_SUCCESS_ROOT" -name 'runtime-*.json' -type f | LC_ALL=C sort | head -n 1)" \
  --nodes 4 \
  --gpus-per-node 2 \
  --processes-per-node 2 \
  --world-size "$WORLD_SIZE" \
  --backend nccl \
  --parallelism fsdp2 \
  --precision "$PRECISION" \
  --scope "$SCOPE" \
  > "$RUN_ROOT/launch_manifest_creation.json"
LAUNCH_MANIFEST_SHA256=$(sha256sum "$RUN_ROOT/launch_manifest.json" | awk '{print $1}')
RUN_ARGUMENTS+=(
  --launch-manifest "$RUN_ROOT/launch_manifest.json"
  --expected-launch-manifest-sha256 "$LAUNCH_MANIFEST_SHA256"
)
printf '%s\0' "${RUN_ARGUMENTS[@]}" > "$RUN_ROOT/arguments.bin"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
srun --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 --label nvidia-smi -L \
  > "$RUN_ROOT/gpu_inventory.txt"
{
  echo "job_id=$SLURM_JOB_ID"
  echo "nodes=$SLURM_JOB_NUM_NODES"
  echo "node_list=$SLURM_JOB_NODELIST"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "mode=$MODE"
  echo "precision=$PRECISION"
  echo "world_size=$WORLD_SIZE"
  echo "gpus_per_node=$NPROC_PER_NODE"
  echo "absolute_tolerance=$ABSOLUTE_TOLERANCE"
  echo "relative_tolerance=$RELATIVE_TOLERANCE"
  echo "source_sha256=$SOURCE_SHA256"
  echo "runtime_manifest_sha256=$RUNTIME_MANIFEST_SHA256"
  echo "launch_manifest_sha256=$LAUNCH_MANIFEST_SHA256"
  echo "nccl_socket_ifname=$NCCL_SOCKET_IFNAME"
  echo "nccl_ib_disable=$NCCL_IB_DISABLE"
  echo "nccl_debug=$NCCL_DEBUG"
  echo "nccl_debug_subsys=$NCCL_DEBUG_SUBSYS"
  echo "claim_boundary=synthetic multi-node correctness or profiling only; no biological metric or scaling claim"
  date --iso-8601=seconds
} > "$RUN_ROOT/environment.txt"

export MASTER_ADDR MASTER_PORT WORLD_SIZE
export FRONTIER_ENTRYPOINT=$ENTRYPOINT
export FRONTIER_ARGUMENTS_FILE=$RUN_ROOT/arguments.bin
export FRONTIER_RUN_LOG_ROOT=$RUN_ROOT/node_logs
mkdir "$FRONTIER_RUN_LOG_ROOT"
srun \
  --overlap \
  --nodes=4 \
  --ntasks=4 \
  --ntasks-per-node=1 \
  --cpu-bind=none \
  bash -c '
    set -euo pipefail
    mapfile -d "" -t run_arguments < "$FRONTIER_ARGUMENTS_FILE"
    local_root=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/frontier-mn-$SLURM_JOB_ID-$SLURM_PROCID.XXXXXX")
    cleanup() {
      chmod -R u+w "$local_root" 2>/dev/null || true
      rm -rf -- "$local_root"
    }
    trap cleanup EXIT
    export TMPDIR=$local_root
    "$FRONTIER_PYTHON_BIN" -m torch.distributed.run \
      --nnodes="$SLURM_JOB_NUM_NODES" \
      --nproc-per-node=2 \
      --node-rank="$SLURM_PROCID" \
      --rdzv-backend=c10d \
      --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
      "$FRONTIER_SOURCE_ROOT/$FRONTIER_ENTRYPOINT" \
      "${run_arguments[@]}" \
      > "$FRONTIER_RUN_LOG_ROOT/node-$SLURM_PROCID.log" 2>&1
  '

date --iso-8601=seconds >> "$RUN_ROOT/environment.txt"
find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 -r sha256sum \
  > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"

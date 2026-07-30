#!/bin/bash
#SBATCH --job-name=abf-ddp-parity-v2
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/abf-ddp-parity-v2-%j.out
#SBATCH --error=/tmp/abf-ddp-parity-v2-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
EXECUTION_SHA256=${ABF_EXECUTION_SHA256:?ABF_EXECUTION_SHA256 is required}
EXECUTION_PROTOCOL=$SOURCE_ROOT/protocols/antibody_ddp_scientific_parity_execution_v2_1.json
GATE_PROTOCOL=$SOURCE_ROOT/protocols/antibody_ddp_scientific_parity_gate_v2.json
SPLIT_ROOT=$SOURCE_ROOT/results/pretraining/antibody_ddp_parity_split_v2
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}

for absolute_path in "$PYTHON_BIN" "$SOURCE_ROOT" "$OUTPUT_ROOT" "$SHARD_ROOT" "$LOCAL_BASE"; do
  if [[ "$absolute_path" != /* ]]; then
    echo "All runtime, source, output, data, and scratch paths must be absolute" >&2
    exit 2
  fi
done
if [[ "$SOURCE_ROOT" == "/" || "$OUTPUT_ROOT" == "/" || "$SHARD_ROOT" == "/" ]]; then
  echo "Source, output, and shard roots cannot be the filesystem root" >&2
  exit 2
fi
if [[ ! "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$EXECUTION_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Manifest and execution anchors must be lowercase SHA-256 digests" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -d "$SHARD_ROOT" || ! -f "$EXECUTION_PROTOCOL" ]]; then
  echo "The runtime, dense shard, or execution protocol is unavailable" >&2
  exit 2
fi
if [[ "$(sha256sum "$EXECUTION_PROTOCOL" | awk '{print $1}')" != "$EXECUTION_SHA256" ]]; then
  echo "The execution protocol differs from its external anchor" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite an existing run root" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
mkdir "$RUN_ROOT"

MONITOR_PID=
stop_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=
  fi
}
seal_run() {
  stop_monitor
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
export NCCL_DEBUG=WARN
export ABF_TELEMETRY_INTERVAL_SECONDS=5

VISIBLE_GPU_COUNT=$(
  "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())'
)
if [[ "$VISIBLE_GPU_COUNT" != "8" ]]; then
  echo "The scientific parity allocation must expose exactly eight GPUs" >&2
  exit 2
fi
RTX_5090_COUNT=$(
  nvidia-smi --query-gpu=name --format=csv,noheader \
    | awk '$0 == "NVIDIA GeForce RTX 5090" { count += 1 } END { print count + 0 }'
)
if [[ "$RTX_5090_COUNT" != "8" ]]; then
  echo "The scientific parity allocation requires eight RTX 5090 GPUs" >&2
  exit 2
fi

"$PYTHON_BIN" - "$EXECUTION_PROTOCOL" "$SOURCE_ROOT" "$MANIFEST_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys

execution_path = pathlib.Path(sys.argv[1])
source_root = pathlib.Path(sys.argv[2])
manifest_sha256 = sys.argv[3]
payload = json.loads(execution_path.read_text(encoding="ascii"))
if payload.get("format_version") != "antibody_ddp_scientific_parity_execution_v2_1":
    raise SystemExit("Unexpected execution protocol")
if payload.get("inputs", {}).get("dense_manifest_payload_sha256") != manifest_sha256:
    raise SystemExit("Dense manifest differs from the execution protocol")
paths = {
    "gate_protocol_sha256": "protocols/antibody_ddp_scientific_parity_gate_v2.json",
    "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
    "communication_protocol_sha256": "protocols/antibody_ddp_communication_optimization_v1.json",
    "model_family_sha256": "protocols/antibody_model_family_v1.json",
    "training_protocol_sha256": "protocols/antibody_training_v1.json",
}
for key, relative in paths.items():
    observed = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    if observed != payload["inputs"][key]:
        raise SystemExit(f"Execution input differs: {relative}")
runner = source_root / "scripts/run_antibody_pretraining_optimized.py"
if hashlib.sha256(runner.read_bytes()).hexdigest() != payload["implementation"][runner.name]:
    raise SystemExit("Optimized runner differs from the execution protocol")
split_root = source_root / "results/pretraining/antibody_ddp_parity_split_v2"
for name, expected in payload["inputs"]["split_files"].items():
    if hashlib.sha256((split_root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Parity split differs: {name}")
if hashlib.sha256((split_root / "receipt.json").read_bytes()).hexdigest() != payload["inputs"]["split_receipt_file_sha256"]:
    raise SystemExit("Parity split receipt differs")
PY

cp "$EXECUTION_PROTOCOL" "$RUN_ROOT/execution_protocol.json"
{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "partition=${SLURM_JOB_PARTITION:-}"
  echo "source_root=$SOURCE_ROOT"
  echo "output_root=$OUTPUT_ROOT"
  echo "shard_root=$SHARD_ROOT"
  echo "manifest_sha256=$MANIFEST_SHA256"
  echo "execution_sha256=$EXECUTION_SHA256"
  echo "world_size=8"
  echo "run_count=9"
  echo "updates_per_run=500"
  echo "claim_boundary=training-excluded dense-pilot MLM systems parity only"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
  nvidia-smi topo -m
} > "$RUN_ROOT/environment.txt"

run_mode() {
  local label=$1
  local seed=$2
  local mode=$3
  local mode_root=$RUN_ROOT/$label
  local cache_root=$LOCAL_BASE/abf-ddp-parity-$SLURM_JOB_ID-$label
  if [[ -e "$mode_root" || -e "$cache_root" ]]; then
    echo "Refusing to overwrite a run or compile-cache root" >&2
    exit 2
  fi
  mkdir "$mode_root"
  mkdir -p "$cache_root/inductor" "$cache_root/triton" "$cache_root/torch_extensions"
  export TORCHINDUCTOR_CACHE_DIR=$cache_root/inductor
  export TRITON_CACHE_DIR=$cache_root/triton
  export TORCH_EXTENSIONS_DIR=$cache_root/torch_extensions
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
    --format=csv -l 5 > "$mode_root/gpu.csv" &
  MONITOR_PID=$!
  local args=(
    --shard "$SHARD_ROOT"
    --manifest-sha256 "$MANIFEST_SHA256"
    --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json"
    --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json"
    --efficiency-protocol "$SOURCE_ROOT/protocols/antibody_efficiency_gate_v1.json"
    --ddp-communication-protocol "$SOURCE_ROOT/protocols/antibody_ddp_communication_optimization_v1.json"
    --parity-protocol "$GATE_PROTOCOL"
    --parity-split-root "$SPLIT_ROOT"
    --run-seed "$seed"
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
  case "$mode" in
    safe_eager)
      args+=(--ddp-communication-dtype float32)
      ;;
    fp32_compiled)
      args+=(
        --ddp-communication-dtype float32
        --compile-blocks
        --compile-prewarm-buckets
        --fused-adamw
        --pinned-staging
        --bucketed-padding
      )
      ;;
    bf16_selected)
      args+=(
        --ddp-communication-dtype bfloat16
        --ddp-batched-grad-copy
        --compile-blocks
        --compile-prewarm-buckets
        --fused-adamw
        --pinned-staging
        --bucketed-padding
      )
      ;;
    *)
      echo "Unknown scientific parity mode" >&2
      exit 2
      ;;
  esac
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "$SOURCE_ROOT/scripts/run_antibody_pretraining_optimized.py" "${args[@]}" \
    2>&1 | tee "$mode_root/training.log"
  stop_monitor
  chmod -R a-w "$mode_root"
}

run_mode seed17_safe_eager 17 safe_eager
run_mode seed17_fp32_compiled 17 fp32_compiled
run_mode seed17_bf16_selected 17 bf16_selected
run_mode seed42_bf16_selected 42 bf16_selected
run_mode seed42_safe_eager 42 safe_eager
run_mode seed42_fp32_compiled 42 fp32_compiled
run_mode seed73_fp32_compiled 73 fp32_compiled
run_mode seed73_bf16_selected 73 bf16_selected
run_mode seed73_safe_eager 73 safe_eager

date --iso-8601=seconds >> "$RUN_ROOT/environment.txt"
find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 -r sha256sum \
  > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"

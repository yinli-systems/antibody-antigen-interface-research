#!/bin/bash
#SBATCH --job-name=abf-dynamo-bisection
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/abf-dynamo-bisection-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/abf-dynamo-bisection-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
PROTOCOL_SHA256=${ABF_BISECTION_PROTOCOL_SHA256:?required}
MINIFIER_PROTOCOL_SHA256=${ABF_MINIFIER_PROTOCOL_SHA256:?required}
MINIFIER_AUDIT=${ABF_MINIFIER_AUDIT:?ABF_MINIFIER_AUDIT is required}
MINIFIER_AUDIT_SHA256=${ABF_MINIFIER_AUDIT_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}
PROTOCOL=${SOURCE_ROOT}/protocols/antibody_dynamo_region_bisection_v1.json
MINIFIER_PROTOCOL=${SOURCE_ROOT}/protocols/antibody_compile_accuracy_minifier_v1.json
SPLIT_ROOT=${SOURCE_ROOT}/results/pretraining/antibody_ddp_parity_split_v2
RUN_ROOT=${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${OUTPUT_ROOT}" \
  "${SHARD_ROOT}" "${MINIFIER_AUDIT}" "${RUNTIME_MANIFEST}" \
  "${LOCAL_BASE}"; do
  if [[ "${path}" != /* ]]; then
    echo "All source, output, data, evidence, runtime, and cache paths must be absolute" >&2
    exit 2
  fi
done
for digest in "${MANIFEST_SHA256}" "${PROTOCOL_SHA256}" \
  "${MINIFIER_PROTOCOL_SHA256}" "${MINIFIER_AUDIT_SHA256}" \
  "${RUNTIME_MANIFEST_SHA256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "All external anchors must be lowercase SHA-256 digests" >&2
    exit 2
  fi
done
test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" = "${PROTOCOL_SHA256}"
test "$(sha256sum "${MINIFIER_PROTOCOL}" | awk '{print $1}')" = \
  "${MINIFIER_PROTOCOL_SHA256}"
test "$(sha256sum "${MINIFIER_AUDIT}" | awk '{print $1}')" = \
  "${MINIFIER_AUDIT_SHA256}"
test ! -e "${RUN_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
mkdir "${RUN_ROOT}"

MONITOR_PID=
stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=
  fi
}
seal_run() {
  stop_monitor
  if [[ -d "${RUN_ROOT}" && ! -e "${RUN_ROOT}/artifact_sha256.txt" ]]; then
    find "${RUN_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum \
      > "${RUN_ROOT}/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "${RUN_ROOT}" 2>/dev/null || true
}
trap seal_run EXIT

unset TORCHDYNAMO_REPRO_LEVEL TORCHDYNAMO_REPRO_AFTER
unset TORCHDYNAMO_REPRO_FORWARD_ONLY TORCHDYNAMO_REPRO_IGNORE_NON_FP
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export ABF_TELEMETRY_INTERVAL_SECONDS=5

if [[ "$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')" != 8 ]]; then
  echo "Dynamo region bisection requires exactly eight visible GPUs" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '$0 == "NVIDIA GeForce RTX 5090" {n++} END {print n+0}')" != 8 ]]; then
  echo "Dynamo region bisection requires eight RTX 5090 GPUs" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${PROTOCOL}" "${SOURCE_ROOT}" "${SHARD_ROOT}" \
  "${MANIFEST_SHA256}" "${MINIFIER_PROTOCOL}" \
  "${MINIFIER_PROTOCOL_SHA256}" "${MINIFIER_AUDIT}" \
  "${MINIFIER_AUDIT_SHA256}" "${RUNTIME_MANIFEST}" \
  "${RUNTIME_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

protocol_path = Path(sys.argv[1])
source_root = Path(sys.argv[2])
shard_root = Path(sys.argv[3])
manifest_sha256 = sys.argv[4]
minifier_protocol = Path(sys.argv[5])
minifier_protocol_sha256 = sys.argv[6]
minifier_audit = Path(sys.argv[7])
minifier_audit_sha256 = sys.argv[8]
runtime_manifest = Path(sys.argv[9])
runtime_manifest_sha256 = sys.argv[10]
protocol = json.loads(protocol_path.read_text(encoding="ascii"))
if protocol.get("format_version") != "antibody_dynamo_region_bisection_protocol_v1":
    raise SystemExit("Unexpected Dynamo region-bisection protocol")
if protocol["inputs"]["dense_manifest_payload_sha256"] != manifest_sha256:
    raise SystemExit("Dense manifest differs")
bindings = {
    "audit_launcher_sha256": "slurm/antibody_dynamo_region_bisection_audit.sbatch",
    "auditor_sha256": "scripts/audit_antibody_dynamo_region_bisection.py",
    "base_runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
    "communication_protocol_sha256": "protocols/antibody_ddp_communication_optimization_v1.json",
    "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
    "launcher_sha256": "slurm/antibody_dynamo_region_bisection.sbatch",
    "model_family_sha256": "protocols/antibody_model_family_v1.json",
    "parity_evaluation_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy",
    "parity_protocol_sha256": "protocols/antibody_ddp_scientific_parity_gate_v2.json",
    "parity_split_receipt_file_sha256": "results/pretraining/antibody_ddp_parity_split_v2/receipt.json",
    "parity_training_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy",
    "training_protocol_sha256": "protocols/antibody_training_v1.json",
    "wrapper_sha256": "scripts/run_antibody_dynamo_region_bisection.py",
}
for key, relative in bindings.items():
    observed = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    if observed != protocol["inputs"][key]:
        raise SystemExit(f"Protocol-bound input differs: {relative}")
for key, path, expected in (
    ("minifier_protocol_file_sha256", minifier_protocol, minifier_protocol_sha256),
    ("minifier_audit_file_sha256", minifier_audit, minifier_audit_sha256),
):
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected or observed != protocol["inputs"][key]:
        raise SystemExit(f"Predecessor binding differs: {path}")
if not (shard_root / "manifest.json").is_file():
    raise SystemExit("Dense shard manifest is unavailable")
if hashlib.sha256(runtime_manifest.read_bytes()).hexdigest() != runtime_manifest_sha256:
    raise SystemExit("Runtime manifest differs")
if runtime_manifest_sha256 != protocol["inputs"]["runtime_manifest_file_sha256"]:
    raise SystemExit("Runtime manifest binding differs")
PY

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}" \
  > "${RUN_ROOT}/runtime_verification.json"
cp "${PROTOCOL}" "${RUN_ROOT}/protocol.json"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "partition=${SLURM_JOB_PARTITION:-}"
  echo "source_root=${SOURCE_ROOT}"
  echo "shard_root=${SHARD_ROOT}"
  echo "world_size=8"
  echo "run_count=5"
  echo "updates_per_run=500"
  echo "global_residue_token_target_per_update=262144"
  echo "ddp_gradient_reduction_dtype=float32"
  echo "claim_boundary=training-excluded Dynamo region bisection only"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
  nvidia-smi topo -m
} > "${RUN_ROOT}/environment.txt"

run_mode() {
  local label=$1
  local mode=$2
  local compiled=$3
  local mode_root=${RUN_ROOT}/${label}
  local cache_root=${LOCAL_BASE}/abf-dynamo-bisection-${SLURM_JOB_ID}-${label}
  test ! -e "${mode_root}"
  test ! -e "${cache_root}"
  mkdir "${mode_root}"
  mkdir -p "${cache_root}/inductor" "${cache_root}/triton" \
    "${cache_root}/torch_extensions"
  export TORCHINDUCTOR_CACHE_DIR=${cache_root}/inductor
  export TRITON_CACHE_DIR=${cache_root}/triton
  export TORCH_EXTENSIONS_DIR=${cache_root}/torch_extensions
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
    --format=csv -l 5 > "${mode_root}/gpu.csv" &
  MONITOR_PID=$!
  local args=(
    --region-bisection-protocol "${PROTOCOL}"
    --region-bisection-mode "${mode}"
    --shard "${SHARD_ROOT}"
    --manifest-sha256 "${MANIFEST_SHA256}"
    --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json"
    --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json"
    --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json"
    --ddp-communication-protocol "${SOURCE_ROOT}/protocols/antibody_ddp_communication_optimization_v1.json"
    --parity-protocol "${SOURCE_ROOT}/protocols/antibody_ddp_scientific_parity_gate_v2.json"
    --parity-split-root "${SPLIT_ROOT}"
    --run-seed 17
    --output "${mode_root}/run"
    --preset abf_650m
    --steps 500
    --micro-batch 16
    --target-tokens 262144
    --batch-quantum 512
    --backend ddp
    --ddp-bucket-cap-mb 25
    --ddp-communication-dtype float32
    --fsdp-output-dtype bfloat16
    --length-bucket-boundaries 96,128,144,194
    --deterministic-algorithms
    --audit-gradient-rank-parity
    --skip-checkpoint
  )
  if [[ "${compiled}" == true ]]; then
    args+=(--compile-blocks --compile-prewarm-buckets)
  elif [[ "${compiled}" != false ]]; then
    echo "Compiled control must be true or false" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "${SOURCE_ROOT}/scripts/run_antibody_dynamo_region_bisection.py" \
    "${args[@]}" 2>&1 | tee "${mode_root}/training.log"
  stop_monitor
  chmod -R a-w "${mode_root}"
}

run_mode safe_eager_r1 safe_eager false
run_mode dynamo_first_half dynamo_first_half_gc_off true
run_mode dynamo_second_half dynamo_second_half_gc_off true
run_mode dynamo_full_control dynamo_full_control_gc_off true
run_mode safe_eager_r2 safe_eager false

date --iso-8601=seconds >> "${RUN_ROOT}/environment.txt"
find "${RUN_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "${RUN_ROOT}/artifact_sha256.txt"
chmod -R a-w "${RUN_ROOT}"

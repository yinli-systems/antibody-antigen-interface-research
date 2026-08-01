#!/bin/bash
#SBATCH --job-name=abf-compile-minifier
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/abf-compile-minifier-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/abf-compile-minifier-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
MINIFIER_PROTOCOL_SHA256=${ABF_MINIFIER_PROTOCOL_SHA256:?required}
ATTRIBUTION_PROTOCOL_SHA256=${ABF_ATTRIBUTION_PROTOCOL_SHA256:?required}
PREDECESSOR_AUDIT=${ABF_PREDECESSOR_AUDIT:?required}
PREDECESSOR_AUDIT_SHA256=${ABF_PREDECESSOR_AUDIT_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}
MINIFIER_PROTOCOL=${SOURCE_ROOT}/protocols/antibody_compile_accuracy_minifier_v1.json
ATTRIBUTION_PROTOCOL=${SOURCE_ROOT}/protocols/antibody_compile_backend_attribution_v1.json
SPLIT_ROOT=${SOURCE_ROOT}/results/pretraining/antibody_ddp_parity_split_v2
RUN_ROOT=${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${OUTPUT_ROOT}" \
  "${SHARD_ROOT}" "${PREDECESSOR_AUDIT}" "${RUNTIME_MANIFEST}" \
  "${LOCAL_BASE}"; do
  if [[ "${path}" != /* ]]; then
    echo "All source, output, data, and runtime paths must be absolute" >&2
    exit 2
  fi
done
for digest in "${MANIFEST_SHA256}" "${MINIFIER_PROTOCOL_SHA256}" \
  "${ATTRIBUTION_PROTOCOL_SHA256}" "${PREDECESSOR_AUDIT_SHA256}" \
  "${RUNTIME_MANIFEST_SHA256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "All external anchors must be lowercase SHA-256 digests" >&2
    exit 2
  fi
done
test "$(sha256sum "${MINIFIER_PROTOCOL}" | awk '{print $1}')" = \
  "${MINIFIER_PROTOCOL_SHA256}"
test "$(sha256sum "${ATTRIBUTION_PROTOCOL}" | awk '{print $1}')" = \
  "${ATTRIBUTION_PROTOCOL_SHA256}"
test "$(sha256sum "${PREDECESSOR_AUDIT}" | awk '{print $1}')" = \
  "${PREDECESSOR_AUDIT_SHA256}"
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
seal_on_exit() {
  local status=$?
  trap - EXIT
  set +e
  stop_monitor
  if [[ -d "${RUN_ROOT}" && ! -e "${RUN_ROOT}/artifact_sha256.txt" ]]; then
    find "${RUN_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum \
      > "${RUN_ROOT}/artifact_sha256.txt"
    if [[ $? -ne 0 && ${status} -eq 0 ]]; then
      status=1
    fi
  fi
  chmod -R a-w "${RUN_ROOT}"
  exit "${status}"
}
trap seal_on_exit EXIT

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export ABF_TELEMETRY_INTERVAL_SECONDS=5

if [[ "$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')" != 8 ]]; then
  echo "The Dynamo minifier requires exactly eight visible GPUs" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader \
  | awk '$0 == "NVIDIA GeForce RTX 5090" {n++} END {print n+0}')" != 8 ]]; then
  echo "The Dynamo minifier requires eight RTX 5090 GPUs" >&2
  exit 2
fi

${PYTHON_BIN} - "${MINIFIER_PROTOCOL}" "${SOURCE_ROOT}" \
  "${PREDECESSOR_AUDIT}" "${PREDECESSOR_AUDIT_SHA256}" \
  "${ATTRIBUTION_PROTOCOL_SHA256}" "${MANIFEST_SHA256}" \
  "${RUNTIME_MANIFEST}" "${RUNTIME_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

protocol_path = Path(sys.argv[1])
source_root = Path(sys.argv[2])
predecessor_path = Path(sys.argv[3])
predecessor_sha256 = sys.argv[4]
attribution_protocol_sha256 = sys.argv[5]
manifest_sha256 = sys.argv[6]
runtime_manifest = Path(sys.argv[7])
runtime_manifest_sha256 = sys.argv[8]
protocol = json.loads(protocol_path.read_text(encoding="ascii"))
if protocol.get("format_version") != "antibody_compile_accuracy_minifier_protocol_v1":
    raise SystemExit("Unexpected accuracy-minifier protocol")
expected = protocol["inputs"]
bindings = {
    "attribution_protocol_sha256": "protocols/antibody_compile_backend_attribution_v1.json",
    "auditor_sha256": "scripts/audit_antibody_compile_accuracy_minifier.py",
    "base_attribution_wrapper_sha256": "scripts/run_antibody_compile_backend_attribution.py",
    "base_runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
    "communication_protocol_sha256": "protocols/antibody_ddp_communication_optimization_v1.json",
    "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
    "launcher_sha256": "slurm/antibody_compile_accuracy_minifier.sbatch",
    "model_family_sha256": "protocols/antibody_model_family_v1.json",
    "parity_evaluation_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy",
    "parity_protocol_sha256": "protocols/antibody_ddp_scientific_parity_gate_v2.json",
    "parity_split_receipt_file_sha256": "results/pretraining/antibody_ddp_parity_split_v2/receipt.json",
    "parity_training_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy",
    "training_protocol_sha256": "protocols/antibody_training_v1.json",
    "wrapper_sha256": "scripts/run_antibody_compile_accuracy_minifier.py",
}
for key, relative in bindings.items():
    observed = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    if observed != expected[key]:
        raise SystemExit(f"Protocol-bound source differs: {relative}")
if hashlib.sha256(predecessor_path.read_bytes()).hexdigest() != predecessor_sha256:
    raise SystemExit("Predecessor audit differs")
if predecessor_sha256 != protocol["predecessor"]["audit_file_sha256"]:
    raise SystemExit("Predecessor binding differs")
if attribution_protocol_sha256 != expected["attribution_protocol_sha256"]:
    raise SystemExit("Attribution protocol binding differs")
if manifest_sha256 != expected["dense_manifest_payload_sha256"]:
    raise SystemExit("Dense manifest binding differs")
if hashlib.sha256(runtime_manifest.read_bytes()).hexdigest() != runtime_manifest_sha256:
    raise SystemExit("Runtime manifest file differs")
if runtime_manifest_sha256 != expected["runtime_manifest_file_sha256"]:
    raise SystemExit("Runtime manifest binding differs")
PY

${PYTHON_BIN} "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}" \
  > "${RUN_ROOT}/runtime_verification.json"
cp "${MINIFIER_PROTOCOL}" "${RUN_ROOT}/protocol.json"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "world_size=8"
  echo "boundary=dynamo"
  echo "updates=500"
  echo "claim_boundary=compiler accuracy-minifier diagnostic only"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total \
    --format=csv
  nvidia-smi topo -m
} > "${RUN_ROOT}/environment.txt"

MODE_ROOT=${RUN_ROOT}/dynamo_eager
CACHE_ROOT=${LOCAL_BASE}/abf-compile-minifier-${SLURM_JOB_ID}
mkdir "${MODE_ROOT}"
test ! -e "${CACHE_ROOT}"
mkdir -p "${CACHE_ROOT}/inductor" "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT}/triton
export TORCH_EXTENSIONS_DIR=${CACHE_ROOT}/torch_extensions
nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
  --format=csv -l 5 > "${MODE_ROOT}/gpu.csv" &
MONITOR_PID=$!

${PYTHON_BIN} -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${SOURCE_ROOT}/scripts/run_antibody_compile_accuracy_minifier.py" \
  --accuracy-minifier-protocol "${MINIFIER_PROTOCOL}" \
  --predecessor-audit "${PREDECESSOR_AUDIT}" \
  --minifier-boundary dynamo \
  --backend-attribution-protocol "${ATTRIBUTION_PROTOCOL}" \
  --backend-attribution-mode dynamo_eager_gc_off \
  --shard "${SHARD_ROOT}" \
  --manifest-sha256 "${MANIFEST_SHA256}" \
  --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json" \
  --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json" \
  --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json" \
  --ddp-communication-protocol "${SOURCE_ROOT}/protocols/antibody_ddp_communication_optimization_v1.json" \
  --parity-protocol "${SOURCE_ROOT}/protocols/antibody_ddp_scientific_parity_gate_v2.json" \
  --parity-split-root "${SPLIT_ROOT}" \
  --run-seed 17 \
  --output "${MODE_ROOT}/run" \
  --preset abf_650m \
  --steps 500 \
  --micro-batch 16 \
  --target-tokens 262144 \
  --batch-quantum 512 \
  --backend ddp \
  --ddp-bucket-cap-mb 25 \
  --ddp-communication-dtype float32 \
  --fsdp-output-dtype bfloat16 \
  --length-bucket-boundaries 96,128,144,194 \
  --deterministic-algorithms \
  --audit-gradient-rank-parity \
  --skip-checkpoint \
  --compile-blocks \
  --compile-prewarm-buckets \
  2>&1 | tee "${MODE_ROOT}/training.log"
stop_monitor
date --iso-8601=seconds >> "${RUN_ROOT}/environment.txt"
find "${RUN_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "${RUN_ROOT}/artifact_sha256.txt"
chmod -R a-w "${RUN_ROOT}"

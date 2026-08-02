#!/bin/bash
#SBATCH --job-name=abf-component-confirm
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/abf-component-confirm-%j.out
#SBATCH --error=/tmp/abf-component-confirm-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
PROTOCOL=${ABF_COMPONENT_CONFIRMATION_PROTOCOL:?required}
PROTOCOL_SHA256=${ABF_PROTOCOL_SHA256:?required}
LOCALIZATION_PROTOCOL=${ABF_LOCALIZATION_PROTOCOL:?required}
LOCALIZATION_PROTOCOL_SHA256=${ABF_LOCALIZATION_PROTOCOL_SHA256:?required}
LOCALIZATION_AUDIT=${ABF_LOCALIZATION_AUDIT:?required}
LOCALIZATION_AUDIT_SHA256=${ABF_LOCALIZATION_AUDIT_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}
SPLIT_ROOT=${SOURCE_ROOT}/results/pretraining/antibody_ddp_parity_split_v2
RUN_ROOT=${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${OUTPUT_ROOT}" \
  "${SHARD_ROOT}" "${PROTOCOL}" "${LOCALIZATION_PROTOCOL}" \
  "${LOCALIZATION_AUDIT}" "${RUNTIME_MANIFEST}" "${LOCAL_BASE}"; do
  if [[ "${path}" != /* ]]; then
    echo "All source, data, evidence, output, and runtime paths must be absolute" >&2
    exit 2
  fi
done
for digest in "${MANIFEST_SHA256}" "${PROTOCOL_SHA256}" \
  "${LOCALIZATION_PROTOCOL_SHA256}" "${LOCALIZATION_AUDIT_SHA256}" \
  "${RUNTIME_MANIFEST_SHA256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "All external anchors must be lowercase SHA-256 digests" >&2
    exit 2
  fi
done
test "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" = "${PROTOCOL_SHA256}"
test "$(sha256sum "${LOCALIZATION_PROTOCOL}" | awk '{print $1}')" = "${LOCALIZATION_PROTOCOL_SHA256}"
test "$(sha256sum "${LOCALIZATION_AUDIT}" | awk '{print $1}')" = "${LOCALIZATION_AUDIT_SHA256}"
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

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN ABF_TELEMETRY_INTERVAL_SECONDS=5

if [[ "$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')" != 8 ]]; then
  echo "Component confirmation requires exactly eight visible GPUs" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '$0 == "NVIDIA GeForce RTX 5090" {n++} END {print n+0}')" != 8 ]]; then
  echo "Component confirmation requires eight RTX 5090 GPUs" >&2
  exit 2
fi

SELECTED_CANDIDATE=$("${PYTHON_BIN}" - "${PROTOCOL}" "${SOURCE_ROOT}" \
  "${SHARD_ROOT}" "${MANIFEST_SHA256}" "${LOCALIZATION_PROTOCOL_SHA256}" \
  "${LOCALIZATION_AUDIT_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

protocol_path = Path(sys.argv[1])
source_root = Path(sys.argv[2])
shard_root = Path(sys.argv[3])
manifest_sha256 = sys.argv[4]
localization_protocol_sha256 = sys.argv[5]
localization_audit_sha256 = sys.argv[6]
protocol = json.loads(protocol_path.read_text(encoding="ascii"))
if protocol.get("format_version") != "antibody_production_batch_efficiency_protocol_v3":
    raise SystemExit("Unexpected confirmation protocol")
if protocol.get("protocol_variant") != "antibody_inductor_component_confirmation_protocol_v1":
    raise SystemExit("Unexpected component confirmation variant")
if protocol["inputs"]["dense_manifest_payload_sha256"] != manifest_sha256:
    raise SystemExit("Dense manifest differs")
if protocol["inputs"]["localization_protocol_file_sha256"] != localization_protocol_sha256:
    raise SystemExit("Localization protocol differs")
if protocol["inputs"]["localization_audit_file_sha256"] != localization_audit_sha256:
    raise SystemExit("Localization audit differs")
bindings = {
    "communication_protocol_sha256": "protocols/antibody_ddp_communication_optimization_v1.json",
    "confirmation_audit_launcher_sha256": "slurm/antibody_inductor_component_confirmation_audit.sbatch",
    "confirmation_auditor_sha256": "scripts/audit_antibody_inductor_component_confirmation.py",
    "confirmation_freezer_sha256": "scripts/freeze_antibody_inductor_component_confirmation.py",
    "confirmation_launcher_sha256": "slurm/antibody_inductor_component_confirmation.sbatch",
    "confirmation_wrapper_sha256": "scripts/run_antibody_inductor_component_confirmation.py",
    "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
    "mfu_protocol_sha256": "protocols/antibody_foundation_scale_v2.json",
    "model_family_sha256": "protocols/antibody_model_family_v1.json",
    "parity_evaluation_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy",
    "parity_extension_protocol_sha256": "protocols/antibody_ddp_scientific_parity_extension_600_seed17_v1.json",
    "parity_protocol_sha256": "protocols/antibody_ddp_scientific_parity_gate_v2.json",
    "parity_split_receipt_file_sha256": "results/pretraining/antibody_ddp_parity_split_v2/receipt.json",
    "parity_training_indices_sha256": "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy",
    "runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
    "runner_wrapper_sha256": "scripts/run_antibody_pretraining_parity_extension.py",
    "training_protocol_sha256": "protocols/antibody_training_v1.json",
}
for key, relative in bindings.items():
    observed = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    if observed != protocol["inputs"][key]:
        raise SystemExit(f"Protocol-bound input differs: {relative}")
if not (shard_root / "manifest.json").is_file():
    raise SystemExit("Dense shard manifest is unavailable")
print(protocol["component_confirmation_selection"]["selected_candidate"])
PY
)
if [[ "${SELECTED_CANDIDATE}" != inductor_attention_only \
  && "${SELECTED_CANDIDATE}" != inductor_ffn_only ]]; then
  echo "Selected component candidate is not registered" >&2
  exit 2
fi

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
  echo "run_count=4"
  echo "updates_per_run=600"
  echo "selected_candidate=${SELECTED_CANDIDATE}"
  echo "claim_boundary=training-excluded repeated component confirmation"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
  nvidia-smi topo -m
} > "${RUN_ROOT}/environment.txt"

run_mode() {
  local label=$1
  local mode=$2
  local mode_root=${RUN_ROOT}/${label}
  local cache_root=${LOCAL_BASE}/abf-component-confirm-${SLURM_JOB_ID}-${label}
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
    --component-confirmation-protocol "${PROTOCOL}"
    --component-confirmation-mode "${mode}"
    --shard "${SHARD_ROOT}"
    --manifest-sha256 "${MANIFEST_SHA256}"
    --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json"
    --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json"
    --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json"
    --ddp-communication-protocol "${SOURCE_ROOT}/protocols/antibody_ddp_communication_optimization_v1.json"
    --parity-protocol "${SOURCE_ROOT}/protocols/antibody_ddp_scientific_parity_gate_v2.json"
    --parity-extension-protocol "${SOURCE_ROOT}/protocols/antibody_ddp_scientific_parity_extension_600_seed17_v1.json"
    --parity-split-root "${SPLIT_ROOT}"
    --run-seed 17
    --output "${mode_root}/run"
    --preset abf_650m
    --steps 600
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
  if [[ "${mode}" == "${SELECTED_CANDIDATE}" ]]; then
    args+=(--compile-blocks --compile-prewarm-buckets --disable-compiled-cudagraphs)
  elif [[ "${mode}" != safe_eager ]]; then
    echo "Unknown component confirmation mode" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "${SOURCE_ROOT}/scripts/run_antibody_inductor_component_confirmation.py" \
    "${args[@]}" 2>&1 | tee "${mode_root}/training.log"
  stop_monitor
  chmod -R a-w "${mode_root}"
}

run_mode safe_eager_r1 safe_eager
run_mode component_compiled_r1 "${SELECTED_CANDIDATE}"
run_mode component_compiled_r2 "${SELECTED_CANDIDATE}"
run_mode safe_eager_r2 safe_eager

date --iso-8601=seconds >> "${RUN_ROOT}/environment.txt"
find "${RUN_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "${RUN_ROOT}/artifact_sha256.txt"
chmod -R a-w "${RUN_ROOT}"

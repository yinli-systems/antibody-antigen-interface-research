#!/bin/bash
#SBATCH --job-name=abf-oas-4g-eff
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-eff-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-eff-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
RECOVERY_AUDIT=${ABF_RECOVERY_AUDIT:?ABF_RECOVERY_AUDIT is required}
RECOVERY_AUDIT_FILE_SHA256=${ABF_RECOVERY_AUDIT_FILE_SHA256:?required}
RECOVERY_AUDIT_PAYLOAD_SHA256=${ABF_RECOVERY_AUDIT_PAYLOAD_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${OUTPUT_ROOT}" "${RECOVERY_AUDIT}" "${RUNTIME_MANIFEST}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
for digest in "${CORPUS_PROTOCOL_FILE_SHA256}" \
  "${RECOVERY_AUDIT_FILE_SHA256}" "${RECOVERY_AUDIT_PAYLOAD_SHA256}" \
  "${RUNTIME_MANIFEST_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "External anchor is not a SHA-256" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Output root already exists" >&2; exit 2; }

MANIFEST_PATH=${CORPUS_ROOT}/corpus_manifest.json
CORPUS_AUDIT=${CORPUS_ROOT}/independent_audit_receipt.json
for path in "${MANIFEST_PATH}" "${CORPUS_AUDIT}" "${RECOVERY_AUDIT}"; do
  [[ -f "${path}" && ! -L "${path}" ]] \
    || { echo "Audit prerequisite is unavailable" >&2; exit 2; }
done
MANIFEST_FILE_SHA256=$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')
CORPUS_AUDIT_FILE_SHA256=$(sha256sum "${CORPUS_AUDIT}" | awk '{print $1}')
MANIFEST_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["manifest_sha256"])' \
  "${MANIFEST_PATH}")
CORPUS_AUDIT_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["receipt_sha256"])' \
  "${CORPUS_AUDIT}")

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN ABF_TELEMETRY_INTERVAL_SECONDS=5

[[ "$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')" == 4 ]] \
  || { echo "Gate requires exactly four visible GPUs" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader \
  | awk '$0 == "NVIDIA GeForce RTX 5090" {n++} END {print n+0}')" == 4 ]] \
  || { echo "Gate requires four RTX 5090 GPUs" >&2; exit 2; }
ulimit -Sn 4096
[[ "$(ulimit -Sn)" == 4096 ]]

BUNDLE_ROOT=${OUTPUT_ROOT}/protocol-bundle
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/freeze_antibody_four_gpu_efficiency_gate.py" \
  --source-root "${SOURCE_ROOT}" \
  --corpus-root "${CORPUS_ROOT}" \
  --manifest-file-sha256 "${MANIFEST_FILE_SHA256}" \
  --manifest-payload-sha256 "${MANIFEST_PAYLOAD_SHA256}" \
  --corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}" \
  --independent-audit "${CORPUS_AUDIT}" \
  --independent-audit-file-sha256 "${CORPUS_AUDIT_FILE_SHA256}" \
  --independent-audit-payload-sha256 "${CORPUS_AUDIT_PAYLOAD_SHA256}" \
  --recovery-audit "${RECOVERY_AUDIT}" \
  --recovery-audit-file-sha256 "${RECOVERY_AUDIT_FILE_SHA256}" \
  --recovery-audit-payload-sha256 "${RECOVERY_AUDIT_PAYLOAD_SHA256}" \
  --output-dir "${BUNDLE_ROOT}"
PROTOCOL=${BUNDLE_ROOT}/protocol.json

RUN_ROOT=${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}
mkdir "${RUN_ROOT}"
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}
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

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}" \
  > "${RUN_ROOT}/runtime_verification.json"
cp "${PROTOCOL}" "${RUN_ROOT}/protocol.json"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "world_size=4"
  echo "run_order=safe_eager_r1,compiled_r1,compiled_r2,safe_eager_r2"
  echo "updates_per_run=600"
  echo "warmup_updates_excluded=100"
  echo "global_residue_tokens_per_update=131072"
  echo "validation_panel_records=8192"
  echo "claim_boundary=real-OAS paired efficiency and equivalence gate only"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
  nvidia-smi topo -m
} > "${RUN_ROOT}/environment.txt"

COMMON_ARGS=(
  --production-corpus-root "${CORPUS_ROOT}"
  --production-corpus-manifest-file-sha256 "${MANIFEST_FILE_SHA256}"
  --production-corpus-manifest-payload-sha256 "${MANIFEST_PAYLOAD_SHA256}"
  --production-corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}"
  --production-corpus-independent-audit "${CORPUS_AUDIT}"
  --production-corpus-independent-audit-file-sha256 "${CORPUS_AUDIT_FILE_SHA256}"
  --production-corpus-independent-audit-payload-sha256 "${CORPUS_AUDIT_PAYLOAD_SHA256}"
  --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json"
  --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json"
  --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json"
  --run-role continuous
  --preset abf_650m
  --steps 600
  --micro-batch 16
  --target-tokens 131072
  --batch-quantum 256
  --run-seed 17
  --length-bucket-boundaries 96,128,160,194
  --ddp-bucket-cap-mb 25
  --maximum-open-arrays 1024
  --minimum-soft-open-file-limit 4096
)

run_mode() {
  local label=$1
  local mode=$2
  local gate=${BUNDLE_ROOT}/${mode}_gate.json
  local mode_root=${RUN_ROOT}/${label}
  local cache_root=${LOCAL_BASE}/abf-oas-4g-eff-${SLURM_JOB_ID}-${label}
  [[ ! -e "${mode_root}" && ! -e "${cache_root}" ]] \
    || { echo "Run or cache root already exists" >&2; exit 2; }
  mkdir "${mode_root}"
  mkdir -p "${cache_root}/inductor" "${cache_root}/triton" \
    "${cache_root}/torch_extensions"
  export TORCHINDUCTOR_CACHE_DIR=${cache_root}/inductor
  export TRITON_CACHE_DIR=${cache_root}/triton
  export TORCH_EXTENSIONS_DIR=${cache_root}/torch_extensions
  unset TORCH_LOGS || true
  local optimization=()
  if [[ "${mode}" == compiled ]]; then
    optimization=(
      --compile-blocks
      --disable-compiled-cudagraphs
      --compile-prewarm-buckets
      --fused-adamw
      --pinned-staging
      --bucketed-padding
      --length-bucketing
    )
    export TORCH_LOGS=recompiles
  elif [[ "${mode}" != safe_eager ]]; then
    echo "Unknown efficiency mode" >&2
    exit 2
  fi
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
    --format=csv -l 5 > "${mode_root}/gpu.csv" &
  MONITOR_PID=$!
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "${SOURCE_ROOT}/scripts/run_antibody_pretraining_production_gate.py" \
    "${COMMON_ARGS[@]}" --gate-protocol "${gate}" \
    --output "${mode_root}/run" "${optimization[@]}" \
    2>&1 | tee "${mode_root}/training.log"
  stop_monitor
  unset TORCH_LOGS || true
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "${SOURCE_ROOT}/scripts/evaluate_antibody_production_checkpoint.py" \
    --production-corpus-root "${CORPUS_ROOT}" \
    --production-corpus-manifest-file-sha256 "${MANIFEST_FILE_SHA256}" \
    --production-corpus-manifest-payload-sha256 "${MANIFEST_PAYLOAD_SHA256}" \
    --production-corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}" \
    --production-corpus-independent-audit "${CORPUS_AUDIT}" \
    --production-corpus-independent-audit-file-sha256 "${CORPUS_AUDIT_FILE_SHA256}" \
    --production-corpus-independent-audit-payload-sha256 "${CORPUS_AUDIT_PAYLOAD_SHA256}" \
    --gate-protocol "${gate}" \
    --run-root "${mode_root}/run" \
    --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json" \
    --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json" \
    --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json" \
    --output "${mode_root}/evaluation.json" \
    --preset abf_650m \
    --panel-sample-count 8192 \
    --panel-seed 2026081501 \
    --panel-indices-sha256 cc9219f83bfdaef51e97ee9845f31ba9dd2cbde9863715d6fdfa7a50478197ec \
    --mask-seed 2026081503 \
    --mask-update 0 \
    --micro-batch 16 \
    --maximum-open-arrays 1024 \
    2>&1 | tee "${mode_root}/evaluation.log"
}

run_mode safe_eager_r1 safe_eager
run_mode compiled_r1 compiled
run_mode compiled_r2 compiled
run_mode safe_eager_r2 safe_eager

date --iso-8601=seconds >> "${RUN_ROOT}/environment.txt"
find "${RUN_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "${RUN_ROOT}/artifact_sha256.txt"
chmod -R a-w "${RUN_ROOT}"
trap - EXIT

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/audit_antibody_four_gpu_efficiency_gate.py" \
  --run-root "${RUN_ROOT}" \
  --protocol "${PROTOCOL}" \
  --bundle-root "${BUNDLE_ROOT}" \
  --output "${OUTPUT_ROOT}/independent-audit.json"
chmod a-w "${OUTPUT_ROOT}/independent-audit.json"
GATE_PASSED=$("${PYTHON_BIN}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1], encoding="ascii"))["gate_passed"]).lower())' \
  "${OUTPUT_ROOT}/independent-audit.json")
echo "four_gpu_efficiency_audit=PASS"
echo "configuration_gate_passed=${GATE_PASSED}"
echo "independent_audit=${OUTPUT_ROOT}/independent-audit.json"

#!/bin/bash
#SBATCH --job-name=abf-oas-4g-shape-v2
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-shape-v2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-shape-v2-%j.err

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
  || { echo "Probe requires exactly four visible GPUs" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader \
  | awk '$0 == "NVIDIA GeForce RTX 5090" {n++} END {print n+0}')" == 4 ]] \
  || { echo "Probe requires four RTX 5090 GPUs" >&2; exit 2; }
ulimit -Sn 4096
[[ "$(ulimit -Sn)" == 4096 ]]

BUNDLE_ROOT=${OUTPUT_ROOT}/protocol-bundle
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/freeze_antibody_four_gpu_shape_stability_probe_v2.py" \
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
  echo "run_order=static_194_r1,hybrid_160_194_r1,hybrid_128_194_r1,hybrid_128_194_r2,hybrid_160_194_r2,static_194_r2"
  echo "updates_per_run=300"
  echo "warmup_updates_excluded=100"
  echo "global_residue_tokens_per_update=131072"
  echo "claim_boundary=real-OAS hybrid-shape systems probe only"
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
  --steps 300
  --micro-batch 16
  --target-tokens 131072
  --batch-quantum 256
  --run-seed 17
  --ddp-bucket-cap-mb 25
  --maximum-open-arrays 1024
  --minimum-soft-open-file-limit 4096
  --skip-checkpoint
  --compile-blocks
  --disable-compiled-cudagraphs
  --compile-prewarm-buckets
  --fused-adamw
  --pinned-staging
  --bucketed-padding
  --length-bucketing
)

run_candidate() {
  local label=$1
  local candidate=$2
  local boundaries=$3
  local gate=${BUNDLE_ROOT}/${candidate}_gate.json
  local mode_root=${RUN_ROOT}/${label}
  local cache_root=${LOCAL_BASE}/abf-oas-4g-shape-v2-${SLURM_JOB_ID}-${label}
  [[ ! -e "${mode_root}" && ! -e "${cache_root}" ]] \
    || { echo "Run or cache root already exists" >&2; exit 2; }
  mkdir "${mode_root}"
  mkdir -p "${cache_root}/inductor" "${cache_root}/triton" \
    "${cache_root}/torch_extensions"
  export TORCHINDUCTOR_CACHE_DIR=${cache_root}/inductor
  export TRITON_CACHE_DIR=${cache_root}/triton
  export TORCH_EXTENSIONS_DIR=${cache_root}/torch_extensions
  export TORCH_LOGS=recompiles
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
    --format=csv -l 5 > "${mode_root}/gpu.csv" &
  MONITOR_PID=$!
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "${SOURCE_ROOT}/scripts/run_antibody_pretraining_production_gate.py" \
    "${COMMON_ARGS[@]}" \
    --length-bucket-boundaries "${boundaries}" \
    --gate-protocol "${gate}" \
    --output "${mode_root}/run" \
    2>&1 | tee "${mode_root}/training.log"
  stop_monitor
  unset TORCH_LOGS || true
}

run_candidate static_194_r1 static_194 194
run_candidate hybrid_160_194_r1 hybrid_160_194 160,194
run_candidate hybrid_128_194_r1 hybrid_128_194 128,194
run_candidate hybrid_128_194_r2 hybrid_128_194 128,194
run_candidate hybrid_160_194_r2 hybrid_160_194 160,194
run_candidate static_194_r2 static_194 194

date --iso-8601=seconds >> "${RUN_ROOT}/environment.txt"
find "${RUN_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "${RUN_ROOT}/artifact_sha256.txt"
chmod -R a-w "${RUN_ROOT}"
trap - EXIT

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/audit_antibody_four_gpu_shape_stability_probe_v2.py" \
  --run-root "${RUN_ROOT}" \
  --protocol "${PROTOCOL}" \
  --bundle-root "${BUNDLE_ROOT}" \
  --output "${OUTPUT_ROOT}/independent-audit.json"
chmod a-w "${OUTPUT_ROOT}/independent-audit.json"
PROBE_PASSED=$("${PYTHON_BIN}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1], encoding="ascii"))["probe_passed"]).lower())' \
  "${OUTPUT_ROOT}/independent-audit.json")
SELECTED=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["selected_candidate_for_full_efficiency_gate"])' \
  "${OUTPUT_ROOT}/independent-audit.json")
echo "four_gpu_shape_stability_v2_audit=PASS"
echo "hybrid_shape_candidate_passed=${PROBE_PASSED}"
echo "selected_candidate=${SELECTED}"
echo "independent_audit=${OUTPUT_ROOT}/independent-audit.json"

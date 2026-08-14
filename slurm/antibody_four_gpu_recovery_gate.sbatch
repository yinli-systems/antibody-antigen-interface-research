#!/bin/bash
#SBATCH --job-name=abf-oas-recovery-4g
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/abf-oas-recovery-4g-%j.out
#SBATCH --error=/tmp/abf-oas-recovery-4g-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
CORPUS_MANIFEST_FILE_SHA256=${ABF_CORPUS_MANIFEST_FILE_SHA256:?required}
CORPUS_MANIFEST_PAYLOAD_SHA256=${ABF_CORPUS_MANIFEST_PAYLOAD_SHA256:?required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
CORPUS_AUDIT_PATH=${ABF_CORPUS_AUDIT_PATH:?required}
CORPUS_AUDIT_FILE_SHA256=${ABF_CORPUS_AUDIT_FILE_SHA256:?required}
CORPUS_AUDIT_PAYLOAD_SHA256=${ABF_CORPUS_AUDIT_PAYLOAD_SHA256:?required}
GATE_PROTOCOL=${ABF_GATE_PROTOCOL:?ABF_GATE_PROTOCOL is required}
GATE_PROTOCOL_SHA256=${ABF_GATE_PROTOCOL_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${OUTPUT_ROOT}" \
  "${CORPUS_ROOT}" "${CORPUS_AUDIT_PATH}" "${GATE_PROTOCOL}" \
  "${RUNTIME_MANIFEST}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
for digest in "${CORPUS_MANIFEST_FILE_SHA256}" \
  "${CORPUS_MANIFEST_PAYLOAD_SHA256}" "${CORPUS_PROTOCOL_FILE_SHA256}" \
  "${CORPUS_AUDIT_FILE_SHA256}" "${CORPUS_AUDIT_PAYLOAD_SHA256}" \
  "${GATE_PROTOCOL_SHA256}" "${RUNTIME_MANIFEST_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "External anchor is not a SHA-256" >&2; exit 2; }
done
[[ "$(sha256sum "${GATE_PROTOCOL}" | awk '{print $1}')" == "${GATE_PROTOCOL_SHA256}" ]]

MINIMUM_SOFT_OPEN_FILE_LIMIT=4096
HARD_OPEN_FILE_LIMIT=$(ulimit -Hn)
if [[ "${HARD_OPEN_FILE_LIMIT}" != unlimited ]]; then
  (( HARD_OPEN_FILE_LIMIT >= MINIMUM_SOFT_OPEN_FILE_LIMIT ))
fi
ulimit -Sn "${MINIMUM_SOFT_OPEN_FILE_LIMIT}"
[[ "$(ulimit -Sn)" == "${MINIMUM_SOFT_OPEN_FILE_LIMIT}" ]]

RUN_ROOT=${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
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

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}" \
  > "${RUN_ROOT}/runtime_verification.json"
cp "${GATE_PROTOCOL}" "${RUN_ROOT}/protocol.json"

DISABLE_COMPILED_CUDAGRAPHS=$("${PYTHON_BIN}" - "${GATE_PROTOCOL}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="ascii"))
execution = value["execution"]
if execution.get("world_size") != 4:
    raise SystemExit("Protocol does not freeze four GPUs")
optimization = execution["optimization"]
if optimization.get("compile_blocks") is not True:
    raise SystemExit("Regional compilation is not frozen")
print("true" if optimization.get("disable_compiled_cudagraphs") else "false")
PY
)

COMMON_ARGS=(
  --candidate-gate-protocol "${GATE_PROTOCOL}"
  --production-corpus-root "${CORPUS_ROOT}"
  --production-corpus-manifest-file-sha256 "${CORPUS_MANIFEST_FILE_SHA256}"
  --production-corpus-manifest-payload-sha256 "${CORPUS_MANIFEST_PAYLOAD_SHA256}"
  --production-corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}"
  --production-corpus-independent-audit "${CORPUS_AUDIT_PATH}"
  --production-corpus-independent-audit-file-sha256 "${CORPUS_AUDIT_FILE_SHA256}"
  --production-corpus-independent-audit-payload-sha256 "${CORPUS_AUDIT_PAYLOAD_SHA256}"
  --gate-protocol "${GATE_PROTOCOL}"
  --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json"
  --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json"
  --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json"
  --preset abf_650m
  --micro-batch 16
  --target-tokens 131072
  --batch-quantum 256
  --run-seed 17
  --length-bucket-boundaries 96,128,160,194
  --ddp-bucket-cap-mb 25
  --maximum-open-arrays 1024
  --minimum-soft-open-file-limit 4096
  --compile-blocks
  --compile-prewarm-buckets
)
if [[ "${DISABLE_COMPILED_CUDAGRAPHS}" == true ]]; then
  COMMON_ARGS+=(--disable-compiled-cudagraphs)
elif [[ "${DISABLE_COMPILED_CUDAGRAPHS}" != false ]]; then
  echo "CUDAGraph control is invalid" >&2
  exit 2
fi

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "partition=${SLURM_JOB_PARTITION:-}"
  echo "source_root=${SOURCE_ROOT}"
  echo "corpus_root=${CORPUS_ROOT}"
  echo "world_size=4"
  echo "target_residue_tokens=131072"
  echo "target_residue_tokens_per_gpu=32768"
  echo "claim_boundary=real-OAS exact recovery gate only"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
  nvidia-smi topo -m
} > "${RUN_ROOT}/environment.txt"

run_gate() {
  local role=$1
  local steps=$2
  shift 2
  local output=${RUN_ROOT}/${role}
  local cache_root=${LOCAL_BASE}/abf-oas-recovery-4g-${SLURM_JOB_ID}-${role}
  [[ ! -e "${output}" && ! -e "${cache_root}" ]] \
    || { echo "Run or cache root already exists" >&2; exit 2; }
  mkdir -p "${cache_root}/inductor" "${cache_root}/triton" \
    "${cache_root}/torch_extensions"
  export TORCHINDUCTOR_CACHE_DIR=${cache_root}/inductor
  export TRITON_CACHE_DIR=${cache_root}/triton
  export TORCH_EXTENSIONS_DIR=${cache_root}/torch_extensions
  nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
    --format=csv -l 5 > "${RUN_ROOT}/${role}-gpu.csv" &
  MONITOR_PID=$!
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "${SOURCE_ROOT}/scripts/run_antibody_pretraining_production_candidate_gate.py" \
    "${COMMON_ARGS[@]}" --run-role "${role}" --steps "${steps}" \
    --output "${output}" "$@" 2>&1 | tee "${RUN_ROOT}/${role}-training.log"
  stop_monitor
}

run_gate continuous 4 --skip-checkpoint
run_gate stage1 2
run_gate resumed 2 --resume-root "${RUN_ROOT}/stage1" --skip-checkpoint

date --iso-8601=seconds >> "${RUN_ROOT}/environment.txt"
find "${RUN_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "${RUN_ROOT}/artifact_sha256.txt"
chmod -R a-w "${RUN_ROOT}"
trap - EXIT

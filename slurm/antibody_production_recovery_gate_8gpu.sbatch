#!/bin/bash
#SBATCH --job-name=abf-oas-recovery-8g
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/abf-oas-recovery-8g-%j.out
#SBATCH --error=/tmp/abf-oas-recovery-8g-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
CORPUS_MANIFEST_FILE_SHA256=${ABF_CORPUS_MANIFEST_FILE_SHA256:?required}
CORPUS_MANIFEST_PAYLOAD_SHA256=${ABF_CORPUS_MANIFEST_PAYLOAD_SHA256:?required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
CORPUS_AUDIT_PATH=${ABF_CORPUS_AUDIT_PATH:?ABF_CORPUS_AUDIT_PATH is required}
CORPUS_AUDIT_FILE_SHA256=${ABF_CORPUS_AUDIT_FILE_SHA256:?required}
CORPUS_AUDIT_PAYLOAD_SHA256=${ABF_CORPUS_AUDIT_PAYLOAD_SHA256:?required}
GATE_PROTOCOL=${ABF_GATE_PROTOCOL:?ABF_GATE_PROTOCOL is required}
GATE_PROTOCOL_SHA256=${ABF_GATE_PROTOCOL_SHA256:?required}
CONFIRMATION_PROTOCOL=${ABF_CONFIRMATION_PROTOCOL:?ABF_CONFIRMATION_PROTOCOL is required}
CONFIRMATION_PROTOCOL_SHA256=${ABF_CONFIRMATION_PROTOCOL_SHA256:?required}
CONFIRMATION_AUDIT=${ABF_CONFIRMATION_AUDIT:?ABF_CONFIRMATION_AUDIT is required}
CONFIRMATION_AUDIT_SHA256=${ABF_CONFIRMATION_AUDIT_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${OUTPUT_ROOT}" \
  "${CORPUS_ROOT}" "${CORPUS_AUDIT_PATH}" "${GATE_PROTOCOL}" \
  "${CONFIRMATION_PROTOCOL}" "${CONFIRMATION_AUDIT}" \
  "${RUNTIME_MANIFEST}"; do
  if [[ "${path}" != /* ]]; then
    echo "All source, data, evidence, output, and runtime paths must be absolute" >&2
    exit 2
  fi
done
for digest in "${CORPUS_MANIFEST_FILE_SHA256}" \
  "${CORPUS_MANIFEST_PAYLOAD_SHA256}" "${CORPUS_PROTOCOL_FILE_SHA256}" \
  "${CORPUS_AUDIT_FILE_SHA256}" "${CORPUS_AUDIT_PAYLOAD_SHA256}" \
  "${GATE_PROTOCOL_SHA256}" "${CONFIRMATION_PROTOCOL_SHA256}" \
  "${CONFIRMATION_AUDIT_SHA256}" "${RUNTIME_MANIFEST_SHA256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "All external anchors must be lowercase SHA-256 digests" >&2
    exit 2
  fi
done
test "$(sha256sum "${GATE_PROTOCOL}" | awk '{print $1}')" = "${GATE_PROTOCOL_SHA256}"
test "$(sha256sum "${CONFIRMATION_PROTOCOL}" | awk '{print $1}')" = "${CONFIRMATION_PROTOCOL_SHA256}"
test "$(sha256sum "${CONFIRMATION_AUDIT}" | awk '{print $1}')" = "${CONFIRMATION_AUDIT_SHA256}"

MINIMUM_SOFT_OPEN_FILE_LIMIT=4096
HARD_OPEN_FILE_LIMIT=$(ulimit -Hn)
if [[ "${HARD_OPEN_FILE_LIMIT}" != unlimited ]]; then
  (( HARD_OPEN_FILE_LIMIT >= MINIMUM_SOFT_OPEN_FILE_LIMIT ))
fi
ulimit -Sn "${MINIMUM_SOFT_OPEN_FILE_LIMIT}"
test "$(ulimit -Sn)" = "${MINIMUM_SOFT_OPEN_FILE_LIMIT}"

RUN_ROOT=${OUTPUT_ROOT}/slurm-${SLURM_JOB_ID}
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite an existing run root" >&2
  exit 2
fi
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

if [[ "$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')" != 8 ]]; then
  echo "The real-OAS recovery gate requires exactly eight visible GPUs" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '$0 == "NVIDIA GeForce RTX 5090" {n++} END {print n+0}')" != 8 ]]; then
  echo "The real-OAS recovery gate requires eight RTX 5090 GPUs" >&2
  exit 2
fi

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}" \
  > "${RUN_ROOT}/runtime_verification.json"
cp "${GATE_PROTOCOL}" "${RUN_ROOT}/protocol.json"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "partition=${SLURM_JOB_PARTITION:-}"
  echo "source_root=${SOURCE_ROOT}"
  echo "corpus_root=${CORPUS_ROOT}"
  echo "world_size=8"
  echo "run_count=3"
  echo "target_residue_tokens=262144"
  echo "claim_boundary=real-OAS exact recovery only"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
  nvidia-smi topo -m
} > "${RUN_ROOT}/environment.txt"

DISABLE_COMPILED_CUDAGRAPHS=$("${PYTHON_BIN}" - "${GATE_PROTOCOL}" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
controls = value["candidate_selection"]["execution_controls"]
backend_controls = {
    "compile_backend": "inductor",
    "disable_compiled_cudagraphs": True,
    "disable_python_gc": True,
}
legacy_keys = {"accuracy_first_bf16_reduction", "disable_python_gc"}
if controls == backend_controls:
    print("true")
elif set(controls) == legacy_keys:
    print("false")
else:
    raise SystemExit("Candidate execution controls differ")
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
  --target-tokens 262144
  --batch-quantum 512
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
  echo "CUDAGraph control must be true or false" >&2
  exit 2
fi

run_gate() {
  local role=$1
  local steps=$2
  local output=${RUN_ROOT}/${role}
  shift 2
  local cache_root=${LOCAL_BASE}/abf-oas-recovery-8g-${SLURM_JOB_ID}-${role}
  if [[ -e "${output}" || -e "${cache_root}" ]]; then
    echo "Refusing to overwrite a run or compile-cache root" >&2
    exit 2
  fi
  mkdir -p "${cache_root}/inductor" "${cache_root}/triton" \
    "${cache_root}/torch_extensions"
  export TORCHINDUCTOR_CACHE_DIR=${cache_root}/inductor
  export TRITON_CACHE_DIR=${cache_root}/triton
  export TORCH_EXTENSIONS_DIR=${cache_root}/torch_extensions
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
    --format=csv -l 5 > "${RUN_ROOT}/${role}-gpu.csv" &
  MONITOR_PID=$!
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
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

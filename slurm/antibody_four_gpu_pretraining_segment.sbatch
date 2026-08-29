#!/bin/bash
#SBATCH --job-name=abf-oas-4g-pretrain
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-pretrain-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-pretrain-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
CAMPAIGN_ROOT=${ABF_CAMPAIGN_ROOT:?ABF_CAMPAIGN_ROOT is required}
GATE_PROTOCOL=${ABF_GATE_PROTOCOL:?ABF_GATE_PROTOCOL is required}
GATE_PROTOCOL_SHA256=${ABF_GATE_PROTOCOL_SHA256:?ABF_GATE_PROTOCOL_SHA256 is required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}
SEGMENT_INDEX=${ABF_SEGMENT_INDEX:?ABF_SEGMENT_INDEX is required}
PREVIOUS_SEGMENT_ROOT=${ABF_PREVIOUS_SEGMENT_ROOT:-}
PREVIOUS_SEGMENT_AUDIT=${ABF_PREVIOUS_SEGMENT_AUDIT:-}

[[ "${SEGMENT_INDEX}" =~ ^[0-9]+$ ]] || { echo "Invalid segment index" >&2; exit 2; }
for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${CAMPAIGN_ROOT}" "${GATE_PROTOCOL}" "${RUNTIME_MANIFEST}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
for digest in "${CORPUS_PROTOCOL_FILE_SHA256}" "${RUNTIME_MANIFEST_SHA256}" \
  "${GATE_PROTOCOL_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "External anchor is not a SHA-256" >&2; exit 2; }
done
[[ -f "${GATE_PROTOCOL}" && ! -L "${GATE_PROTOCOL}" ]] \
  || { echo "Campaign gate is unavailable" >&2; exit 2; }
[[ "$(sha256sum "${GATE_PROTOCOL}" | awk '{print $1}')" == "${GATE_PROTOCOL_SHA256}" ]] \
  || { echo "Campaign gate hash differs" >&2; exit 2; }

MANIFEST_PATH=${CORPUS_ROOT}/corpus_manifest.json
CORPUS_AUDIT=${CORPUS_ROOT}/independent_audit_receipt.json
MANIFEST_FILE_SHA256=$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')
CORPUS_AUDIT_FILE_SHA256=$(sha256sum "${CORPUS_AUDIT}" | awk '{print $1}')
MANIFEST_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["manifest_sha256"])' \
  "${MANIFEST_PATH}")
CORPUS_AUDIT_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["receipt_sha256"])' \
  "${CORPUS_AUDIT}")
SEGMENT_STEPS=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["campaign"]["segment_maximum_steps"])' \
  "${GATE_PROTOCOL}")
CAMPAIGN_TOKEN_LIMIT=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["campaign"]["campaign_token_limit"])' \
  "${GATE_PROTOCOL}")

printf -v SEGMENT_LABEL 'segment-%06d' "${SEGMENT_INDEX}"
SEGMENT_ROOT=${CAMPAIGN_ROOT}/${SEGMENT_LABEL}
AUDIT_ROOT=${CAMPAIGN_ROOT}/audits
AUDIT_OUTPUT=${AUDIT_ROOT}/${SEGMENT_LABEL}.json
[[ ! -e "${SEGMENT_ROOT}" && ! -e "${AUDIT_OUTPUT}" ]] \
  || { echo "Segment output already exists" >&2; exit 2; }
mkdir -p "${AUDIT_ROOT}"
mkdir "${SEGMENT_ROOT}"

if [[ "${SEGMENT_INDEX}" == 0 ]]; then
  [[ -z "${PREVIOUS_SEGMENT_ROOT}" && -z "${PREVIOUS_SEGMENT_AUDIT}" ]] \
    || { echo "First segment cannot resume" >&2; exit 2; }
  RUN_ROLE=stage1
  RESUME_ARGS=()
  AUDIT_PREVIOUS_ARGS=()
else
  [[ "${PREVIOUS_SEGMENT_ROOT}" == /* && "${PREVIOUS_SEGMENT_AUDIT}" == /* ]] \
    || { echo "Resumed segment requires absolute predecessor evidence" >&2; exit 2; }
  [[ -d "${PREVIOUS_SEGMENT_ROOT}" && -f "${PREVIOUS_SEGMENT_AUDIT}" ]] \
    || { echo "Predecessor evidence is unavailable" >&2; exit 2; }
  RUN_ROLE=resumed
  RESUME_ARGS=(--resume-root "${PREVIOUS_SEGMENT_ROOT}/run")
  AUDIT_PREVIOUS_ARGS=(
    --previous-segment-root "${PREVIOUS_SEGMENT_ROOT}"
    --previous-segment-audit "${PREVIOUS_SEGMENT_AUDIT}"
  )
fi

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
"${PYTHON_BIN}" -c \
  'import json,sys; from pathlib import Path; from freeze_antibody_four_gpu_pretraining_campaign import _campaign_source; gate=json.load(open(sys.argv[2], encoding="ascii")); raise SystemExit(0 if gate.get("campaign_source") == _campaign_source(Path(sys.argv[1])) else 1)' \
  "${SOURCE_ROOT}" "${GATE_PROTOCOL}" \
  || { echo "Campaign source closure differs" >&2; exit 2; }
[[ "$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')" == 4 ]] \
  || { echo "Segment requires exactly four visible GPUs" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader \
  | awk '$0 == "NVIDIA GeForce RTX 5090" {n++} END {print n+0}')" == 4 ]] \
  || { echo "Segment requires four RTX 5090 GPUs" >&2; exit 2; }
ulimit -Sn 4096
[[ "$(ulimit -Sn)" == 4096 ]]

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/generate_runtime_manifest.py" verify \
  --manifest "${RUNTIME_MANIFEST}" \
  --expected-file-sha256 "${RUNTIME_MANIFEST_SHA256}" \
  > "${SEGMENT_ROOT}/runtime_verification.json"
cp "${GATE_PROTOCOL}" "${SEGMENT_ROOT}/protocol.json"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "segment_index=${SEGMENT_INDEX}"
  echo "run_role=${RUN_ROLE}"
  echo "segment_maximum_steps=${SEGMENT_STEPS}"
  echo "campaign_token_limit=${CAMPAIGN_TOKEN_LIMIT}"
  echo "world_size=4"
  echo "claim_boundary=OAS MLM production-pretraining segment only"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
  nvidia-smi topo -m
} > "${SEGMENT_ROOT}/environment.txt"

LOCAL_BASE=${SLURM_TMPDIR:-/tmp}/abf-oas-4g-pretrain-${SLURM_JOB_ID}
mkdir -p "${LOCAL_BASE}/inductor" "${LOCAL_BASE}/triton" \
  "${LOCAL_BASE}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_BASE}/inductor
export TRITON_CACHE_DIR=${LOCAL_BASE}/triton
export TORCH_EXTENSIONS_DIR=${LOCAL_BASE}/torch_extensions
export TORCH_LOGS=recompiles

MONITOR_PID=
stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=
  fi
}
seal_segment() {
  stop_monitor
  if [[ -d "${SEGMENT_ROOT}" && ! -e "${SEGMENT_ROOT}/artifact_sha256.txt" ]]; then
    find "${SEGMENT_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum \
      > "${SEGMENT_ROOT}/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "${SEGMENT_ROOT}" 2>/dev/null || true
}
trap seal_segment EXIT
nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,pstate \
  --format=csv -l 5 > "${SEGMENT_ROOT}/gpu.csv" &
MONITOR_PID=$!

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "${SOURCE_ROOT}/scripts/run_antibody_pretraining_production_gate.py" \
  --production-corpus-root "${CORPUS_ROOT}" \
  --production-corpus-manifest-file-sha256 "${MANIFEST_FILE_SHA256}" \
  --production-corpus-manifest-payload-sha256 "${MANIFEST_PAYLOAD_SHA256}" \
  --production-corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}" \
  --production-corpus-independent-audit "${CORPUS_AUDIT}" \
  --production-corpus-independent-audit-file-sha256 "${CORPUS_AUDIT_FILE_SHA256}" \
  --production-corpus-independent-audit-payload-sha256 "${CORPUS_AUDIT_PAYLOAD_SHA256}" \
  --gate-protocol "${GATE_PROTOCOL}" \
  --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json" \
  --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json" \
  --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json" \
  --output "${SEGMENT_ROOT}/run" \
  --run-role "${RUN_ROLE}" \
  --preset abf_650m \
  --steps "${SEGMENT_STEPS}" \
  --micro-batch 16 \
  --target-tokens 131072 \
  --batch-quantum 256 \
  --run-seed 17 \
  --campaign-token-limit "${CAMPAIGN_TOKEN_LIMIT}" \
  --length-bucket-boundaries 194 \
  --ddp-bucket-cap-mb 25 \
  --maximum-open-arrays 1024 \
  --minimum-soft-open-file-limit 4096 \
  --compile-blocks --disable-compiled-cudagraphs --compile-prewarm-buckets \
  --fused-adamw --pinned-staging --bucketed-padding --length-bucketing \
  --authorization-mode production_pretraining \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "${SEGMENT_ROOT}/training.log"
stop_monitor
unset TORCH_LOGS

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "${SOURCE_ROOT}/scripts/evaluate_antibody_production_checkpoint.py" \
  --production-corpus-root "${CORPUS_ROOT}" \
  --production-corpus-manifest-file-sha256 "${MANIFEST_FILE_SHA256}" \
  --production-corpus-manifest-payload-sha256 "${MANIFEST_PAYLOAD_SHA256}" \
  --production-corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}" \
  --production-corpus-independent-audit "${CORPUS_AUDIT}" \
  --production-corpus-independent-audit-file-sha256 "${CORPUS_AUDIT_FILE_SHA256}" \
  --production-corpus-independent-audit-payload-sha256 "${CORPUS_AUDIT_PAYLOAD_SHA256}" \
  --gate-protocol "${GATE_PROTOCOL}" \
  --run-root "${SEGMENT_ROOT}/run" \
  --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json" \
  --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json" \
  --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json" \
  --output "${SEGMENT_ROOT}/evaluation.json" \
  --preset abf_650m \
  --panel-sample-count 8192 \
  --panel-seed 2026081501 \
  --panel-indices-sha256 cc9219f83bfdaef51e97ee9845f31ba9dd2cbde9863715d6fdfa7a50478197ec \
  --mask-seed 2026081503 --mask-update 0 \
  --micro-batch 16 --maximum-open-arrays 1024 \
  --authorization-mode production_pretraining \
  2>&1 | tee "${SEGMENT_ROOT}/evaluation.log"

date --iso-8601=seconds >> "${SEGMENT_ROOT}/environment.txt"
find "${SEGMENT_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "${SEGMENT_ROOT}/artifact_sha256.txt"
chmod -R a-w "${SEGMENT_ROOT}"
trap - EXIT

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/audit_antibody_four_gpu_pretraining_segment.py" \
  --gate-protocol "${GATE_PROTOCOL}" \
  --segment-root "${SEGMENT_ROOT}" \
  --segment-index "${SEGMENT_INDEX}" \
  --output "${AUDIT_OUTPUT}" \
  "${AUDIT_PREVIOUS_ARGS[@]}"
chmod a-w "${AUDIT_OUTPUT}"
echo "four_gpu_pretraining_segment_audit=PASS"
echo "segment_index=${SEGMENT_INDEX}"
echo "segment_audit=${AUDIT_OUTPUT}"

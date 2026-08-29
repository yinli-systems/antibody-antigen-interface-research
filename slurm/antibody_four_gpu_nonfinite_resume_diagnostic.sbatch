#!/bin/bash
#SBATCH --job-name=abf-nonfinite-diag
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-nonfinite-diag-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-nonfinite-diag-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
CAMPAIGN_ROOT=${ABF_CAMPAIGN_ROOT:?ABF_CAMPAIGN_ROOT is required}
GATE_PROTOCOL=${ABF_GATE_PROTOCOL:?ABF_GATE_PROTOCOL is required}
GATE_PROTOCOL_SHA256=${ABF_GATE_PROTOCOL_SHA256:?required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
RESUME_ROOT=${ABF_RESUME_ROOT:?ABF_RESUME_ROOT is required}
DIAGNOSTIC_SCRIPT=${ABF_DIAGNOSTIC_SCRIPT:?ABF_DIAGNOSTIC_SCRIPT is required}
INCIDENT_ROOT=${ABF_INCIDENT_ROOT:?ABF_INCIDENT_ROOT is required}

for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" "${CAMPAIGN_ROOT}" "${GATE_PROTOCOL}" "${RESUME_ROOT}" "${DIAGNOSTIC_SCRIPT}" "${INCIDENT_ROOT}"; do
  [[ "${path}" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ "${INCIDENT_ROOT}" == "${CAMPAIGN_ROOT}"/incidents/* ]] || { echo "Incident output must remain inside the campaign" >&2; exit 2; }
[[ "${GATE_PROTOCOL_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo "Gate SHA-256 is invalid" >&2; exit 2; }
[[ "$(sha256sum "${GATE_PROTOCOL}" | awk '{print $1}')" == "${GATE_PROTOCOL_SHA256}" ]] || { echo "Campaign gate hash differs" >&2; exit 2; }
[[ -f "${DIAGNOSTIC_SCRIPT}" && ! -L "${DIAGNOSTIC_SCRIPT}" ]] || { echo "Diagnostic runner is unavailable" >&2; exit 2; }
[[ -f "${RESUME_ROOT}/receipt.json" && -d "${RESUME_ROOT}/checkpoint" ]] || { echo "Authenticated predecessor checkpoint is unavailable" >&2; exit 2; }
[[ ! -e "${INCIDENT_ROOT}" ]] || { echo "Refusing to overwrite an incident record" >&2; exit 2; }
mkdir -p "${INCIDENT_ROOT}"

MANIFEST_PATH=${CORPUS_ROOT}/corpus_manifest.json
CORPUS_AUDIT=${CORPUS_ROOT}/independent_audit_receipt.json
MANIFEST_FILE_SHA256=$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')
CORPUS_AUDIT_FILE_SHA256=$(sha256sum "${CORPUS_AUDIT}" | awk '{print $1}')
MANIFEST_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["manifest_sha256"])' "${MANIFEST_PATH}")
CORPUS_AUDIT_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["receipt_sha256"])' "${CORPUS_AUDIT}")

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
ulimit -Sn 4096

[[ "$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')" == 4 ]] || { echo "Diagnostic requires exactly four visible GPUs" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '$0 == "NVIDIA GeForce RTX 4090" {n++} END {print n+0}')" == 4 ]] || { echo "Diagnostic requires four RTX 4090 GPUs" >&2; exit 2; }

LOCAL_BASE=${SLURM_TMPDIR:-/tmp}/abf-nonfinite-diag-${SLURM_JOB_ID}
mkdir -p "${LOCAL_BASE}/inductor" "${LOCAL_BASE}/triton" "${LOCAL_BASE}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_BASE}/inductor
export TRITON_CACHE_DIR=${LOCAL_BASE}/triton
export TORCH_EXTENSIONS_DIR=${LOCAL_BASE}/torch_extensions
export TORCH_LOGS=recompiles

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "diagnostic_runner_sha256=$(sha256sum "${DIAGNOSTIC_SCRIPT}" | awk '{print $1}')"
  echo "resume_root=${RESUME_ROOT}"
  echo "authorization=incident_diagnostic_only"
  echo "test_access=false"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv
} > "${INCIDENT_ROOT}/environment.txt"

set +e
RUN_ARGS=(
  "${DIAGNOSTIC_SCRIPT}"
  --production-corpus-root "${CORPUS_ROOT}"
  --production-corpus-manifest-file-sha256 "${MANIFEST_FILE_SHA256}"
  --production-corpus-manifest-payload-sha256 "${MANIFEST_PAYLOAD_SHA256}"
  --production-corpus-protocol-file-sha256 "${CORPUS_PROTOCOL_FILE_SHA256}"
  --production-corpus-independent-audit "${CORPUS_AUDIT}"
  --production-corpus-independent-audit-file-sha256 "${CORPUS_AUDIT_FILE_SHA256}"
  --production-corpus-independent-audit-payload-sha256 "${CORPUS_AUDIT_PAYLOAD_SHA256}"
  --gate-protocol "${GATE_PROTOCOL}"
  --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json"
  --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json"
  --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json"
  --output "${INCIDENT_ROOT}/run"
  --run-role resumed
  --resume-root "${RESUME_ROOT}"
  --preset abf_650m
  --steps 3000
  --micro-batch 16
  --target-tokens 131072
  --batch-quantum 256
  --run-seed 17
  --campaign-token-limit 30000000000
  --length-bucket-boundaries 194
  --ddp-bucket-cap-mb 25
  --maximum-open-arrays 1024
  --minimum-soft-open-file-limit 4096
  --compile-blocks
  --disable-compiled-cudagraphs
  --compile-prewarm-buckets
  --fused-adamw
  --pinned-staging
  --bucketed-padding
  --length-bucketing
  --authorization-mode production_pretraining
)
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=4 "${RUN_ARGS[@]}" 2>&1 | tee "${INCIDENT_ROOT}/diagnostic.log"
DIAGNOSTIC_EXIT=${PIPESTATUS[0]}
set -e

printf '%s\n' "${DIAGNOSTIC_EXIT}" > "${INCIDENT_ROOT}/diagnostic.exitcode"
find "${INCIDENT_ROOT}" -type f ! -name artifact_sha256.txt -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "${INCIDENT_ROOT}/artifact_sha256.txt"
chmod -R a-w "${INCIDENT_ROOT}"
echo "diagnostic_exit_code=${DIAGNOSTIC_EXIT}"
exit "${DIAGNOSTIC_EXIT}"

#!/bin/bash
#SBATCH --job-name=abf-panel-recovery
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-panel-recovery-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-panel-recovery-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/contact/bin/python3.10}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
CORPUS_ROOT=${ABF_CORPUS_ROOT:?ABF_CORPUS_ROOT is required}
CAMPAIGN_ROOT=${ABF_CAMPAIGN_ROOT:?ABF_CAMPAIGN_ROOT is required}
GATE_PROTOCOL=${ABF_GATE_PROTOCOL:?ABF_GATE_PROTOCOL is required}
GATE_PROTOCOL_SHA256=${ABF_GATE_PROTOCOL_SHA256:?required}
CORPUS_PROTOCOL_FILE_SHA256=${ABF_CORPUS_PROTOCOL_FILE_SHA256:?required}
RUNTIME_MANIFEST=${ABF_RUNTIME_MANIFEST:?ABF_RUNTIME_MANIFEST is required}
RUNTIME_MANIFEST_SHA256=${ABF_RUNTIME_MANIFEST_SHA256:?required}
RECOVERY_PROTOCOL=${ABF_RECOVERY_PROTOCOL:?ABF_RECOVERY_PROTOCOL is required}
RECOVERY_RECEIPT=${ABF_RECOVERY_RECEIPT:?ABF_RECOVERY_RECEIPT is required}
SEGMENT_INDEX=${ABF_SEGMENT_INDEX:?ABF_SEGMENT_INDEX is required}
FAILED_JOB_ID=${ABF_FAILED_JOB_ID:?ABF_FAILED_JOB_ID is required}

[[ ${SLURM_JOB_PARTITION:-} == gpu_4090 ]] \
  || { echo "gpu_4090 is required" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -cx 'NVIDIA GeForce RTX 4090')" == 4 ]] \
  || { echo "four RTX 4090 GPUs are required" >&2; exit 2; }
[[ ${SEGMENT_INDEX} =~ ^[0-9]+$ && ${FAILED_JOB_ID} =~ ^[0-9]+$ ]] \
  || { echo "segment and job identities must be integers" >&2; exit 2; }
for path in "${PYTHON_BIN}" "${SOURCE_ROOT}" "${CORPUS_ROOT}" \
  "${CAMPAIGN_ROOT}" "${GATE_PROTOCOL}" "${RUNTIME_MANIFEST}" \
  "${RECOVERY_PROTOCOL}" "${RECOVERY_RECEIPT}"; do
  [[ ${path} == /* && -e ${path} ]] || { echo "missing absolute path ${path}" >&2; exit 2; }
done
[[ "$(sha256sum "${GATE_PROTOCOL}" | cut -d ' ' -f 1)" == "${GATE_PROTOCOL_SHA256}" ]] \
  || { echo "campaign gate hash differs" >&2; exit 2; }
FAILED_STATE=$(sacct -n -X -j "${FAILED_JOB_ID}" --format=State --parsable2 \
  | head -1 | cut -d '|' -f 1)
[[ ${FAILED_STATE} == FAILED ]] \
  || { echo "source segment job is not FAILED: ${FAILED_STATE}" >&2; exit 2; }

FREEZER=${SOURCE_ROOT}/recovery/freeze_antibody_panel_anchor_recovery.py
AUDITOR=${SOURCE_ROOT}/recovery/audit_antibody_panel_anchor_recovery.py
RECOVERY_SBATCH=${SOURCE_ROOT}/recovery/antibody_panel_anchor_recovery.sbatch
ORIGINAL_SBATCH=${SOURCE_ROOT}/slurm/antibody_four_gpu_pretraining_segment.sbatch
"${PYTHON_BIN}" "${FREEZER}" verify \
  --protocol "${RECOVERY_PROTOCOL}" \
  --receipt "${RECOVERY_RECEIPT}" \
  --source-root "${SOURCE_ROOT}" > /dev/null

printf -v SEGMENT_LABEL 'segment-%06d' "${SEGMENT_INDEX}"
SEGMENT_ROOT=${CAMPAIGN_ROOT}/${SEGMENT_LABEL}
AUDIT_OUTPUT=${CAMPAIGN_ROOT}/audits/${SEGMENT_LABEL}.json
RECOVERY_ROOT=${CAMPAIGN_ROOT}/recoveries/${SEGMENT_LABEL}-job-${SLURM_JOB_ID}
CLOSURE_ROOT=${RECOVERY_ROOT}/segment
[[ -d ${SEGMENT_ROOT} && ! -e ${AUDIT_OUTPUT} && ! -e ${RECOVERY_ROOT} ]] \
  || { echo "recovery paths are not create-only" >&2; exit 2; }
mkdir -p "${CAMPAIGN_ROOT}/recoveries"
mkdir "${RECOVERY_ROOT}"
cp -al "${SEGMENT_ROOT}" "${CLOSURE_ROOT}"
find "${CLOSURE_ROOT}" -type d -exec chmod u+w {} +
[[ -f ${CLOSURE_ROOT}/artifact_sha256.txt ]] \
  || { echo "closure manifest is absent" >&2; exit 2; }
unlink "${CLOSURE_ROOT}/artifact_sha256.txt"
[[ ! -e ${CLOSURE_ROOT}/evaluation.json ]] \
  || { echo "closure evaluation already exists" >&2; exit 2; }

MANIFEST_PATH=${CORPUS_ROOT}/corpus_manifest.json
CORPUS_AUDIT=${CORPUS_ROOT}/independent_audit_receipt.json
MANIFEST_FILE_SHA256=$(sha256sum "${MANIFEST_PATH}" | cut -d ' ' -f 1)
CORPUS_AUDIT_FILE_SHA256=$(sha256sum "${CORPUS_AUDIT}" | cut -d ' ' -f 1)
MANIFEST_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["manifest_sha256"])' \
  "${MANIFEST_PATH}")
CORPUS_AUDIT_PAYLOAD_SHA256=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["receipt_sha256"])' \
  "${CORPUS_AUDIT}")
readarray -t PANEL < <("${PYTHON_BIN}" -c \
  'import json,sys; x=json.load(open(sys.argv[1], encoding="ascii")); e=x["evidence"]; print(e["panel_record_count"]); print(e["panel_seed"]); print(e["corrected_panel_anchor"])' \
  "${RECOVERY_PROTOCOL}")

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts:${SOURCE_ROOT}/recovery
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}/abf-panel-recovery-${SLURM_JOB_ID}
mkdir -p "${LOCAL_BASE}/inductor" "${LOCAL_BASE}/triton" \
  "${LOCAL_BASE}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_BASE}/inductor
export TRITON_CACHE_DIR=${LOCAL_BASE}/triton
export TORCH_EXTENSIONS_DIR=${LOCAL_BASE}/torch_extensions
ulimit -Sn 4096

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
  --run-root "${CLOSURE_ROOT}/run" \
  --model-family "${SOURCE_ROOT}/protocols/antibody_model_family_v1.json" \
  --training-protocol "${SOURCE_ROOT}/protocols/antibody_training_v1.json" \
  --efficiency-protocol "${SOURCE_ROOT}/protocols/antibody_efficiency_gate_v1.json" \
  --output "${CLOSURE_ROOT}/evaluation.json" \
  --preset abf_650m \
  --panel-sample-count "${PANEL[0]}" \
  --panel-seed "${PANEL[1]}" \
  --panel-indices-sha256 "${PANEL[2]}" \
  --mask-seed 2026081503 --mask-update 0 \
  --micro-batch 16 --maximum-open-arrays 1024 \
  --authorization-mode production_pretraining \
  2>&1 | tee "${CLOSURE_ROOT}/recovery_evaluation.log"

find "${CLOSURE_ROOT}" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum \
  > "${CLOSURE_ROOT}/artifact_sha256.txt"
chmod -R a-w "${CLOSURE_ROOT}"

PREVIOUS_ARGS=()
if (( SEGMENT_INDEX > 0 )); then
  printf -v PREVIOUS_LABEL 'segment-%06d' "$((SEGMENT_INDEX - 1))"
  PREVIOUS_ROOT=${CAMPAIGN_ROOT}/${PREVIOUS_LABEL}
  PREVIOUS_AUDIT=${CAMPAIGN_ROOT}/audits/${PREVIOUS_LABEL}.json
  PREVIOUS_ARGS=(
    --previous-segment-root "${PREVIOUS_ROOT}"
    --previous-segment-audit "${PREVIOUS_AUDIT}"
  )
fi
"${PYTHON_BIN}" "${AUDITOR}" \
  --recovery-protocol "${RECOVERY_PROTOCOL}" \
  --recovery-receipt "${RECOVERY_RECEIPT}" \
  --gate-protocol "${GATE_PROTOCOL}" \
  --segment-root "${SEGMENT_ROOT}" \
  --closure-root "${CLOSURE_ROOT}" \
  --segment-index "${SEGMENT_INDEX}" \
  --failed-job-id "${FAILED_JOB_ID}" \
  --output "${AUDIT_OUTPUT}" \
  "${PREVIOUS_ARGS[@]}"
chmod a-w "${AUDIT_OUTPUT}"

NEXT_AUTHORIZED=$("${PYTHON_BIN}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1], encoding="ascii"))["authorization"]["next_segment_submission"]).lower())' \
  "${AUDIT_OUTPUT}")
if [[ ${NEXT_AUTHORIZED} == true ]]; then
  NEXT_INDEX=$((SEGMENT_INDEX + 1))
  NEXT_JOB_ID=$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
    --export="ALL,ABF_SEGMENT_INDEX=${NEXT_INDEX},ABF_PREVIOUS_SEGMENT_ROOT=${SEGMENT_ROOT},ABF_PREVIOUS_SEGMENT_AUDIT=${AUDIT_OUTPUT}" \
    "${ORIGINAL_SBATCH}")
  WATCHDOG_JOB_ID=$(sbatch --parsable --dependency="afternotok:${NEXT_JOB_ID}" \
    --export="ALL,ABF_SEGMENT_INDEX=${NEXT_INDEX},ABF_FAILED_JOB_ID=${NEXT_JOB_ID}" \
    "${RECOVERY_SBATCH}")
  echo "panel_anchor_recovery=PASS"
  echo "segment_audit=${AUDIT_OUTPUT}"
  echo "next_segment_job_id=${NEXT_JOB_ID}"
  echo "next_recovery_watchdog_job_id=${WATCHDOG_JOB_ID}"
else
  CAMPAIGN_COMPLETE=$("${PYTHON_BIN}" -c \
    'import json,sys; print(str(json.load(open(sys.argv[1], encoding="ascii"))["authorization"]["campaign_complete"]).lower())' \
    "${AUDIT_OUTPUT}")
  [[ ${CAMPAIGN_COMPLETE} == true ]] \
    || { echo "recovery audit authorizes neither continuation nor completion" >&2; exit 2; }
  echo "panel_anchor_recovery=PASS"
  echo "campaign_complete=true"
fi

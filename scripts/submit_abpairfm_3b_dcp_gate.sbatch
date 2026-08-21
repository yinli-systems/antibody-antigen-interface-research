#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-dcp
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-dcp-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-dcp-%j.err

set -euo pipefail
umask 077

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly RUNNER="${PROJECT}/scripts/run_abpairfm_3b_dcp_gate.py"
readonly PROTOCOL="${PROJECT}/protocols/abpairfm_3b_dcp_recovery_gate_v3.json"
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly OUTPUT_ROOT="/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/dcp/job-${SLURM_JOB_ID}"

export PYTHONPATH="${PROJECT}/src:${PROJECT}/scripts"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

readonly SOURCE_SHA256="$("${PYTHON}" "${RUNNER}" digest --project-root "${PROJECT}")"
readonly PROTOCOL_SHA256="$(sha256sum "${PROTOCOL}" | awk '{print $1}')"

freeze_evidence() {
  if [[ -d "${OUTPUT_ROOT}" ]]; then
    chmod -R a-w "${OUTPUT_ROOT}" || true
  fi
}
trap freeze_evidence EXIT

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${RUNNER}" run \
  --mode save \
  --project-root "${PROJECT}" \
  --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-size 5678116398 \
  --expected-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 "${PROTOCOL_SHA256}" \
  --output-root "${OUTPUT_ROOT}"

readonly SAVE_RESULT="${OUTPUT_ROOT}/save_result.json"
readonly SAVE_RESULT_SHA256="$(sha256sum "${SAVE_RESULT}" | awk '{print $1}')"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${RUNNER}" run \
  --mode restore \
  --project-root "${PROJECT}" \
  --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-size 5678116398 \
  --expected-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 "${PROTOCOL_SHA256}" \
  --output-root "${OUTPUT_ROOT}" \
  --save-result "${SAVE_RESULT}" \
  --expected-save-result-sha256 "${SAVE_RESULT_SHA256}"

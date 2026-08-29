#!/usr/bin/env bash
#SBATCH -J abpairfm-a1-dev-contact
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-a1-dev-contact-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-a1-dev-contact-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=${ABPAIRFM_STAGE_A1_CONTACT_PROJECT:?missing immutable evaluator project root}
readonly PROTOCOL=${PROJECT}/protocols/abpairfm_stage_a1_development_contact_v1.json
readonly PROTOCOL_SHA256=${ABPAIRFM_STAGE_A1_CONTACT_PROTOCOL_SHA256:?missing protocol SHA-256}
readonly SOURCE_SHA256=${ABPAIRFM_STAGE_A1_CONTACT_SOURCE_SHA256:?missing source SHA-256}
readonly INITIAL_CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly DCP_ROOT=${ABPAIRFM_STAGE_A1_CONTACT_DCP_ROOT:?missing Stage-A1 DCP root}
readonly DCP_GENERATION=${ABPAIRFM_STAGE_A1_CONTACT_DCP_GENERATION:-stage-a1-000000}
readonly DCP_CONTENT_SHA256=${ABPAIRFM_STAGE_A1_CONTACT_DCP_CONTENT_SHA256:?missing DCP content SHA-256}
readonly PROMOTION_RESULT=${ABPAIRFM_STAGE_A1_CONTACT_PROMOTION_RESULT:?missing promotion result}
readonly OUTPUT_ROOT=${ABPAIRFM_STAGE_A1_CONTACT_OUTPUT_ROOT:?missing output root}

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 8 ]]; then
  echo "Stage-A1 development contact requires exactly eight visible GPUs" >&2
  exit 2
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ "${name}" != "NVIDIA GeForce RTX 4090" ]]; then
    echo "Forbidden GPU model: ${name}" >&2
    exit 2
  fi
done

for path in "${PYTHON}" "${PROTOCOL}" "${INITIAL_CHECKPOINT}" \
  "${PROMOTION_RESULT}"; do
  test -f "${path}"
done
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "${PROJECT}/scripts/run_abpairfm_stage_a1_frozen_development_contact.py" \
  --project-root "${PROJECT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 "${PROTOCOL_SHA256}" \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --initial-checkpoint "${INITIAL_CHECKPOINT}" \
  --dcp-root "${DCP_ROOT}" \
  --dcp-generation "${DCP_GENERATION}" \
  --expected-dcp-content-sha256 "${DCP_CONTENT_SHA256}" \
  --promotion-result "${PROMOTION_RESULT}" \
  --output-root "${OUTPUT_ROOT}"

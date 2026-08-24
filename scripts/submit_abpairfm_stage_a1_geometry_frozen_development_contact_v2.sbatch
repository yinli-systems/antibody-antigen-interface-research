#!/usr/bin/env bash
#SBATCH -J abpairfm-a1-geometry-dev-v2
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/evaluation/abpairfm-a1-geometry-dev-v2-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/evaluation/abpairfm-a1-geometry-dev-v2-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=${ABPAIRFM_STAGE_A1_CONTACT_PROJECT:?missing immutable evaluator project root}
readonly PROTOCOL=${PROJECT}/protocols/abpairfm_stage_a1_geometry_development_contact_v2.json
readonly PROTOCOL_SHA256=${ABPAIRFM_STAGE_A1_CONTACT_PROTOCOL_SHA256:?missing protocol SHA-256}
readonly SOURCE_SHA256=${ABPAIRFM_STAGE_A1_CONTACT_SOURCE_SHA256:?missing source SHA-256}
readonly INITIAL_CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly DCP_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a1_geometry_ranking_canary_v17/checkpoints
readonly DCP_GENERATION=pilot-000000
readonly DCP_CONTENT_SHA256=87d1a57148abde8bf47b3aaea4437219ad9c71f0df5d99a53e9de800b48bc479
readonly PROMOTION_RESULT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a1_geometry_ranking_canary_v17/RESULT.json
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/evaluation/abpairfm_stage_a1_geometry_development_contact_v2

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 2 ]]; then
  echo "Stage-A1 geometry development contact requires two visible GPUs" >&2
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
  --nproc_per_node=2 \
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

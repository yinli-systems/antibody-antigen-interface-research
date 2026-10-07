#!/usr/bin/env bash
#SBATCH --job-name=abpairfm-bipartite-head
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-bipartite-head-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-bipartite-head-%A_%a.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly PROJECT=${ROOT}/abpairfm_bipartite_decoder_canary_v1/source
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly INITIAL=${ROOT}/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly STAGE_A0=${ROOT}/results/pretraining/abpairfm_3b_stage_a0_string25m_v1/checkpoints
readonly DATA=${ROOT}/data/abpairfm_3b_v1/pinder_contact_curriculum_v1
readonly CAMPAIGN=${ROOT}/results/pretraining/abpairfm_bipartite_decoder_canary_v1
readonly SOURCE_SHA=78736d49f17eb622bcb2f48cc7f1853563d111b42f5969f5d21c0ebbe2700b51

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    readonly ARM=low_rank_control
    readonly PROTOCOL_NAME=PROTOCOL_PINDER_B_LINK_LOW_RANK_CONTROL_V1.json
    readonly PROTOCOL_SHA=ee4ea5a195975adf2e02f3560372b71bbf0b0f8d265f759d31b9383b97c56c34
    ;;
  1)
    readonly ARM=bipartite_edge_refinement
    readonly PROTOCOL_NAME=PROTOCOL_PINDER_B_LINK_EDGE_REFINEMENT_V1.json
    readonly PROTOCOL_SHA=682f6b5667912acc41c227b1e6e8b2913a8d9e922a7d1729125fe95b2a32bef2
    ;;
  *)
    echo "Unknown decoder arm: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

readonly OUTPUT=${CAMPAIGN}/${ARM}
readonly PROTOCOL=${PROJECT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/${PROTOCOL_NAME}
readonly RESULT=${OUTPUT}/RESULT.json
readonly PROGRESS=${OUTPUT}/PROGRESS.json
readonly CHECKPOINT_ROOT=${OUTPUT}/checkpoints
readonly GPU_LOG=${OUTPUT}/GPU_UTILIZATION.csv

export PYTHONPATH=${PROJECT}/src
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ${#GPU_NAMES[@]} -ne 4 ]]; then
  echo "Decoder canary requires exactly four visible GPUs" >&2
  exit 2
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ ${name} != "NVIDIA GeForce RTX 4090" ]]; then
    echo "Forbidden GPU model: ${name}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT}" "${CHECKPOINT_ROOT}"
for path in "${PROTOCOL}" "${INITIAL}" "${DATA}/B-canary-contact-labels-v1.jsonl.gz" "${DATA}/B_CANARY_CONTACT_LABELS_V1_RECEIPT.json"; do
  test -f "${path}"
done
test ! -e "${RESULT}"
test ! -e "${PROGRESS}"

nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.current.sm,pstate --format=csv --loop=5 > "${GPU_LOG}" &
readonly GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${PROJECT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/run_pinder_b_canary_v1.py" \
  --project-root "${PROJECT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 "${PROTOCOL_SHA}" \
  --expected-source-sha256 "${SOURCE_SHA}" \
  --manifest "${DATA}/B-canary-contact-labels-v1.jsonl.gz" \
  --manifest-receipt "${DATA}/B_CANARY_CONTACT_LABELS_V1_RECEIPT.json" \
  --initial-checkpoint "${INITIAL}" \
  --expected-initial-checkpoint-size 5678116398 \
  --expected-initial-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --input-checkpoint-root "${STAGE_A0}" \
  --input-generation segment-000130 \
  --expected-input-content-sha256 be22eb1889dd4f15b3d2c2ad449cb5084bde91435dd8d4d392f9295d6f10f839 \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --result "${RESULT}" \
  --progress "${PROGRESS}" \
  --expected-world-size 4

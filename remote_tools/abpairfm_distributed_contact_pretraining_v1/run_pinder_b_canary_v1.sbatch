#!/usr/bin/env bash
#SBATCH --job-name=abpairfm-pinder-bcanary
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-pinder-bcanary-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-pinder-bcanary-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly PROJECT=${ROOT}/abpairfm_pinder_b_canary_v1/source
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly INITIAL=${ROOT}/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly STAGE_A0=${ROOT}/results/pretraining/abpairfm_3b_stage_a0_string25m_v1/checkpoints
readonly DATA=${ROOT}/data/abpairfm_3b_v1/pinder_contact_curriculum_v1
readonly OUTPUT=${ROOT}/results/pretraining/abpairfm_pinder_b_canary_v1
readonly PROTOCOL=${PROJECT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/PROTOCOL_PINDER_B_CANARY_V1.json
readonly RESULT=${OUTPUT}/B_CANARY_RESULT_V1.json
readonly PROGRESS=${OUTPUT}/B_CANARY_PROGRESS_V1.json
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
  echo "B-canary requires exactly four visible GPUs" >&2
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
test ! -e "${CHECKPOINT_ROOT}/b-canary-000000"

nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.current.sm,pstate --format=csv --loop=5 > "${GPU_LOG}" &
readonly GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${PROJECT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/run_pinder_b_canary_v1.py" \
  --project-root "${PROJECT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 a7390bf0bd3b2111c55048531a4eb4e950c8c713daeea9e43c5bcd3d94dcc9fb \
  --expected-source-sha256 f9f4e153f6884a70fcaa66bcbf711f1f17cbdbab2f95b76e942c8c182c5ba0fc \
  --manifest "${DATA}/B-canary-contact-labels-v1.jsonl.gz" \
  --manifest-receipt "${DATA}/B_CANARY_CONTACT_LABELS_V1_RECEIPT.json" \
  --initial-checkpoint "${INITIAL}" \
  --expected-initial-checkpoint-size 5678116398 \
  --expected-initial-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --input-checkpoint-root "${STAGE_A0}" \
  --input-generation segment-000127 \
  --expected-input-content-sha256 3f3e79d4b49fa16c3d837a1417ed3c8b677da11e357e5fb9af133edcfa83bd2e \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --result "${RESULT}" \
  --progress "${PROGRESS}" \
  --expected-world-size 4

#!/usr/bin/env bash
#SBATCH --job-name=abpairfm-pinder-b32k-r3
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-pinder-b32k-r3-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-pinder-b32k-r3-%j.err

set -euo pipefail
readonly ROOT=/ssd/scxi253/model_deployment
readonly PROJECT=${ROOT}/abpairfm_pinder_b32k_recovery_v3/source
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly INITIAL=${ROOT}/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly STAGE_A0=${ROOT}/results/pretraining/abpairfm_3b_stage_a0_string25m_v1/checkpoints
readonly DATA=${ROOT}/data/abpairfm_3b_v1/pinder_contact_curriculum_v1
readonly OUTPUT=${ROOT}/results/pretraining/abpairfm_pinder_b32k_v1
readonly PROTOCOL=${PROJECT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/PROTOCOL_PINDER_B32K_RECOVERY_V3.json
readonly RESULT=${OUTPUT}/B32K_RESULT_V1.json
readonly PROGRESS=${OUTPUT}/B32K_PROGRESS_V1.json
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
test "${#GPU_NAMES[@]}" -eq 8
for name in "${GPU_NAMES[@]}"; do test "${name}" = "NVIDIA GeForce RTX 4090"; done
mkdir -p "${OUTPUT}" "${CHECKPOINT_ROOT}"
test ! -e "${RESULT}"; test ! -e "${PROGRESS}"; test ! -e "${CHECKPOINT_ROOT}/b32k-002048"

nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.current.sm,pstate --format=csv --loop=5 > "${GPU_LOG}" &
readonly GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
 "${PROJECT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/run_pinder_b_canary_v1.py" \
 --project-root "${PROJECT}" --protocol "${PROTOCOL}" \
 --expected-protocol-sha256 4022183bff55d12278a200605178ffed7887779b7263ea46e6622495a9b48a25 \
 --expected-source-sha256 64f63449b78629b02f1382562a66f9e231701a73e2532686ef1585fbac765810 \
 --manifest "${DATA}/B32K-contact-labels-recovery-v3.jsonl.gz" \
 --manifest-receipt "${DATA}/B32K_CONTACT_LABELS_RECOVERY_V3_RECEIPT.json" \
 --initial-checkpoint "${INITIAL}" --expected-initial-checkpoint-size 5678116398 \
 --expected-initial-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
 --input-checkpoint-root "${STAGE_A0}" --input-generation segment-000129 \
 --expected-input-content-sha256 71c6b345d251a350b2299884f7576b989413b40ec7f40fea31eb2f30d1c00e2c \
 --checkpoint-root "${CHECKPOINT_ROOT}" --result "${RESULT}" --progress "${PROGRESS}" \
 --expected-world-size 8

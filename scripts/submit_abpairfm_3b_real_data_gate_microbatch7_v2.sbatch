#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-real200-mb7
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-real200-mb7-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-real200-mb7-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/real_gate
readonly MANIFEST=${DATA_ROOT}/string_high_direct_seed1701_22400_v1.jsonl.zst
readonly MANIFEST_RECEIPT=${DATA_ROOT}/string_high_direct_seed1701_22400_v1.receipt.json
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/real_gate
readonly OUTPUT=${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-8gpu-hsdp2x4-microbatch7-200update-v2.json
readonly PROGRESS=${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-microbatch7-progress-v2.json

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

mkdir -p "${OUTPUT_ROOT}"
for path in "${MANIFEST}" "${MANIFEST_RECEIPT}"; do
  test -f "${path}"
done
test ! -e "${OUTPUT}"
test ! -e "${PROGRESS}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "${PROJECT}/scripts/run_abpairfm_3b_real_data_gate_v2.py" \
  --checkpoint "${CHECKPOINT}" \
  --expected-size 5678116398 \
  --expected-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --manifest "${MANIFEST}" \
  --manifest-receipt "${MANIFEST_RECEIPT}" \
  --output "${OUTPUT}" \
  --progress "${PROGRESS}" \
  --expected-world-size 8 \
  --shard-group-size 4 \
  --microbatch-pairs 7 \
  --accumulation-steps 2 \
  --updates 200 \
  --maximum-tokens 1024 \
  --learning-rate 1e-6 \
  --weight-decay 0.01 \
  --seed 1701 \
  --progress-interval 10

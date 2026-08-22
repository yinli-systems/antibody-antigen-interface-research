#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-ab8-200
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=00:45:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-ab8-200-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-ab8-200-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/systems
readonly SEQUENCE_TOKENS=1024
readonly WARMUP_STEPS=10
readonly MEASURED_STEPS=200
readonly ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"

IFS=',' read -r -a CUDA_DEVICE_IDS <<< "${ALLOCATED_CUDA_VISIBLE_DEVICES}"
if [[ "${#CUDA_DEVICE_IDS[@]}" -ne 8 ]]; then
  echo "Expected exactly eight allocated GPUs, observed ${ALLOCATED_CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
readonly BASELINE_CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_IDS[0]},${CUDA_DEVICE_IDS[1]},${CUDA_DEVICE_IDS[2]},${CUDA_DEVICE_IDS[3]}"
readonly BASELINE_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-same-node-4gpu-fsdp4-${SEQUENCE_TOKENS}token-keep-gathered-200step-v1.json"
readonly CANDIDATE_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-same-node-8gpu-hsdp2x4-${SEQUENCE_TOKENS}token-keep-gathered-200step-v1.json"
readonly COMPARISON_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-same-node-4v8gpu-scaling-200step-v1.json"

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

mkdir -p "${OUTPUT_ROOT}"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES="${BASELINE_CUDA_VISIBLE_DEVICES}" "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${PROJECT}/scripts/run_abpairfm_3b_fsdp2_smoke.py" \
  --checkpoint "${CHECKPOINT}" \
  --expected-size 5678116398 \
  --expected-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --expected-world-size 4 \
  --shard-group-size 0 \
  --sequence-tokens "${SEQUENCE_TOKENS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measured-steps "${MEASURED_STEPS}" \
  --no-reshard-after-forward \
  --output "${BASELINE_OUTPUT}"

CUDA_VISIBLE_DEVICES="${ALLOCATED_CUDA_VISIBLE_DEVICES}" "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "${PROJECT}/scripts/run_abpairfm_3b_fsdp2_smoke.py" \
  --checkpoint "${CHECKPOINT}" \
  --expected-size 5678116398 \
  --expected-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --expected-world-size 8 \
  --shard-group-size 4 \
  --sequence-tokens "${SEQUENCE_TOKENS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measured-steps "${MEASURED_STEPS}" \
  --no-reshard-after-forward \
  --output "${CANDIDATE_OUTPUT}"

"${PYTHON}" "${PROJECT}/scripts/summarize_abpairfm_3b_scaling_ab.py" \
  --baseline "${BASELINE_OUTPUT}" \
  --candidate "${CANDIDATE_OUTPUT}" \
  --minimum-speedup 1.4 \
  --maximum-p95-to-median 1.2 \
  --maximum-memory-fraction 0.92 \
  --output "${COMPARISON_OUTPUT}"

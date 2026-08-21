#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-sys4g
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-sys4g-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-sys4g-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/systems
readonly WORLD_SIZE="${ABPAIRFM_WORLD_SIZE:-4}"
readonly SHARD_GROUP_SIZE="${ABPAIRFM_SHARD_GROUP_SIZE:-0}"
readonly SEQUENCE_TOKENS="${ABPAIRFM_SEQUENCE_TOKENS:-128}"
readonly WARMUP_STEPS="${ABPAIRFM_WARMUP_STEPS:-1}"
readonly MEASURED_STEPS="${ABPAIRFM_MEASURED_STEPS:-2}"
readonly RESHARD_AFTER_FORWARD="${ABPAIRFM_RESHARD_AFTER_FORWARD:-1}"

case "${RESHARD_AFTER_FORWARD}" in
  1) readonly RESHARD_LABEL=reshard; readonly RESHARD_ARGUMENT=--reshard-after-forward ;;
  0) readonly RESHARD_LABEL=keep-gathered; readonly RESHARD_ARGUMENT=--no-reshard-after-forward ;;
  *) echo "ABPAIRFM_RESHARD_AFTER_FORWARD must be 0 or 1" >&2; exit 2 ;;
esac

readonly OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-${WORLD_SIZE}gpu-${SEQUENCE_TOKENS}token-${RESHARD_LABEL}-smoke-v1.json"

case "${WORLD_SIZE}" in
  4|8) ;;
  *) echo "ABPAIRFM_WORLD_SIZE must be 4 or 8" >&2; exit 2 ;;
esac

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS="${ABPAIRFM_TRACE_LAYERS:-0}"

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${WORLD_SIZE}" \
  "${PROJECT}/scripts/run_abpairfm_3b_fsdp2_smoke.py" \
  --checkpoint "${CHECKPOINT}" \
  --expected-size 5678116398 \
  --expected-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --expected-world-size "${WORLD_SIZE}" \
  --shard-group-size "${SHARD_GROUP_SIZE}" \
  --sequence-tokens "${SEQUENCE_TOKENS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measured-steps "${MEASURED_STEPS}" \
  "${RESHARD_ARGUMENT}" \
  --output "${OUTPUT}"

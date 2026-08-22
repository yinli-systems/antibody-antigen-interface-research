#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-mbatch
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-mbatch-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-mbatch-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/systems
readonly WORLD_SIZE=8
readonly SHARD_GROUP_SIZE=4
readonly SEQUENCE_TOKENS=1024
readonly MICROBATCH_PAIRS="${ABPAIRFM_MICROBATCH_PAIRS:-2}"
readonly WARMUP_STEPS="${ABPAIRFM_WARMUP_STEPS:-5}"
readonly MEASURED_STEPS="${ABPAIRFM_MEASURED_STEPS:-20}"
readonly BASELINE_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-8gpu-hsdp2x4-${SEQUENCE_TOKENS}token-microbatch1-ab-v1.json"
readonly CANDIDATE_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-8gpu-hsdp2x4-${SEQUENCE_TOKENS}token-microbatch${MICROBATCH_PAIRS}-ab-v1.json"
readonly COMPARISON_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-8gpu-hsdp2x4-${SEQUENCE_TOKENS}token-microbatch1v${MICROBATCH_PAIRS}-ab-v1.json"

case "${MICROBATCH_PAIRS}" in
  2|3|4|6|8) ;;
  *) echo "ABPAIRFM_MICROBATCH_PAIRS must be 2, 3, 4, 6, or 8" >&2; exit 2 ;;
esac

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

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
  --microbatch-pairs 1 \
  --warmup-steps "${WARMUP_STEPS}" \
  --measured-steps "${MEASURED_STEPS}" \
  --no-reshard-after-forward \
  --output "${BASELINE_OUTPUT}"

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
  --microbatch-pairs "${MICROBATCH_PAIRS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measured-steps "${MEASURED_STEPS}" \
  --no-reshard-after-forward \
  --output "${CANDIDATE_OUTPUT}"

"${PYTHON}" "${PROJECT}/scripts/summarize_abpairfm_3b_microbatch_ab.py" \
  --baseline "${BASELINE_OUTPUT}" \
  --candidate "${CANDIDATE_OUTPUT}" \
  --minimum-throughput-speedup 1.1 \
  --maximum-p95-to-median 1.2 \
  --maximum-memory-fraction 0.92 \
  --output "${COMPARISON_OUTPUT}"

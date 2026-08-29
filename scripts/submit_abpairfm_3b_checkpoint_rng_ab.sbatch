#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-rng
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-rng-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-rng-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/systems
readonly COMMON_LABEL="8gpu-hsdp2x4-1024token-microbatch7-checkpoint-rng-ab-v1"
readonly BASELINE_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-${COMMON_LABEL}-preserve.json"
readonly CANDIDATE_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-${COMMON_LABEL}-no-preserve.json"
readonly COMPARISON_OUTPUT="${OUTPUT_ROOT}/job-${SLURM_JOB_ID}-${COMMON_LABEL}-comparison.json"

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

mkdir -p "${OUTPUT_ROOT}"

run_arm() {
  local preserve_rng_state="$1"
  local output="$2"
  local rng_option=--activation-checkpoint-preserve-rng-state
  if [[ "${preserve_rng_state}" == "false" ]]; then
    rng_option=--no-activation-checkpoint-preserve-rng-state
  fi
  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    "${PROJECT}/scripts/run_abpairfm_3b_fsdp2_smoke.py" \
    --checkpoint "${CHECKPOINT}" \
    --expected-size 5678116398 \
    --expected-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
    --expected-world-size 8 \
    --shard-group-size 4 \
    --sequence-tokens 1024 \
    --microbatch-pairs 7 \
    --warmup-steps 5 \
    --measured-steps 20 \
    --no-reshard-after-forward \
    --activation-checkpoint-policy full \
    "${rng_option}" \
    --output "${output}"
}

run_arm true "${BASELINE_OUTPUT}"
run_arm false "${CANDIDATE_OUTPUT}"

"${PYTHON}" "${PROJECT}/scripts/summarize_abpairfm_3b_checkpoint_rng_ab.py" \
  --baseline "${BASELINE_OUTPUT}" \
  --candidate "${CANDIDATE_OUTPUT}" \
  --minimum-throughput-speedup 1.02 \
  --maximum-p95-to-median 1.2 \
  --maximum-memory-fraction 0.92 \
  --maximum-loss-absolute-delta 5e-4 \
  --maximum-gradient-norm-relative-delta 1e-3 \
  --output "${COMPARISON_OUTPUT}"

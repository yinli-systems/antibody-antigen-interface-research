#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-a1-geometry-head-v16
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a1-geometry-head-v16-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a1-geometry-head-v16-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=${ABPAIRFM_STAGE_A1_PROJECT:?missing immutable Stage-A1 project root}
readonly SOURCE_SHA256=${ABPAIRFM_STAGE_A1_SOURCE_SHA256:?missing frozen source SHA-256}
readonly PROTOCOL=${PROJECT}/protocols/abpairfm_stage_a1_geometry_head_only_canary_v16.json
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a1_geometry_head_only_canary_v16

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 4 ]]; then
  echo "Stage-A1 geometry head-only canary requires exactly four visible GPUs" >&2
  exit 2
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ "${name}" != "NVIDIA GeForce RTX 4090" ]]; then
    echo "Forbidden GPU model: ${name}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"
test ! -e "${OUTPUT_ROOT}/RESULT.json"
test ! -e "${OUTPUT_ROOT}/PROGRESS.json"
test ! -e "${OUTPUT_ROOT}/checkpoints/pilot-000000"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${PROJECT}/scripts/run_abpairfm_stage_a1_multitask_pilot.py" \
  --project-root "${PROJECT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 e4b90b873d1a1f3fcc4c4ab2bd75fce4e45d4539e1b1127c21fddc88476ae50d \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --manifest /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_distance_bins_v1/STAGE_A1_DISTANCE_BIN_MANIFEST.jsonl.zst \
  --manifest-receipt /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_distance_bins_v1/STAGE_A1_DISTANCE_BIN_MANIFEST_RECEIPT.json \
  --initial-checkpoint /ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt \
  --expected-initial-checkpoint-size 5678116398 \
  --expected-initial-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --input-checkpoint-root /ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a0_string25m_v1/checkpoints \
  --input-generation segment-000031 \
  --expected-input-content-sha256 73fd08e73bf41118ece0e657059c5211ce5930c1c651e3cfd219554692edf08b \
  --checkpoint-root "${OUTPUT_ROOT}/checkpoints" \
  --result "${OUTPUT_ROOT}/RESULT.json" \
  --progress "${OUTPUT_ROOT}/PROGRESS.json" \
  --expected-world-size 4 \
  --shard-group-size 4 \
  --maximum-tokens 1024 \
  --efficiency-microbatch-pairs-per-rank 48 \
  --efficiency-activation-checkpoint-mode off \
  --progress-interval 10 \
  --rank-phase-telemetry off

#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-a1-ac-v8
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a1-ac-v8-%A_%a.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a1-ac-v8-%A_%a.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=${ABPAIRFM_STAGE_A1_PROJECT:?missing immutable Stage-A1 project root}
readonly PROTOCOL=${PROJECT}/protocols/abpairfm_stage_a1_efficiency_ac_gate_v8.json
readonly PROTOCOL_SHA256=4b53c2354f1c6e21831a72de3e34707475b7ecc4caaee147a9931e6f84cc7976
readonly MANIFEST=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_pilot_v4/STAGE_A1_PILOT_MANIFEST.jsonl.zst
readonly MANIFEST_RECEIPT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_pilot_v4/STAGE_A1_PILOT_MANIFEST_RECEIPT.json
readonly INITIAL_CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly INPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a0_string25m_v1/checkpoints
readonly OUTPUT_BASE=${ABPAIRFM_STAGE_A1_OUTPUT_ROOT:-/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a1_efficiency_ac_gate_v8}
readonly SOURCE_SHA256=${ABPAIRFM_STAGE_A1_SOURCE_SHA256:?missing frozen Stage-A1 source SHA-256}
readonly AC_MODES=(selective_attention off)
readonly AC_MODE=${AC_MODES[${SLURM_ARRAY_TASK_ID}]}
readonly OUTPUT_ROOT=${OUTPUT_BASE}/${AC_MODE}

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 8 ]]; then
  echo "Stage-A1 activation-checkpoint gate requires exactly eight visible GPUs" >&2
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
  --nproc_per_node=8 \
  "${PROJECT}/scripts/run_abpairfm_stage_a1_multitask_pilot.py" \
  --project-root "${PROJECT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 "${PROTOCOL_SHA256}" \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --manifest "${MANIFEST}" \
  --manifest-receipt "${MANIFEST_RECEIPT}" \
  --initial-checkpoint "${INITIAL_CHECKPOINT}" \
  --expected-initial-checkpoint-size 5678116398 \
  --expected-initial-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --input-checkpoint-root "${INPUT_ROOT}" \
  --input-generation segment-000024 \
  --expected-input-content-sha256 7dbde1798181967cc5406dae6eed99a58d8d80c097f16c684d1ce731e5d16311 \
  --checkpoint-root "${OUTPUT_ROOT}/checkpoints" \
  --result "${OUTPUT_ROOT}/RESULT.json" \
  --progress "${OUTPUT_ROOT}/PROGRESS.json" \
  --expected-world-size 8 \
  --shard-group-size 4 \
  --maximum-tokens 64 \
  --efficiency-microbatch-pairs-per-rank 64 \
  --efficiency-activation-checkpoint-mode "${AC_MODE}" \
  --progress-interval 1

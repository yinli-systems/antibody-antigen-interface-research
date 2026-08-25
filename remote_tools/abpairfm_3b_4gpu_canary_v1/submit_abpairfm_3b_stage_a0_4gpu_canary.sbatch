#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-a0-4gpu-canary
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH --exclude=wqd10naf05g3
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a0-4gpu-canary-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a0-4gpu-canary-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_4gpu_canary_v1/source
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a0_string25m_v1
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a0_4gpu_canary_v1
readonly PROTOCOL=${PROJECT}/protocols/abpairfm_3b_stage_a0_campaign_v1.json
readonly PROTOCOL_SHA256=a783f3b20516fbb2c85dba27833d3125683184ae2be608dfcfdbf91e58a3747b
readonly SEGMENT_INDEX=${ABPAIRFM_SEGMENT_INDEX:-74}
readonly SEGMENT=$(printf 'segment-%06d' "${SEGMENT_INDEX}")
readonly MANIFEST=${DATA_ROOT}/string_seed1701_${SEGMENT}_22400_v1.jsonl.zst
readonly MANIFEST_RECEIPT=${DATA_ROOT}/string_seed1701_${SEGMENT}_22400_v1.receipt.json
readonly RESULT=${OUTPUT_ROOT}/results/${SEGMENT}.json
readonly PROGRESS=${OUTPUT_ROOT}/progress/${SEGMENT}.json
readonly DCP_ROOT=${OUTPUT_ROOT}/checkpoints

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
  echo "The frozen canary requires exactly four visible GPUs" >&2
  exit 2
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ "${name}" != "NVIDIA GeForce RTX 4090" ]]; then
    echo "Forbidden GPU model: ${name}" >&2
    exit 2
  fi
done
RESUME_ARGUMENTS=()
: "${ABPAIRFM_INPUT_CHECKPOINT_ROOT:?canary requires checkpoint root}"
: "${ABPAIRFM_INPUT_GENERATION:?canary requires checkpoint generation}"
: "${ABPAIRFM_EXPECTED_INPUT_CONTENT_SHA256:?canary requires checkpoint content SHA-256}"
RESUME_ARGUMENTS=(
  --input-checkpoint-root "${ABPAIRFM_INPUT_CHECKPOINT_ROOT}"
  --input-generation "${ABPAIRFM_INPUT_GENERATION}"
  --expected-input-content-sha256 "${ABPAIRFM_EXPECTED_INPUT_CONTENT_SHA256}"
)
mkdir -p "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/progress" "${DCP_ROOT}"
for path in "${MANIFEST}" "${MANIFEST_RECEIPT}" "${PROTOCOL}" "${CHECKPOINT}"; do
  test -f "${path}"
done
test ! -e "${RESULT}"
test ! -e "${PROGRESS}"
test ! -e "${DCP_ROOT}/${SEGMENT}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${PROJECT}/scripts/run_abpairfm_3b_stage_a0_segment.py" \
  --project-root "${PROJECT}" \
  --campaign-protocol "${PROTOCOL}" \
  --expected-campaign-protocol-sha256 "${PROTOCOL_SHA256}" \
  --expected-source-sha256 e4c27894b2eac1e6db57ab39d41ed85a32c65ce38d74be45087b05f7f983dad4 \
  --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-size 5678116398 \
  --expected-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --manifest "${MANIFEST}" \
  --manifest-receipt "${MANIFEST_RECEIPT}" \
  --segment-index "${SEGMENT_INDEX}" \
  --checkpoint-root "${DCP_ROOT}" \
  --result "${RESULT}" \
  --progress "${PROGRESS}" \
  "${RESUME_ARGUMENTS[@]}" \
  --expected-world-size 4 \
  --shard-group-size 4 \
  --microbatch-pairs 7 \
  --accumulation-steps 4 \
  --updates 20 \
  --maximum-tokens 1024 \
  --maximum-learning-rate 1e-6 \
  --warmup-residue-tokens 100000000 \
  --target-residue-tokens 5000000000 \
  --weight-decay 0.01 \
  --seed 1701 \
  --progress-interval 2 \
  --gpu-bf16-peak-flops 165200000000000

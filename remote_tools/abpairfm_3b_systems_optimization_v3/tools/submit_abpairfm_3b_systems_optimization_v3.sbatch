#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-systems-v3
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --exclude=wqd10naf05g3
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-systems-v3-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-systems-v3-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_systems_optimization_v3/source
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a0_string25m_v1
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a0_4gpu_systems_canary_v3
readonly CAMPAIGN_PROTOCOL=${PROJECT}/protocols/abpairfm_3b_stage_a0_campaign_v1.json
readonly CAMPAIGN_PROTOCOL_SHA256=a783f3b20516fbb2c85dba27833d3125683184ae2be608dfcfdbf91e58a3747b
readonly SEGMENT_INDEX=74
readonly SEGMENT=segment-000074
readonly MANIFEST=${DATA_ROOT}/string_seed1701_${SEGMENT}_22400_v1.jsonl.zst
readonly MANIFEST_RECEIPT=${DATA_ROOT}/string_seed1701_${SEGMENT}_22400_v1.receipt.json
readonly RESULT=${OUTPUT_ROOT}/results/${SEGMENT}.json
readonly PROGRESS=${OUTPUT_ROOT}/progress/${SEGMENT}.json
readonly DCP_ROOT=${OUTPUT_ROOT}/checkpoints

export PYTHONPATH=${PROJECT}/src
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ${#GPU_NAMES[@]} -ne 4 ]]; then
  echo "The frozen systems canary requires exactly four visible GPUs" >&2
  exit 2
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ ${name} != "NVIDIA GeForce RTX 4090" ]]; then
    echo "Forbidden GPU model: ${name}" >&2
    exit 2
  fi
done

readonly INPUT_CHECKPOINT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a0_string25m_v1/checkpoints
readonly INPUT_GENERATION=segment-000073
readonly INPUT_CONTENT_SHA256=38a0987985f119afb374f2062e407bc9c5006f3f7ad66711744086d9561810b1

mkdir -p ${OUTPUT_ROOT}/results ${OUTPUT_ROOT}/progress ${DCP_ROOT}
for path in ${MANIFEST} ${MANIFEST_RECEIPT} ${CAMPAIGN_PROTOCOL} ${CHECKPOINT}; do
  test -f ${path}
done
test ! -e ${RESULT}
test ! -e ${PROGRESS}
test ! -e ${DCP_ROOT}/${SEGMENT}

${PYTHON} -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  ${PROJECT}/scripts/run_abpairfm_3b_stage_a0_segment.py \
  --project-root ${PROJECT} \
  --campaign-protocol ${CAMPAIGN_PROTOCOL} \
  --expected-campaign-protocol-sha256 ${CAMPAIGN_PROTOCOL_SHA256} \
  --expected-source-sha256 2d741e7aa303f891729179b432a8b3f17a0363f0c3bdb00cbf776df318c018cf \
  --checkpoint ${CHECKPOINT} \
  --expected-checkpoint-size 5678116398 \
  --expected-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --manifest ${MANIFEST} \
  --manifest-receipt ${MANIFEST_RECEIPT} \
  --segment-index ${SEGMENT_INDEX} \
  --checkpoint-root ${DCP_ROOT} \
  --result ${RESULT} \
  --progress ${PROGRESS} \
  --input-checkpoint-root ${INPUT_CHECKPOINT_ROOT} \
  --input-generation ${INPUT_GENERATION} \
  --expected-input-content-sha256 ${INPUT_CONTENT_SHA256} \
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
  --gpu-bf16-peak-flops 165200000000000 \
  --length-bucketed-microbatches

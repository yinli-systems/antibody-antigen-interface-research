#!/usr/bin/env bash
#SBATCH -J abpairfm-transfer-probe
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-transfer-probe-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-transfer-probe-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=${ABPAIRFM_PROBE_PROJECT:-/ssd/scxi253/model_deployment/abpairfm_3b_v1/source}
readonly CHECKPOINT=${ABPAIRFM_PROBE_INITIAL_CHECKPOINT:-/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt}
readonly PANEL=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/development_anchor/development_pairs_v1.jsonl
readonly PANEL_RECEIPT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/development_anchor/development_pairs_v1.receipt.json
readonly PROTOCOL=${ABPAIRFM_PROBE_PROTOCOL:-${PROJECT}/protocols/abpairfm_checkpoint_transfer_probe_v1.json}
readonly MODEL_VARIANT=${ABPAIRFM_PROBE_MODEL_VARIANT:?missing model variant}
readonly OUTPUT_ROOT=${ABPAIRFM_PROBE_OUTPUT_ROOT:?missing output root}
readonly PROTOCOL_SHA256=${ABPAIRFM_PROBE_PROTOCOL_SHA256:?missing protocol SHA-256}
readonly SOURCE_SHA256=${ABPAIRFM_PROBE_SOURCE_SHA256:?missing source SHA-256}
readonly SMOKE_ANCHORS=${ABPAIRFM_PROBE_SMOKE_ANCHORS:-0}
readonly CHECKPOINT_SIZE=${ABPAIRFM_PROBE_INITIAL_CHECKPOINT_SIZE:-5678116398}
readonly CHECKPOINT_SHA256=${ABPAIRFM_PROBE_INITIAL_CHECKPOINT_SHA256:-7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500}
readonly LAYER_COUNT=${ABPAIRFM_PROBE_LAYER_COUNT:-36}
readonly EMBED_DIMENSION=${ABPAIRFM_PROBE_EMBED_DIMENSION:-2560}
readonly HEAD_COUNT=${ABPAIRFM_PROBE_HEAD_COUNT:-40}
readonly FFN_DIMENSION=${ABPAIRFM_PROBE_FFN_DIMENSION:-10240}
readonly MAXIMUM_BATCH_SIZE=${ABPAIRFM_PROBE_MAXIMUM_BATCH_SIZE:-7}
readonly MAXIMUM_ATTENTION_CELLS=${ABPAIRFM_PROBE_MAXIMUM_ATTENTION_CELLS:-7340032}

export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 8 ]]; then
  echo "Transfer probe requires exactly eight visible GPUs" >&2
  exit 2
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ "${name}" != "NVIDIA GeForce RTX 4090" ]]; then
    echo "Forbidden GPU model: ${name}" >&2
    exit 2
  fi
done

for path in "${CHECKPOINT}" "${PANEL}" "${PANEL_RECEIPT}" "${PROTOCOL}"; do
  test -f "${path}"
done
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

DCP_ARGUMENTS=()
if [[ "${MODEL_VARIANT}" == "stage_a0_3b" ]]; then
  : "${ABPAIRFM_PROBE_CHECKPOINT_TARGET:?missing checkpoint target}"
  : "${ABPAIRFM_PROBE_DCP_ROOT:?missing DCP root}"
  : "${ABPAIRFM_PROBE_DCP_GENERATION:?missing DCP generation}"
  : "${ABPAIRFM_PROBE_DCP_CONTENT_SHA256:?missing DCP content SHA-256}"
  DCP_ARGUMENTS=(
    --checkpoint-target-residue-tokens "${ABPAIRFM_PROBE_CHECKPOINT_TARGET}"
    --dcp-root "${ABPAIRFM_PROBE_DCP_ROOT}"
    --dcp-generation "${ABPAIRFM_PROBE_DCP_GENERATION}"
    --expected-dcp-content-sha256 "${ABPAIRFM_PROBE_DCP_CONTENT_SHA256}"
  )
elif [[ "${MODEL_VARIANT}" != "base_esm2_3b" && "${MODEL_VARIANT}" != "base_esm2_650m" ]]; then
  echo "Unsupported model variant: ${MODEL_VARIANT}" >&2
  exit 2
fi

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "${PROJECT}/scripts/run_abpairfm_checkpoint_transfer_probe.py" \
  --model-variant "${MODEL_VARIANT}" \
  --project-root "${PROJECT}" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 "${PROTOCOL_SHA256}" \
  --expected-source-sha256 "${SOURCE_SHA256}" \
  --initial-checkpoint "${CHECKPOINT}" \
  --expected-initial-checkpoint-size "${CHECKPOINT_SIZE}" \
  --expected-initial-checkpoint-sha256 "${CHECKPOINT_SHA256}" \
  --layer-count "${LAYER_COUNT}" \
  --embed-dimension "${EMBED_DIMENSION}" \
  --head-count "${HEAD_COUNT}" \
  --ffn-dimension "${FFN_DIMENSION}" \
  --pair-panel "${PANEL}" \
  --pair-panel-receipt "${PANEL_RECEIPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --maximum-batch-size "${MAXIMUM_BATCH_SIZE}" \
  --maximum-attention-cells "${MAXIMUM_ATTENTION_CELLS}" \
  --smoke-anchor-count "${SMOKE_ANCHORS}" \
  "${DCP_ARGUMENTS[@]}"

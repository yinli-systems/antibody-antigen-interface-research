#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-a0-manifest
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:30:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a0-manifest-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a0-manifest-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a0_string25m_v1
readonly PROTOCOL=${PROJECT}/protocols/abpairfm_3b_stage_a0_campaign_v1.json
readonly PROTOCOL_SHA256=e96bafe3b884a4304c17098592c9c4944ade86c7895925ebf349030256322ba5
readonly SEGMENT_INDEX=${ABPAIRFM_SEGMENT_INDEX:-0}
readonly SEGMENT=$(printf 'segment-%06d' "${SEGMENT_INDEX}")
readonly MANIFEST=${OUTPUT_ROOT}/string_seed1701_${SEGMENT}_22400_v1.jsonl.zst
readonly RECEIPT=${OUTPUT_ROOT}/string_seed1701_${SEGMENT}_22400_v1.receipt.json
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden manifest allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
if [[ "${SEGMENT_INDEX}" -lt 0 ]]; then
  echo "ABPAIRFM_SEGMENT_INDEX must be non-negative" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
test ! -e "${MANIFEST}"
test ! -e "${RECEIPT}"

"${PYTHON}" "${PROJECT}/scripts/build_abpairfm_3b_stage_a0_segment.py" \
  --pairs "${ROOT}/physical_combined700_cluster50_devclean30_scale_v2.tsv.zst" \
  --pair-receipt "${ROOT}/physical_combined700_cluster50_devclean30_scale_v2.receipt.json" \
  --endpoint-fasta "${ROOT}/cluster_assets_high700_v1/endpoint_sequences.fasta.zst" \
  --cluster-receipt "${ROOT}/cluster_assets_high700_v1/cluster_assets.receipt.json" \
  --protocol "${PROTOCOL}" \
  --expected-protocol-sha256 "${PROTOCOL_SHA256}" \
  --output "${MANIFEST}" \
  --receipt "${RECEIPT}" \
  --segment-index "${SEGMENT_INDEX}" \
  --record-count 22400 \
  --candidate-headroom 22400 \
  --minimum-residues-per-chain 20 \
  --threads 6

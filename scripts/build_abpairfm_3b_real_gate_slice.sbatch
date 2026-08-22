#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-real-slice
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-real-slice-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-real-slice-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/real_gate
readonly RECORD_COUNT=${ABPAIRFM_REAL_GATE_RECORD_COUNT:-12800}
readonly OUTPUT=${OUTPUT_ROOT}/string_high_direct_seed1701_${RECORD_COUNT}_v1.jsonl.zst
readonly RECEIPT=${OUTPUT_ROOT}/string_high_direct_seed1701_${RECORD_COUNT}_v1.receipt.json
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden real-gate slice allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
case "${RECORD_COUNT}" in
  12800|25600|51200) ;;
  *) echo "ABPAIRFM_REAL_GATE_RECORD_COUNT must be 12800, 25600, or 51200" >&2; exit 2 ;;
esac
mkdir -p "${OUTPUT_ROOT}"
test ! -e "${OUTPUT}"
test ! -e "${RECEIPT}"

"${PYTHON}" "${PROJECT}/scripts/build_abpairfm_3b_real_gate_slice.py" \
  --pairs "${ROOT}/direct_physical_combined700_cluster50_expanded_devclean_v1.tsv.zst" \
  --pair-receipt "${ROOT}/direct_physical_combined700_cluster50_expanded_devclean_v1.receipt.json" \
  --endpoint-fasta "${ROOT}/cluster_assets_expanded_v1/endpoint_sequences.fasta.zst" \
  --cluster-receipt "${ROOT}/direct_physical_combined700_cluster50_expanded_v1.receipt.json" \
  --output "${OUTPUT}" \
  --receipt "${RECEIPT}" \
  --seed 1701 \
  --record-count "${RECORD_COUNT}" \
  --candidate-multiplier 2 \
  --minimum-residues-per-chain 20 \
  --threads 6

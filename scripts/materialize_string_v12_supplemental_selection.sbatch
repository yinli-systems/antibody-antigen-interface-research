#!/usr/bin/env bash
#SBATCH -J abpairfm-string-supp-materialize
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-supp-materialize-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-supp-materialize-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly OUTPUT=${ROOT}/direct_physical_supplemental_selected_cluster50_expanded_devclean_v1.tsv.zst
readonly RECEIPT=${ROOT}/direct_physical_supplemental_selected_cluster50_expanded_devclean_v1.receipt.json
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden supplemental materialization allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test ! -e "${OUTPUT}"
test ! -e "${RECEIPT}"

"${PYTHON}" "${PROJECT}/scripts/materialize_string_v12_supplemental_selection.py" \
  --universe "${ROOT}/direct_physical_combined600_699_supplemental_universe_v1.tsv.zst" \
  --universe-receipt "${ROOT}/direct_physical_combined600_699_supplemental_universe_v1.receipt.json" \
  --selection-audit "${ROOT}/supplemental_cluster_pair_audit_expanded_v1.json" \
  --clusters "${ROOT}/cluster_assets_expanded_v1/clusters.tsv.zst" \
  --cluster-receipt "${ROOT}/direct_physical_combined700_cluster50_expanded_v1.receipt.json" \
  --tier1 "${ROOT}/direct_physical_combined700_cluster50_expanded_devclean_v1.tsv.zst" \
  --tier1-receipt "${ROOT}/direct_physical_combined700_cluster50_expanded_devclean_v1.receipt.json" \
  --excluded-clusters "${ROOT}/development_excluded_clusters_expanded_v1.txt" \
  --output "${OUTPUT}" \
  --receipt "${RECEIPT}" \
  --threads 6
